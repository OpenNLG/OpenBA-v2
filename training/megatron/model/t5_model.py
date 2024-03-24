# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 model."""

import torch
import os
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits, get_language_model
from megatron.model.transformer import LayerNorm
from megatron.model.utils import (
    openai_gelu,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
    get_random_index,
    get_random_mask,
    get_tensor_per_partition
)
from .module import MegatronModule
from .language_model import TransformerLanguageModel
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_global_memory_buffer,
)
def t5_extended_attention_mask(attention_mask_list):

    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def t5_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class T5LMHead(MegatronModule):
    """Masked LM head for T5

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output):
        super(T5LMHead, self).__init__()

        args = get_args()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, hidden_states, word_embeddings_weight):
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output


class T5Model(MegatronModule):
    """T5 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 add_encoder=True,
                 add_decoder=True):
        super(T5Model, self).__init__()
        args = get_args()
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.pos_emb_type = args.pos_emb_type

        self.l0module = None

        self.device = torch.cuda.current_device()


        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pos_emb_type=self.pos_emb_type)
        
        if args.is_prune:
            self.prune_zs = self.init_prune_zs(args)
        
        self.initialize_word_embeddings(init_method_normal)

        if self.post_process and self.add_decoder:
            self.lm_head = T5LMHead(
                self.word_embeddings_weight().size(0),
                parallel_output)
            self._lm_head_key = 'lm_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attn_mask,
                decoder_attn_mask, encoder_decoder_attn_mask,
                tokentype_ids=None, lm_labels=None, enc_hidden_states=None, output_hidden_states=False, inference_params=None):
        
        if self.l0module is not None:
            zs = self.l0module()
        else:
            zs = None
        # Converting the attention masks to proper parameter settings
        encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = t5_extended_attention_mask(
            [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask])
        if self.pos_emb_type == "learn":
            encoder_position_ids = t5_position_ids(encoder_input_ids)
            decoder_position_ids = t5_position_ids(decoder_input_ids)
        else:
            encoder_position_ids = None
            decoder_position_ids = None
        assert self.language_model.__class__ == TransformerLanguageModel
        lm_output = self.language_model(encoder_input_ids,
                                        encoder_position_ids,
                                        encoder_attn_mask,
                                        decoder_input_ids,
                                        decoder_position_ids,
                                        decoder_attn_mask,
                                        encoder_decoder_attn_mask,
                                        tokentype_ids=tokentype_ids,
                                        enc_hidden_states=enc_hidden_states, 
                                        zs=zs)
        

        if self.post_process and self.add_decoder:
            decoder_output, encoder_output = lm_output
            if output_hidden_states:
                return encoder_output, decoder_output
            # Output. [s, b, h]

            lm_logits = self.lm_head(decoder_output,
                                     self.word_embeddings_weight())

            if lm_labels is None:
                # [s b h] => [b s h]
                return lm_logits.transpose(0,1).contiguous()
            else:
                # [b s] => [s b]
                lm_labels = lm_labels.transpose(0,1).contiguous()
                if self.fp16_lm_cross_entropy:
                    assert lm_logits.dtype == torch.half
                    lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
                else:
                    lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(),
                                                                                lm_labels)
                # [s b] => [b s]
                lm_loss = lm_loss.transpose(0,1).contiguous()
     

            return {"lm_loss":lm_loss} 
        
        elif self.add_decoder and not self.add_encoder:
            decoder_output, encoder_output = lm_output

            return decoder_output
        else:
            encoder_output = lm_output
            return encoder_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process and self.add_decoder:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
         # Save word_embeddings.
        if self.post_process and not self.pre_process and self.add_decoder:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        if self.l0module is not None:
            state_dict_[self._l0module_key] \
                = self.l0module.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process and self.add_decoder:
            self.lm_head.load_state_dict(state_dict[self._lm_head_key],
                                         strict=strict)
        # Load word embeddings.
        if self.post_process and not self.pre_process and self.add_decoder:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        
        if self.l0module is not None:
            if self._l0module_key in state_dict:
                self.l0module.load_state_dict(
                    state_dict[self._l0module_key], strict=False)

    def init_prune_zs(self, args):
        zs = {}
        hidden_size = args.hidden_size
        hidden_size_remain = args.hidden_size_remain
        num_attention_heads = args.num_attention_heads 
        num_attention_heads_remain = args.num_attention_heads_remain 
        
        ffn_hidden_size = args.ffn_hidden_size
        ffn_hidden_size_remain = args.ffn_hidden_size_remain

        hidden_mask = get_random_mask(hidden_size,hidden_size_remain)
        hidden_mask.requires_grad = False
        zs["hidden_mask"] = hidden_mask

        hidden_index = torch.where(hidden_mask==True)[0]
        hidden_index.requires_grad = False
        zs["hidden_index"] = hidden_index
        head_masks = []
        head_indexes = []
        intermediate_masks = []
        intermediate_indexes = []

        layer_num = args.encoder_num_layers + args.decoder_num_layers

        parallel_size = args.tensor_model_parallel_size
        model_parallel_world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        # assert model_parallel_world_size == 1, "model parallel num must be 1"
        for i in range(layer_num):

            head_mask = get_random_mask(num_attention_heads, num_attention_heads_remain)
            head_mask_per_partition = get_tensor_per_partition(head_mask,rank,model_parallel_world_size)
            head_masks.append(head_mask_per_partition)
            head_indexes.append(torch.where(head_mask_per_partition==True)[0])

            intermediate_mask = get_random_mask(ffn_hidden_size,ffn_hidden_size_remain)
            intermediate_mask_per_partition = get_tensor_per_partition(intermediate_mask,rank,model_parallel_world_size)
            intermediate_masks.append(intermediate_mask_per_partition)
            intermediate_indexes.append(torch.where(intermediate_mask_per_partition==True)[0])

        for tensor in head_masks:
            tensor.requires_grad = False
        for tensor in intermediate_masks:
            tensor.requires_grad = False
        for tensor in head_indexes:
            tensor.requires_grad = False
        for tensor in intermediate_indexes:
            tensor.requires_grad = False    

        zs["head_masks"] = head_masks
        zs["intermediate_masks"] = intermediate_masks

        zs["head_indexes"] = head_indexes 
        zs["intermediate_indexes"] = intermediate_indexes
        save_path = os.path.join(args.save,f"rank{rank}.pt")
        torch.save(zs,save_path)
        return zs
    
    def prune(self, args):

        self.language_model.prune(args,self.prune_zs)


