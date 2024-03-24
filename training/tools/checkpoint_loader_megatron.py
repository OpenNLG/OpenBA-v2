import json
import os
import sys
import types
import torch
import collections

def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')
    group.add_argument('--merged-ckp-name', type=str, required=True,
                       help='Specify the merged ckp name')
    group.add_argument('--tokenizer-model', type=str, default=None,
                        help='specify the tokenizer path, which decide the vocab size')
    group.add_argument('--vocab-extra-ids', type=int, default=100,
                        help='specify the vocab size')
    group.add_argument("--encoder-num-layers", type=int, default=8,
                        help='specify the number of encoder layers')
    group.add_argument("--decoder-num-layers", type=int, default=24,
                        help='specify the number of decoder layers')
    group.add_argument("--num-layers", type=int, default=None,
                        help='specify the number of encoder/decoder layers')
    group.add_argument("--encoder-seq-length", type=int, default=1024,
                        help='specify the length of encoder input sequence')
    group.add_argument("--decoder-seq-length", type=int, default=512,
                        help='specify the length of decoder input sequence')
    group.add_argument("--seq-length", type=int, default=512,
                        help='specify the length of encoder/decoder input sequence')
    group.add_argument("--pos-emb-type", type=str, default='rotary',
                        help='specify the positional embedding type')
    group.add_argument("--mlp-type", type=str, default='SwiGLU',
                        help='specify the mlp module type')

def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_args, set_global_variables
        from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs = load_args_from_checkpoint(margs)
    
    # TODO: change the loading args here
    margs.encoder_num_layers = 8
    margs.decoder_num_layers = 24
    margs.encoder_seq_length = 1024
    margs.decoder_seq_length = 512
    margs.num_layers = None
    margs.seq_length = None
    margs.pos_emb_type = 'rotary'
    margs.mlp_type = 'SwiGLU'
    margs.tokenizer_model = "/data/Megatron-DeepSpeed/data-file/multilingual-spiece.model"

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    
    margs = validate_args(margs)
    
    def check_for_arg(arg_name):
        if getattr(margs, arg_name, None) is None:
            print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
            print(f"Arguments: {margs}")
            queue.put("exit")
            exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('params_dtype')

    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'T5':
        from pretrain_t5 import model_provider
        margs.model_type = ModelType.encoder_and_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = None
    consumed_valid_samples = None
    
    def get_models(count, dtype, pre_process, post_process):
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        models = []
        for rank in range(count):
            parallel_state.set_tensor_model_parallel_rank(rank)
            model_ = [model_provider(pre_process, post_process).to(dtype)]
            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            load_checkpoint(model_, None, None)
            assert(len(model_) == 1)
            model_ = model_[0]
            if consumed_train_samples is not None:
                assert(margs.consumed_train_samples == consumed_train_samples)
            else:
                consumed_train_samples = margs.consumed_train_samples
            if consumed_valid_samples is not None:
                assert(margs.consumed_valid_samples == consumed_valid_samples)
            else:
                consumed_valid_samples = margs.consumed_valid_samples
            models.append(model_)
        return models

    if margs.num_layers_per_virtual_pipeline_stage is not None:
        print("Model with an interleaved pipeline schedule are not yet supported.")
        queue.put("exit")
        exit(1)

    set_global_variables(margs)
    
    margs.tensor_model_parallel_size = 4   # TODO
    margs.pipeline_model_parallel_size = 1
    
    from megatron.core import parallel_state, tensor_parallel
    parallel_state.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    parallel_state.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vocab = json.load(open(args.vocab_file))
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            queue.put("exit")
            exit(1)
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by

    # Get first pipe stage
    parallel_state.set_pipeline_model_parallel_rank(0)
    post_process = pp_size == 1
    
    models = get_models(tp_size, md.params_dtype, True, post_process)
    
    #=========================================
    #===== this is new code for merge ckp ====
    #=========================================
    
    new_state_dict = dict()
    new_state_dict['model'] = {}
    new_state_dict['model']['language_model'] = {}
    new_state_dict['model']['language_model']['embedding'] = {}
    new_state_dict['model']['language_model']['embedding']['word_embeddings'] = collections.OrderedDict()
    new_state_dict['model']['language_model']['embedding']['word_embeddings']['weight'] = torch.cat(
                            [models[tp_rank].language_model.embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
                            dim = 0
                        )
    
    new_state_dict['model']['lm_head'] = collections.OrderedDict()
    new_state_dict['model']['lm_head']['bias'] = torch.cat(
                                                    [models[tp_rank].lm_head.bias for tp_rank in range(tp_size)],
                                                    dim = 0
                                                )
    new_state_dict['model']['language_model']['encoder'] = collections.OrderedDict()
    new_state_dict['model']['language_model']['decoder'] = collections.OrderedDict()
    
    # encoder
    encoder_non_parallel_state_names = [
        "input_layernorm.weight", 
        "input_layernorm.bias", 
        "self_attention.dense.bias",
        "post_attention_layernorm.weight",
        "post_attention_layernorm.bias",
        "self_attention.rotary_embed.freqs"
    ]
    encoder_parallel_state_names = [
        "self_attention.query_key_value.weight", 
        "self_attention.query_key_value.bias",
        "self_attention.dense.weight",     
        "layer.mlp.w1.weight",
        "layer.mlp.w2.weight",
        "layer.mlp.w3.weight",
    ]
    
    print("======> begin to merge encoder parameters")
    # save encoder parameters
    for layer_num in range(len(models[0].language_model.encoder.layers)):
        cur_layer_params = {}
        layer = models[0].language_model.encoder.layers[layer_num]
        # save non-parallel-parameters for each layer
        cur_layer_params["input_layernorm.weight"] = layer.input_layernorm.weight.data
        cur_layer_params["input_layernorm.bias"] = layer.input_layernorm.bias.data
        cur_layer_params["self_attention.dense.bias"] = layer.self_attention.dense.bias.data
        cur_layer_params["post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.data
        cur_layer_params["post_attention_layernorm.bias"] = layer.post_attention_layernorm.bias.data
        cur_layer_params["self_attention.rotary_embed.freqs"] = layer.self_attention.rotary_embed.freqs
        
        # save parallel-paramters for each layer
        qkv_weight, qkv_bias, dense_weight = [], [], []
        mlp_l0_weight, mlp_l1_weight, mlp_l2_weight = [], [], []
        for tp_rank, model in enumerate(models):
            layer = model.language_model.encoder.layers[layer_num]
            qkv_weight.append(layer.self_attention.query_key_value.weight.data)
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            dense_weight.append(layer.self_attention.dense.weight.data)
            mlp_l0_weight.append(layer.mlp.w1.weight.data)
            mlp_l1_weight.append(layer.mlp.w2.weight.data)
            mlp_l2_weight.append(layer.mlp.w3.weight.data)
        cur_layer_params["self_attention.query_key_value.weight"] = torch.cat(qkv_weight, dim=0)
        cur_layer_params["self_attention.query_key_value.bias"] = torch.cat(qkv_bias, dim=0)
        cur_layer_params["self_attention.dense.weight"] = torch.cat(dense_weight, dim=1)
        cur_layer_params["mlp.w1.weight"] = torch.cat(mlp_l0_weight, dim=0)
        cur_layer_params["mlp.w2.weight"] = torch.cat(mlp_l1_weight, dim=1)
        cur_layer_params["mlp.w3.weight"] = torch.cat(mlp_l2_weight, dim=0)

        # save parameters
        for module_name, module_params in cur_layer_params.items():
            new_state_dict['model']['language_model']['encoder'][f'layers.{layer_num}.{module_name}'] = module_params
    
    new_state_dict['model']['language_model']['encoder']['final_layernorm.weight'] \
        = models[0].language_model.encoder.final_layernorm.weight

    new_state_dict['model']['language_model']['encoder']['final_layernorm.bias'] \
        = models[0].language_model.encoder.final_layernorm.bias
    
    print("======> begin to merge decoder parameters")
    
    # save decoder parameters
    for layer_num in range(len(models[0].language_model.decoder.layers)):
        cur_layer_params = {}
        layer = models[0].language_model.decoder.layers[layer_num]
        # save non-parallel-parameters for each layer
        cur_layer_params["input_layernorm.weight"] = layer.input_layernorm.weight.data
        cur_layer_params["input_layernorm.bias"] = layer.input_layernorm.bias.data
        cur_layer_params["post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.data
        cur_layer_params["post_inter_attention_layernorm.weight"] = layer.post_inter_attention_layernorm.weight.data
        cur_layer_params["post_attention_layernorm.bias"] = layer.post_attention_layernorm.bias.data
        cur_layer_params["post_inter_attention_layernorm.bias"] = layer.post_inter_attention_layernorm.bias.data
        cur_layer_params["self_attention.dense.bias"] = layer.self_attention.dense.bias.data
        cur_layer_params["inter_attention.dense.bias"] = layer.self_attention.dense.bias.data
        cur_layer_params["self_attention.rotary_embed.freqs"] = layer.self_attention.rotary_embed.freqs
        
        # save parallel-paramters for each layer
        qkv_weight, qkv_bias, dense_weight = [], [], []
        mlp_l0_weight, mlp_l1_weight, mlp_l2_weight = [], [], []
        ca_q_weight, ca_q_bias, ca_kv_weight, ca_kv_bias = [], [], [], []
        ca_dense_weight = []
       
        for tp_rank, model in enumerate(models):
            layer = model.language_model.decoder.layers[layer_num]
            qkv_weight.append(layer.self_attention.query_key_value.weight.data)
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            dense_weight.append(layer.self_attention.dense.weight.data)
            ca_q_weight.append(layer.inter_attention.query.weight.data)
            ca_q_bias.append(layer.inter_attention.query.bias.data)
            ca_kv_weight.append(layer.inter_attention.key_value.weight)
            ca_kv_bias.append(layer.inter_attention.key_value.bias)
            ca_dense_weight.append(layer.inter_attention.dense.weight)
            mlp_l0_weight.append(layer.mlp.w1.weight.data)
            mlp_l1_weight.append(layer.mlp.w2.weight.data)
            mlp_l2_weight.append(layer.mlp.w3.weight.data)
        cur_layer_params["self_attention.query_key_value.weight"] = torch.cat(qkv_weight, dim=0)
        cur_layer_params["self_attention.query_key_value.bias"] = torch.cat(qkv_bias, dim=0)
        cur_layer_params["self_attention.dense.weight"] = torch.cat(dense_weight, dim=1)
        cur_layer_params["inter_attention.query.weight"] = torch.cat(ca_q_weight, dim=0)
        cur_layer_params["inter_attention.query.bias"] = torch.cat(ca_q_bias, dim=0)
        cur_layer_params["inter_attention.key_value.weight"] = torch.cat(ca_kv_weight, dim=0)
        cur_layer_params["inter_attention.key_value.bias"] = torch.cat(ca_kv_bias, dim=0)
        cur_layer_params["inter_attention.dense.weight"] = torch.cat(ca_dense_weight, dim=1)
        cur_layer_params["mlp.w1.weight"] = torch.cat(mlp_l0_weight, dim=0)
        cur_layer_params["mlp.w2.weight"] = torch.cat(mlp_l1_weight, dim=1)
        cur_layer_params["mlp.w3.weight"] = torch.cat(mlp_l2_weight, dim=0)
        
        # save parameters
        for module_name, module_params in cur_layer_params.items():
            new_state_dict['model']['language_model']['decoder'][f'layers.{layer_num}.{module_name}'] = module_params
    
    new_state_dict['model']['language_model']['decoder']['final_layernorm.weight'] \
        = models[0].language_model.decoder.final_layernorm.weight

    new_state_dict['model']['language_model']['decoder']['final_layernorm.bias'] \
        = models[0].language_model.decoder.final_layernorm.bias

    print("====> begin to save the merged state dict <====")
    print(f"====> saving model to {args.merged_ckp_name} <====")
    
    torch.save(new_state_dict, args.merged_ckp_name)
    
    print("====> merge end <====")
    
    exit()
    

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
