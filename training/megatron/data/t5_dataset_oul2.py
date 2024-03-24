# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 Style dataset."""

import collections

import numpy as np
import torch

from megatron import get_tokenizer, print_rank_0
from megatron.data.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping,
    get_finetune_samples_mapping,
    s_trans_multitask,
)
TASK_SAMPLE_RATIO=[1.0,0.0]
GLOBAL_NOISE_DENSITY=0.5
FIXED_S_NOISE = {"type": "S"}
RSX_MIX = {"type": "X", "mean_noise_span_lengths": [3,8,64], "corr_tokens_ratios":[0.3,0.4,0.3],"s_min_ratio":0.2,"s_max_ratio":0.3}

RECORDED = 0

class OUL2Datasets(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 max_num_samples, encoder_seq_length, decoder_seq_length,
                 ul2_type, seed):

        # Params to store.
        self.name = name
        self.seed = seed
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.ul2_type = ul2_type
        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.


        self.type_mapping, self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                                      data_prefix, max_num_samples,
                                                                      seed, name)
        # Vocab stuff.
        global tokenizer
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.unk_id = tokenizer.unk_id
        self.rtask_id = tokenizer.rtask_id
        self.stask_id = tokenizer.stask_id
        self.xtask_id = tokenizer.xtask_id
        self.sentinel_tokens = tokenizer.sentinel_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        # 0: span-corruption, 1: multi-task
        data_type = self.type_mapping[idx]
        data_index = self.samples_mapping[idx]
        src_tokens, tgt_tokens = self.indexed_dataset[(data_type, data_index)]

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.default_rng(seed=(self.seed + idx))
        s_np_rng = np.random.default_rng(seed=(self.seed + idx))
        if data_type == 1 or self.ul2_type == "sample":
            enc_in_length, dec_in_length = self.encoder_seq_length, self.decoder_seq_length
        else:
            enc_in_length, dec_in_length = None, None

        return build_training_sample(src_tokens,  tgt_tokens,
                                    self.vocab_id_list,
                                    self.vocab_id_to_token_dict,
                                    np_rng, s_np_rng,
                                    self.rtask_id, self.stask_id, self.xtask_id,
                                    self.eos_id, self.bos_id, self.pad_id,
                                    self.sentinel_tokens,
                                    multi_task = (data_type == 1), ul2_type=self.ul2_type,
                                    enc_in_length = enc_in_length, dec_in_length = dec_in_length,)


def build_training_sample(src_tokens,  tgt_tokens,
                        vocab_id_list, vocab_id_to_token_dict,
                        np_rng, s_np_rng,
                        R_token_id, S_token_id, X_token_id,
                        eos_token_id, bos_token_id, pad_token_id,
                        sentinel_tokens=None, expanded_input_length=2048,
                        enc_in_length=None, dec_in_length=None, 
                        multi_task = False, ul2_type = "sample"):
    global RECORDED
    if not multi_task:
        tokens = src_tokens
        if len(tokens) > expanded_input_length: 
            print(f" > warning: ul2's sentence_length = {len(tokens)} > expanded_input_length = {expanded_input_length}, trunct it")
            tokens = tokens[:expanded_input_length]
            truncated = 1 
        elif len(tokens) < expanded_input_length: 
            print(f" > warning: ul2's sentence_length = {len(tokens)} < expanded_input_length = {expanded_input_length}")

    if multi_task: 
        tgt_tokens = tgt_tokens.astype('int64')
        src_tokens = src_tokens.astype('int64')

        train_sample = s_trans_multitask(src_tokens, tgt_tokens, S_token_id, enc_in_length, dec_in_length, bos_token_id, eos_token_id, max(sentinel_tokens))
        # get 'text_enc', 'text_dec', 'labels' with padding and make masks
        train_sample = pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id)
        # train_sample['truncated'] = (len(src_tokens) > enc_in_length + 3) or (len(tgt_tokens) > dec_in_length + 2)
        assert train_sample['text_enc'].shape == (enc_in_length,)
        assert train_sample['text_dec'].shape == train_sample['labels'].shape == (dec_in_length,)
        train_sample["ul2_task_id"] = 1

        # src = train_sample["text_enc"]
        # tgt = train_sample["text_dec"]
        # print(tokenizer.detokenize(src.tolist()))
        # print(tokenizer.detokenize(tgt.tolist()))
        # import pdb
        # pdb.set_trace()
        return train_sample
    else:
        prefix_token_id = X_token_id if multi_task else S_token_id
        ul2_task_id = np_rng.choice([i+1 for i in range(2)], size=1, p=TASK_SAMPLE_RATIO)[0]
        if ul2_task_id == 1:

            tgt_tokens = tgt_tokens.astype('int64')
            src_tokens = src_tokens.astype('int64')
            train_sample = s_trans(np.array([src_tokens]), GLOBAL_NOISE_DENSITY, eos_token_id, bos_token_id, prefix_token_id, s_np_rng, max(sentinel_tokens))
            train_sample = pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id)
            assert train_sample['text_enc'].shape == (enc_in_length,)
            assert train_sample['text_dec'].shape == train_sample['labels'].shape == (dec_in_length,)
            train_sample['ul2_task_id'] = ul2_task_id     
            
        else:
            if RECORDED == 0:
                RECORDED = 1
            train_sample = t5_trans(np.array([src_tokens[:1700]]), eos_token_id,
                        bos_token_id, prefix_token_id,S_token_id,
                        max(sentinel_tokens) + 1, np_rng)

            train_sample = pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id)
            train_sample['ul2_task_id'] = ul2_task_id   
        
        return train_sample




def t5_trans(input_ids, eos_token_id, bos_token_id, prefix_token_id,S_token_id, max_sentinel_token, np_rng):
    train_sample = {}
    #print(input_ids.shape)
    input_ids_mask = np.asarray([random_spans_noise_mask(input_ids.shape[-1], np_rng)])
    labels_mask = ~input_ids_mask

    input_ids_sentinel = create_sentinel_ids(input_ids_mask.astype(np.int8), max_sentinel_token)
    labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), max_sentinel_token)

    if prefix_token_id < 0:
        train_sample["text_enc"] = filter_input_ids(input_ids, input_ids_sentinel, eos_token_id)[0]
    else:
        train_sample["text_enc"] = np.concatenate(\
                        (np.array([prefix_token_id]), \
                        filter_input_ids(input_ids, input_ids_sentinel, eos_token_id)[0]), axis = -1)
    train_sample["labels"] = filter_input_ids(input_ids, labels_sentinel, eos_token_id)[0]
    #import rpdb; rpdb.set_trace('0.0.0.0', 1024+torch.distributed.get_rank())
    train_sample["text_dec"] = np.concatenate((np.array([bos_token_id]), train_sample["labels"][:-1]), axis = -1)

    return train_sample

def add_last_sentinel_with_S(ids, S_token_id):
    sentinel_begin = tokenizer.sentinel_tokens_ids[0]
    sentinel_end = tokenizer.sentinel_tokens_ids[-1]
    to_be_replaced = sentinel_end
    replace_index = -1
    for idx in range(ids.shape[0]):
        if sentinel_begin <= ids[idx] <= sentinel_end and ids[idx] < to_be_replaced:
            sentinel_end = ids[idx]
            replace_index = idx
    ids = np.insert(ids, replace_index, S_token_id)
    return ids


def random_spans_noise_mask(length, np_rng):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * GLOBAL_NOISE_DENSITY))
    #import rpdb; rpdb.set_trace('0.0.0.0', 1024+torch.distributed.get_rank())
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    s_noise_tokens_min=int(np.round(num_noise_tokens * RSX_MIX["s_min_ratio"]))
    s_noise_tokens_max=int(np.round(num_noise_tokens * RSX_MIX["s_max_ratio"]))
    num_s_noise_tokens = int(np_rng.uniform(s_noise_tokens_min, s_noise_tokens_max + 1))
    num_noise_tokens_left=num_noise_tokens-num_s_noise_tokens
    num_rx_noise_types = len(RSX_MIX["mean_noise_span_lengths"])
    rx_noise=[]
    for i in range(num_rx_noise_types):
        num_noise_tokens_tmp = int(np.round(num_noise_tokens_left * RSX_MIX["corr_tokens_ratios"][i]))
        num_noise_spans_tmp = max(int(np.round(num_noise_tokens_tmp / RSX_MIX["mean_noise_span_lengths"][i])),1)
        rx_noise.append((num_noise_tokens_tmp,num_noise_spans_tmp))
    num_s_noise_tokens = num_noise_tokens-sum([i for i in rx_noise[0]])
    num_nonnoise_tokens = length - num_noise_tokens
    # avoid degeneracy by ensuring positive number of noise spans
    num_nonnoise_tokens = length - num_noise_tokens
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np_rng.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length
    noise_span_lengths = []
    for i in range(num_rx_noise_types):
        noise_span_lengths.extend(_random_segmentation(rx_noise[i][0],rx_noise[i][1]))
    noise_span_lengths = np.array(noise_span_lengths)
    np_rng.shuffle(noise_span_lengths)
    noise_span_lengths=noise_span_lengths.tolist()
    noise_span_lengths.append(num_s_noise_tokens)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, len(noise_span_lengths))
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [len(noise_span_lengths) * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise

def filter_input_ids(input_ids, sentinel_ids, eos_token_id):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    input_ids = np.concatenate(
        [input_ids, np.full((batch_size, 1), eos_token_id, dtype=np.int32)], axis=-1
    )
    return input_ids

def create_sentinel_ids(mask_indices, max_sentinel_token):
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids = np.where(sentinel_ids != 0, (max_sentinel_token - sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids

def pad_ids(input_ids, expected_length, pad_token_id):
    padding_length = expected_length - input_ids.shape[-1] if expected_length else 0
    assert padding_length >= 0, f"padding_length must > 0, expected_length is {expected_length}, input_ids_length is {input_ids.shape[-1]}"
    if padding_length:
        input_ids = np.concatenate((input_ids, np.array([pad_token_id] * padding_length)), axis = -1)
    return input_ids

def pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id):
    # train_sample['text_enc'] = train_sample['text_enc'][:enc_in_length]
    # train_sample['text_dec'] = train_sample['text_dec'][:dec_in_length]
    # train_sample['labels'] = train_sample['labels'][:dec_in_length]
    num_tokens_dec = train_sample['text_dec'].shape[-1]
    if dec_in_length is None:
        padding_length_dec = 0
    else:
        padding_length_dec = dec_in_length - num_tokens_dec

    train_sample['text_enc'] = pad_ids(train_sample['text_enc'], enc_in_length, pad_token_id)
    train_sample['text_dec'] = pad_ids(train_sample['text_dec'], dec_in_length, pad_token_id)
    train_sample['labels'] = pad_ids(train_sample['labels'], dec_in_length, -1)

    tokens_enc = train_sample['text_enc']
    tokens_dec_in = train_sample['text_dec']
    tokens_dec_in4mask = tokens_dec_in.copy()
    tokens_dec_in4mask[0] = 1
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in4mask, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in4mask, tokens_dec_in4mask)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in4mask)
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    train_sample['enc_mask'] = enc_mask
    train_sample['enc_dec_mask'] = enc_dec_mask
    train_sample['dec_mask'] = dec_mask
    train_sample['loss_mask'] = loss_mask
    
    return train_sample

def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask

def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (arange[None, ] <= arange[:, None])
    history_mask = history_mask.astype(np.int64)
    return history_mask

    
def s_trans(input_ids, noise_density, eos_token_id, bos_token_id, prefix_token_id, np_rng, sentinel_token):
    train_sample = {}
    length = input_ids.shape[-1]
    max_noise_token_num = min(round(noise_density * length),length)
    random_noise_num = max_noise_token_num
    random_unnoise_num = length - random_noise_num
    train_sample['text_enc'] = np.concatenate(\
                    (np.array([prefix_token_id]), \
                    input_ids[0][:random_unnoise_num],\
                    np.array([sentinel_token, eos_token_id]))
                    , axis = -1)
    train_sample['text_dec'] = np.concatenate(\
                    (np.array([bos_token_id, sentinel_token]), \
                    input_ids[0][random_unnoise_num:],\
                    ), axis = -1)
    train_sample['labels'] = np.concatenate(\
                    (np.array([sentinel_token]),\
                    input_ids[0][random_unnoise_num:],\
                    np.array([eos_token_id]),\
                    ), axis = -1)
    return train_sample

