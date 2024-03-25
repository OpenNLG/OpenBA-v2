
import sys
import os
import sentencepiece as spm
import re
from copy import deepcopy
import torch






state_dict_path = "/nvme/megatron_ckpt/MODEL_CKPT/megatron_ckpt/raw_model/flan3/iter_0026000/hf/pytorch_model.bin"

save_dict_path = "/nvme/megatron_ckpt/MODEL_CKPT/megatron_ckpt/raw_model/flan3/prune_emb_120k_26000/pytorch_model.bin"

old_spm_path = "/nvme/tokenizer/spiece.model"
new_spm_path = "/nvme/tokenizer/spiece120k.model"

ckpt = torch.load(state_dict_path,map_location="cpu")

# embedding = ckpt["model"]["language_model"]["embedding"]['word_embeddings']["weight"]



old_spm = spm.SentencePieceProcessor(old_spm_path)
new_spm = spm.SentencePieceProcessor(new_spm_path)

old_vocab_size = old_spm.vocab_size()
new_vocab_size = new_spm.vocab_size()

ids = []
for i in range(new_vocab_size):
    token = new_spm.IdToPiece(i)
    old_id = old_spm.PieceToId(token)
    ids.append(old_id)


def _vocab_size_with_padding(orig_vocab_size, make_vocab_size_divisible_by=128, tensor_model_parallel_size=1):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * \
        tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    return after

speicial_token_id = [i for i in range(old_vocab_size, old_vocab_size + 103)] # extra ids 100 <R>,<S>,<X>

ids = ids + speicial_token_id
dummy_token_num = _vocab_size_with_padding(new_vocab_size + 103) - len(ids)
if dummy_token_num > 0:
    for i in range(dummy_token_num):
        ids.append(old_vocab_size)

# import pdb
# pdb.set_trace()

def prune_param(state_dict_path,  save_dict_path, ids):
    state_dict = torch.load(state_dict_path)
    print("## successfully load the model ##")
    new_dict = deepcopy(state_dict)
    ids_tensor = torch.LongTensor(ids)
    embedding = state_dict["shared_embedding.weight"]
    new_embedding = torch.index_select(embedding, 0, ids_tensor.to(embedding.device))
    new_dict["shared_embedding.weight"] = new_embedding

    lm_head = state_dict["lm_head.weight"]
    new_lm_head = torch.index_select(lm_head, 0, ids_tensor.to(lm_head.device))
    new_dict["lm_head.weight"] = new_lm_head


    bias = state_dict["lm_head.bias"]
    new_bias = torch.index_select(bias, 0, ids_tensor.to(bias.device))
    new_dict["lm_head.bias"] = new_bias
    
    print("## saving the dict ##")

    torch.save(new_dict,save_dict_path)


prune_param(state_dict_path, save_dict_path, ids)
