import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

new_vocab_num = 25000

vocab_file_path = "/home/amax/qd/emb_prune/old/ori_freq/token_freq.txt"
openba_sp_model_file = "/nvme/tokenizer/spiece.model"
new_sp_model_file = "/nvme/tokenizer/spiece25k.model"

with open(vocab_file_path,'r') as f:
    data = f.readlines()
valid_ids = []
valid_tokens = []
for i in range(new_vocab_num):
    line = data[i]
    token = data[i].split(" ")[0]
    token_id = int(data[i].split(" ")[1])
    valid_ids.append(token_id)
    valid_tokens.append(token)



openba_spm_model = sp_pb2_model.ModelProto()



openba_spm_model = sp_pb2_model.ModelProto()
openba_spm_model.ParseFromString(open(openba_sp_model_file, "rb").read())

token_dict = set(valid_tokens)

new_sp_model = sp_pb2_model.ModelProto()

new_sp_model.ParseFromString(open(openba_sp_model_file, "rb").read())
new_sp_model.pieces.clear()

for item in openba_spm_model.pieces:
    if item.piece in token_dict:
        new_sp_model.pieces.append(item)
    elif item.type != 1:
        new_sp_model.pieces.append(item)
print(len(new_sp_model.pieces))



with open(new_sp_model_file, 'wb') as f:
    f.write(new_sp_model.SerializeToString())