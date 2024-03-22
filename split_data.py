from megatron.data.indexed_dataset import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    index_file_path,
    data_file_path,
)
import numpy as np
import torch


# data_prefix = "/data/pile_bin/val_spancorr"
data_prefix = "/data/chinese_bin/all"
skip_warmup = True
# output_file_prefix = "/data/pile_bin/val_spancorr"
output_file_prefix = "/data/chinese_bin/all"
split_part = 4

indexed_dataset = MMapIndexedDataset(data_prefix, skip_warmup)
num_samples = len(indexed_dataset)
print(num_samples)

builder = [
    MMapIndexedDatasetBuilder(data_file_path(f"{output_file_prefix}_{i}"), dtype=np.int32) for i in range(split_part)
]

np.random.seed(423)
for i in range(num_samples):
    idx = np.random.choice(split_part, 1)[0]
    data = indexed_dataset[(0, i)]
    builder[idx].add_item(
        source_tensor=torch.IntTensor(data[0]), 
        target_tensor=torch.IntTensor(data[1]),
        task="span-corruption"
    )

for i in range(split_part):
    builder[i].finalize(index_file_path(f"{output_file_prefix}_{i}"))

# builder.add_item(indexed_dataset[(0, 0)])
# tokenizer = 
# import pdb; pdb.set_trace()
