import numpy as np
from megatron.data.indexed_dataset import (
    MMapIndexedDatasetBuilder,
    make_dataset,
    index_file_path,
    data_file_path,
)
import numpy as np
import torch


data_prefix = ""
output_prefix = ""
builder = MMapIndexedDatasetBuilder(output_prefix, dtype=np.int32)
dataset = make_dataset(data_prefix, impl='mmap', skip_warmup=True)
import pdb; pdb.set_trace()
