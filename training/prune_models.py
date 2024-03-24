from functools import partial

import torch
import os
from megatron import (
    get_args,
    get_timers,
    print_rank_0
)
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import T5Model
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron import get_tokenizer



# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import GPTModel
from megatron.core.enums import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.utils import report_memory
from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron.utils import throughput_calculator
from megatron.checkpointing import _load_base_checkpoint
from megatron.training import prune

def model_provider(pre_process=True, post_process=True,
                   add_encoder=True, add_decoder=True):
    """Build the model."""

    print_rank_0('building T5 model ...')
    model = T5Model(num_tokentypes=0,
                    parallel_output=True,
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder)
    return model

if __name__ == "__main__":
    prune(model_provider=model_provider, model_type=ModelType.encoder_and_decoder,args_defaults={'tokenizer_type': 'SentencePieceTokenizer'})


