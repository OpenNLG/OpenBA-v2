#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
# export OMP_NUM_THREADS=24

export CUDA_VISIBLE_DEVICES=7
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=17099
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1


LOAD_PATH="/data/megatron_ckpt/raw_model/stage4"
CHECKPOINT_PATH="/nvme/qd/ckpt/ckpt_temp"
TRAIN_DATA_PATH="/nvme/data_pretrained/valid_data_pretrained_spancorr"
VALID_DATA_PATH="/nvme/data_pretrained/valid_data_pretrained_spancorr"
TOKENIZER_PATH="/nvme/tokenizer/spiece.model"
TESNSORBOARD_PATH=$CHECKPOINT_PATH/tensorboard

mkdir -p ${TESNSORBOARD_PATH}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

T5_ARGS="
    --initial-loss-scale 65536 \
    --tensor-model-parallel-size 1 \
    --encoder-num-layers 2 \
    --decoder-num-layers 2 \
    --hidden-size 1024 \
    --num-attention-heads 20 \
    --kv-channels 128 \
    --ffn-hidden-size 2048 \
    --encoder-seq-length 1040 \
    --decoder-seq-length 1040 \
    --max-position-embeddings 3000 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --lr 0.00001 \
    --train-iters 320 \
    --lr-decay-iters 320 \
    --lr-decay-style cosine \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --min-lr 0.000005 \
    --weight-decay 0.1 \
    --lr-warmup-iters 0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --clip-grad 1.0 \
    --fp16 \
    --vocab-extra-ids 100 \
    --ul2-type sample \
    --pos-emb-type rotary \
    --mlp-type SwiGLU \
    --use-distributed-optimizer \
    --no-query-key-layer-scaling \
    --recompute-activations \
    --attention-softmax-in-fp32 \
    --finetune \
    --objectives O_UL2 
"

DATA_ARGS="
    --train-data-path $TRAIN_DATA_PATH \
    --valid-data-path $VALID_DATA_PATH \
    --tokenizer-model $TOKENIZER_PATH \
    --data-impl mmap \
    --num-workers 0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval  100 \
    --eval-interval 1 \
    --eval-iters 2 \
    --tensorboard-dir $TESNSORBOARD_PATH \
"

torchrun $DISTRIBUTED_ARGS pretrain_t5.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH | tee -a $CHECKPOINT_PATH/${NODE_RANK}.log