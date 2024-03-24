#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
# export OMP_NUM_THREADS=24

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=17099
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

LOAD_PATH="/data/megatron_ckpt/raw_model/ori_model_parallel4"
CHECKPOINT_PATH="/data/megatron_ckpt/raw_model/ori_6B/1234"
TRAIN_DATA_PATH="/data/pretrained_data_ori/data_stage3/data_stage3_46B"
VALID_DATA_PATH="/data/pretrained_data_ori/valid/valid_data"
TOKENIZER_PATH="/data/tokenizer/spiece.model"
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
    --seed 1234 \
    --tensor-model-parallel-size 4 \
    --encoder-num-layers 12 \
    --decoder-num-layers 36 \
    --hidden-size 4096 \
    --num-attention-heads 40 \
    --kv-channels 128 \
    --ffn-hidden-size 11008 \
    --encoder-seq-length 570 \
    --decoder-seq-length 381 \
    --max-position-embeddings 768 \
    --micro-batch-size 16 \
    --global-batch-size 1024 \
    --lr 0.00002 \
    --train-iters 22500 \
    --lr-decay-iters 22500 \
    --lr-decay-style cosine \
    --min-lr 0.00001 \
    --weight-decay 0.1 \
    --lr-warmup-iters 1000 \
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
    --finetune
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
    --save-interval  500 \
    --eval-interval 500 \
    --eval-iters 5 \
    --tensorboard-dir $TESNSORBOARD_PATH \
"

PRUNE_ARGS="
    --hidden_size_remain 2560 \
    --ffn_hidden_size_remain 6912 \
    --num_attention_heads_remain 20 \
    --drop_encoder_layers 3,6 \
    --drop_decoder_layers 6,12 \
"

torchrun $DISTRIBUTED_ARGS prune_models.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $PRUNE_ARGS \
    --is_prune true \
    --distributed-backend nccl \
    --dynamic_batch_loading \
    --save $CHECKPOINT_PATH \
    --load $LOAD_PATH | tee -a $CHECKPOINT_PATH/${NODE_RANK}.log