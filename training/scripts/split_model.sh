python tools/checkpoint_split_model.py \
    --load_path /data/megatron_ckpt/raw_model/ori_6B/1234/iter_0000001 \
    --save_path /data/megatron_ckpt/raw_model/ori_6B/1234/parallel1/iter_0000001 \
    --target_tensor_model_parallel_size 1 \
    --print-checkpoint-structure \
