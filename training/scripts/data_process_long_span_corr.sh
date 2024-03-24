python tools/preprocess_data_pretrain.py \
    --json-file /nvme/data_pretrained/pipe/valid_data.json \
    --json-key text \
    --group-size 2048 \
    \
    --tokenizer-model /home/amax/qd/emb_prune/tokenizer/newspiece2.model \
    --vocab_extra_ids 100 \
    \
    --output-prefix /nvme/data_pretrained/valid_data_pretrained \
    --dataset-impl mmap \
    --batch-size 1000 \
    --workers 4 \
    --chunk-size 1 \
    --log-interval 10 \
