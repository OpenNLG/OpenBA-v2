#!/bin/bash

# 设置.json文件的目录路径
json_dir="/data/v2_data/flan/en"

# 在指定目录循环遍历所有.json文件
for file in "$json_dir"/*.json*; do
    # 提取不带扩展名的文件名
    filename=$(basename "$file" .json)

    # 运行Python脚本，并使用当前文件的参数
    python tools/finetune_data_v2.py \
        --json-file "${file}" \
        --input-column inputs \
        --target-column targets \
        --tokenizer-model /data/tokenizer/spiece.model \
        --vocab_extra_ids 100 \
        --output-prefix "${json_dir}/bins/${filename}" \
        --dataset-impl mmap \
        --workers 16 \
        --log-interval 10 \
        --chunk-size 1
done
