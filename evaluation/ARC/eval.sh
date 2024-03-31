#!/bin/bash
name=arc_ABCD_0shot
model="model_name_or_path"
input_folder="data/25shot"
current_output_folder="outputs/${model_name}_${name}"

max_length=1024
decoder_max_length=1024
export MASTER_ADDR=localhost
export MASTER_PORT=1000
current_template="make_ABCD_input_0_shot"

export CUDA_VISIBLE_DEVICES=0
python -u main.py \
    --model-path $model \
    --max-length $max_length \
    --input-folder $input_folder \
    --output-folder $current_output_folder \
    --template-type $current_template \
    --add-prefix \
    --decoder-max-length $decoder_max_length \
    --ptoken S 