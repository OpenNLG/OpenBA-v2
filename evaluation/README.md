# Evaluation

## Overview

We provide test code and scripts for different evaluation tasks to facilitate the reproduction of results, including implementations for specific datasets and adaptation based on the lm-evaluation-harness framework.
Evaluation tasks included:

- World knowledge and math knowledge tasks: MMLU,BBH,GSM8K,MATH
- Chinese tasks: C-EVAL,CMMLU
- Commonsense & reading comprehension tasks: SciQ,PIQA,ARC-E,ARC-C,LogiQA,BoolQ

## Usage
### Implementations for specific datasets
prepare environment:
```bash
pip install -r requirements.txt
```

Evaluation single dataset: ARC for example

```
cd ARC
bash eval.sh 
```



### lm-evaluation-harness

#### Install

Install the lm-evaluation-harness package that is adapted for the OpenBA evaluation.:

```bash
pip install transformers==4.31.0
cd lm-evaluation-harness
pip install -e .
```

Evaluation all datasets：

```shell
 bash script/eval_harness.sh
```

Evaluation single dataset such as 'sciq':

```shell
lm_eval --model hf \
	--model_args pretrained=<model_name>,backend=seq2seq,trust_remote_code=True,max_length=1040 \
	--tasks   'sciq' \
	--batch_size 8 \
	--use_cache <cache_path> \
	--write_out --output_path <write_out_path> \
	--log_samples \
```
