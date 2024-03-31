import argparse
import json
import torch
import os
import glob
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import template
import numpy as np
import os
import torch
import torch.nn as nn
import numpy as np

def post_process_ABCD(out_list, test_inputs):
    golds = [i['data']['ans'] for i in test_inputs]
    encs = [i[1] for i in out_list]
    decs = [i[2] for i in out_list]
    preds = [i[0] for i in out_list]
    assert len(preds) == len(decs) == len(golds) == len(decs), str(len(preds), len(decs), len(golds), len(decs))
    data2write = [json.dumps({'enc':enc, 'dec':dec, 'gold':gold, 'pred':pred}) + '\n' for enc, dec, gold, pred in zip(encs, decs, golds, preds)]
    right = sum([1 for i, j in zip(golds, preds) if i == j])
    cnt = len(golds)
    ABCD_rate = [sum([1 for i in preds if i == j]) / cnt for j in ["A", "B", "C", "D", 'E']]
    return ABCD_rate, cnt, right, data2write

def solve_ABCD(model, tokenizer, input_text, opt_size, decoder_input_text, args, rank, world_size):
    
    
    input_ids = tokenizer(input_text, return_tensors='pt', max_length = args.max_length, truncation=True).input_ids.chunk(world_size)
    decoder_input_ids= tokenizer(decoder_input_text, return_tensors='pt', max_length = args.decoder_max_length - 1, truncation=True).input_ids
    decoder_input_ids = model.module._shift_right(decoder_input_ids).chunk(world_size)
    chunk_size = len(decoder_input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids[rank % chunk_size].cuda(rank), \
                       decoder_input_ids=decoder_input_ids[rank % chunk_size].cuda(rank)).logits[:,-1,:].contiguous()
        
    gather_list = [torch.zeros_like(logits) for _ in range(world_size)] if rank == 0 else []
    dist.gather(logits, gather_list=gather_list, dst=0)
    if rank == 0:
        logits = torch.cat(gather_list, dim=0)

    for single_logits in logits:
        probs = (
            torch.tensor( [ single_logits[tokenizer.convert_tokens_to_ids(chr(65 + i))] for i in range(opt_size) ] ).detach().cpu().numpy()
        )
        # print(probs)
        pred = {0: "A", 1: "B", 2: "C", 3: "D", 4:'E'}[np.argmax(probs)]
    torch.distributed.barrier()
    return pred

def main(rank, world_size, args):
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, max_length=1024)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, trust_remote_code=True).half().cuda(rank)
    model = torch.nn.DataParallel(model, device_ids=[rank])
    model.eval()

    if torch.distributed.get_rank() == 0:
        if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
    
    torch.distributed.barrier()
    
    if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)

    json_input_paths = glob.glob(os.path.join(args.input_folder, '*.json'))
    json_file_names = [os.path.basename(i) for i in json_input_paths]
    json_output_paths = [os.path.join(args.output_folder, i) for i in json_file_names]

    make_input = getattr(template, args.template_type)
    solve = solve_ABCD
    post_process = post_process_ABCD

    all_cnt, all_right = 0, 0
    
    for file_name, input_path, output_path in zip(json_file_names, json_input_paths, json_output_paths):
        out_list = []
        print(file_name, input_path, output_path)
        
        with open(input_path, 'r') as fr, open(output_path, 'w') as fw:
            test_inputs = json.load(fr)
            print(len(test_inputs))
            
            for idx in tqdm(range(0, len(test_inputs))):    
                input_data = test_inputs[idx]
                candidates, decoder_input_text = make_input(file_name, input_data)
        
                input_text = template.choose_longest_input(candidates, args.max_length, tokenizer, args.add_prefix)
                # print("input_text:"+input_text)
                if args.add_prefix: 
                    input_text = f"<{args.ptoken}> " + input_text + " <extra_id_0>"
                    decoder_input_text = f"<extra_id_0> " + decoder_input_text
                
                pred = solve(model, tokenizer, input_text, input_data['data']['opt_num'], decoder_input_text, args, rank, world_size)
                # print("pred: "+pred)
                out_list.append((pred, input_text, decoder_input_text))
                
            if torch.distributed.get_rank() == 0:
                ABCD_rate, cnt, right, data2write = post_process(out_list, test_inputs)
                print(file_name.replace('_test.json', '').replace('_', ' '), ': \n', f"acc: {(right / cnt)*100:.2f}%")
                for label, rate in zip(['A', 'B', 'C', 'D', 'E'], ABCD_rate): print(f"{label}: {rate*100:.2f}%", end = '|')
                print('\n' + '-' * 30)
                all_cnt, all_right = all_cnt + cnt, all_right + right
                for line in data2write:fw.write(line)

    if torch.distributed.get_rank() == 0:
        print(f"all acc: {(all_right / all_cnt)*100:.2f}%", )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default="/public/home/ljt/LLM/wpz/LEO_mmlu/4shot",)
    parser.add_argument("--output-folder", type=str, default="tmp",)
    parser.add_argument("--model-path", type=str, default="/public/home/ljt/LEO/checkpoint/14b_flan_new/iter_0003000_hf",)
    parser.add_argument("--max-length", type=int, default=512,)
    parser.add_argument("--decoder-max-length", type=int, default=128,)
    parser.add_argument("--padding", type=str, default="longest",)
    parser.add_argument("--add-prefix", action='store_true')
    parser.add_argument("--template-type", type = str, default="make_ABCD_input_0_shot")
    parser.add_argument("--ptoken", type = str, default='S')
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    gpu_nums = torch.cuda.device_count()
    # make sure the batch is right
    print("gpu_nums:{:}".format(gpu_nums))

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,args,), nprocs=world_size)
    
    