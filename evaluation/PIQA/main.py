import os
import argparse
import json
import torch
import glob
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import template
import numpy as np
import torch.nn as nn


def post_process_ppl(ans_list, test_inputs):
    golds = [i['data']['ans'] for i in test_inputs]
    encs = [i[1] for i in ans_list]
    decs = [i[2] for i in ans_list]
    preds = [i[0] for i in ans_list]
    
    assert len(preds) == len(decs) == (2 * len(golds)) == len(decs)

    ABCD_dic = {0: "A", 1: "B"}
    new_preds, new_encs, new_decs = [], [], []
    for i in range(len(preds))[::2]:
        new_preds.append(ABCD_dic[np.argmin(preds[i: i + 2])])
        new_encs.append(encs[i])
        new_decs.append(decs[i: i + 2])
    preds, encs, decs = new_preds, new_encs, new_decs

    data2write = [json.dumps({'enc':enc, 'dec':dec, 'gold':gold, 'pred':pred}) + '\n' for enc, dec, gold, pred in zip(encs, decs, golds, preds)]
    right = sum([1 for i, j in zip(golds, preds) if i == j])
    cnt = len(golds)
    ABCD_rate = [sum([1 for i in preds if i == j]) / cnt for j in ["A", "B"]]
    return ABCD_rate, cnt, right, data2write

def solve_ppl(model, tokenizer, text_cache, args, rank, world_size):
    global_bsz = len(text_cache["dec"])
    
    while len(text_cache["dec"]) < args.batch_size:
        text_cache["dec"].append("")
        text_cache["enc"].append("")
        
    enc_input=tokenizer(text_cache["enc"], return_tensors='pt', padding=args.padding, max_length = args.max_length, truncation=True)
    input_ids=enc_input.input_ids.chunk(world_size)
    attention_mask=enc_input.attention_mask.chunk(world_size)
    dec_input=tokenizer(text_cache["dec"], return_tensors='pt', padding=args.padding, max_length=args.decoder_max_length,
              truncation=True)
    decoder_input_ids=dec_input.input_ids
    decoder_attention_mask=dec_input.attention_mask.chunk(world_size)
    
    labels = decoder_input_ids.cuda(rank).chunk(world_size)
    decoder_input_ids = model.module._shift_right(decoder_input_ids).chunk(world_size)
    # print(decoder_input_ids)
    preds = []                                           
    chunk_size = len(decoder_input_ids)
    
    with torch.no_grad():
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        logits = model(input_ids=input_ids[rank % chunk_size].cuda(rank),
                       decoder_input_ids=decoder_input_ids[rank % chunk_size].cuda(rank),
                       attention_mask=attention_mask[rank % chunk_size].cuda(rank),
                       decoder_attention_mask=decoder_attention_mask[rank % chunk_size].cuda(rank)).logits
        
        for single_logits, single_label in zip(logits, labels[rank % chunk_size]):
            zero = torch.zeros_like(single_label)
            single_label = torch.where(single_label == 250199, zero, single_label)
            loss = loss_fct(single_logits[:-1], single_label[:-1]).detach().cpu().item()
            preds.append(loss)
        preds = torch.tensor(preds).cuda(rank)
            
    gather_list = [torch.zeros_like(preds) for _ in range(world_size)] if rank == 0 else []
    dist.gather(preds, gather_list=gather_list, dst=0)
    if rank == 0:
        preds = torch.cat(gather_list, dim=0)
        preds = preds[:global_bsz]

    torch.distributed.barrier()
    return preds.tolist()

def post_process_ABCD(out_list, test_inputs):
    golds = [i['data']['ans'] for i in test_inputs]
    encs = [i[1] for i in out_list]
    decs = [i[2] for i in out_list]
    preds = [i[0] for i in out_list]
    # print(golds)
    # print(encs)
    # print(decs)
    # print(preds)
    print("len: pred({:});decs({:});golds({:});decs({:})".format(len(preds),len(decs),len(golds),len(decs)))
    assert len(preds) == len(decs) == len(golds) == len(decs), str(len(preds), len(decs), len(golds), len(decs))
    # str(len(preds), len(decs), len(golds), len(decs))
    data2write = [json.dumps({'enc':enc, 'dec':dec, 'gold':gold, 'pred':pred}) + '\n' for enc, dec, gold, pred in zip(encs, decs, golds, preds)]
    right = sum([1 for i, j in zip(golds, preds) if i == j])
    cnt = len(golds)
    ABCD_rate = [sum([1 for i in preds if i == j]) / cnt for j in ["A", "B"]]
    return ABCD_rate, cnt, right, data2write

def solve_ABCD(model, tokenizer, text_cache, args, rank, world_size):
    global_bsz = len(text_cache["dec"])

    while len(text_cache["dec"]) < args.batch_size:
        text_cache["dec"].append("")
        text_cache["enc"].append("")
        
    enc_input = tokenizer(text_cache["enc"], return_tensors='pt', padding=args.padding, max_length=args.max_length,
                          truncation=True)
    input_ids = enc_input.input_ids.chunk(world_size)
    attention_mask = enc_input.attention_mask.chunk(world_size)
    dec_input = tokenizer(text_cache["dec"], return_tensors='pt', padding=args.padding,
                          max_length=args.decoder_max_length,
                          truncation=True)
    decoder_input_ids = dec_input.input_ids
    decoder_attention_mask = dec_input.attention_mask.chunk(world_size)

    decoder_input_ids = model.module._shift_right(decoder_input_ids).chunk(world_size)
    preds = []
    chunk_size = len(decoder_input_ids)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids[rank % chunk_size].cuda(rank),
                       decoder_input_ids=decoder_input_ids[rank % chunk_size].cuda(rank),
                       attention_mask=attention_mask[rank % chunk_size].cuda(rank),
                       decoder_attention_mask=decoder_attention_mask[rank % chunk_size].cuda(rank)).logits[:,-1,:].contiguous()
        
    gather_list = [torch.zeros_like(logits) for _ in range(world_size)] if rank == 0 else []
    dist.gather(logits, gather_list=gather_list, dst=0)
    if rank == 0:
        logits = torch.cat(gather_list, dim=0)
        logits = logits[:global_bsz]

    for single_logits in logits:
        probs = (
            torch.tensor(
                [
                    single_logits[tokenizer.convert_tokens_to_ids("A")],
                    single_logits[tokenizer.convert_tokens_to_ids("B")],
                ]
            ).detach().cpu().numpy()
        )
        pred = {0: "A", 1: "B"}[np.argmax(probs)]
        preds.append(pred)

    torch.distributed.barrier()
    return preds

def main(rank, world_size, args):
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, max_length=1024)
    tokenizer.padding_side = "left"
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, trust_remote_code=True).half().cuda(rank)
    model = torch.nn.DataParallel(model, device_ids=[rank])
    model.eval()

    if torch.distributed.get_rank() == 0:
        if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
    
    torch.distributed.barrier()

    json_input_paths = glob.glob(os.path.join(args.input_folder, '*.json'))
    json_file_names = [os.path.basename(i) for i in json_input_paths]
    json_output_paths = [os.path.join(args.output_folder, i) for i in json_file_names]

    make_input = getattr(template, args.template_type)
    solve = solve_ABCD if "ABCD" in args.template_type else solve_ppl
    post_process = post_process_ABCD if "ABCD" in args.template_type else post_process_ppl
   
    all_cnt, all_right = 0, 0
    for file_name, input_path, output_path in zip(json_file_names, json_input_paths, json_output_paths):
        cache_e,cache_d=[],[]
        out_list = []
        with open(input_path, 'r') as fr, open(output_path, 'w') as fw:
            test_inputs = json.load(fr)
            print("test size: "+str(len(test_inputs)))
            
            for idx in tqdm(range(0, len(test_inputs))):    
                input_data = test_inputs[idx] # demo是few shot， data是问题
                candidates, decoder_input_text = make_input(file_name, input_data)
                # import pdb; pdb.set_trace()
                input_text = template.choose_longest_input(candidates, args.max_length, tokenizer, args.add_prefix)
                
                if args.add_prefix: 
                    input_text = f"<{args.ptoken}> " + input_text + "<extra_id_0>"
                    decoder_input_text = [f"<extra_id_0> " + _ for _ in decoder_input_text]
                else:
                    input_text = input_text
                    decoder_input_text = decoder_input_text
                    
                inputs = [input_text for _ in range(len(decoder_input_text))]
                # if ppl, make 4 x prompts
                for e, d in zip(inputs, decoder_input_text):
                    cache_e.append(e)
                    cache_d.append(d)
                    
            print("enc_len{:};dec_len{:}".format(len(cache_e),len(cache_d)))
            chunked_e=[cache_e[i:i+args.batch_size] for i in range(0, len(cache_e), args.batch_size)]
            chunked_d=[cache_d[i:i+args.batch_size] for i in range(0, len(cache_d), args.batch_size)]
            
            for chunk_e,chunk_d in zip(chunked_e,chunked_d):
                text_cache = {"enc":chunk_e,"dec":chunk_d}
                batch_pred = solve(model, tokenizer, text_cache, args, rank, world_size)
                for ans, e, d in zip(batch_pred, text_cache['enc'], text_cache['dec']):
                    out_list.append((ans, e, d))
            

            if torch.distributed.get_rank() == 0:
                ABCD_rate, cnt, right, data2write = post_process(out_list, test_inputs)
                fw.write(f"acc: {(right / cnt)*100:.2f}%")
                print(file_name.replace('_test.json', '').replace('_', ' '), ': \n', f"acc: {(right / cnt)*100:.2f}%")
                for label, rate in zip(['A', 'B'], ABCD_rate): print(f"{label}: {rate*100:.2f}%", end = '|')
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
    parser.add_argument("--batch-size", type = int, default=4)
    parser.add_argument("--ptoken", type = str, default='S')
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    gpu_nums = torch.cuda.device_count()
    # make sure the batch is right
    assert args.batch_size % gpu_nums ==0 

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,args,), nprocs=world_size)
    