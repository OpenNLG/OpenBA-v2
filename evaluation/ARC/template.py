def get_input(demo):
    question = f"Question: {demo['question']} "
    instruction = f"Utilize science knowledge to respond to the following grade-school level, multiple-choice science questions. Choose the correct answer based on understanding of real-world scientific principles."
    options = "Options: "
    for idx in range(demo["opt_num"]):
        options += f"{chr(65 + idx)}. {demo['res'+str(idx+1)]} "
    answer_prompt = f"Answer: "
    input_text = instruction + question + options + answer_prompt
    
    return input_text

def make_ABCD_input_0_shot(subject, data):
    demo = data['data']
    
    input_text = get_input(demo)
    ANS_TEMPLATE = ""
    
    decoder_input_text = ANS_TEMPLATE
    return [input_text], decoder_input_text

def make_ABCD_input_25_shot(subject, data):
    demo = data['data']
    
    ANS_TEMPLATE = "{:}"
    input_text = get_input(demo)
    decoder_input_text = ANS_TEMPLATE
    
    demos = data["demo"]
    fs_input_text = ""
    input_texts = [input_text]
    for demo in demos:
        text = get_input(demo)
        
        fs_input_text += text + ANS_TEMPLATE.format(demo[f"ans"]) + '\n ' # origin
        input_texts.append(fs_input_text + input_text)
    return input_texts, decoder_input_text


def choose_longest_input(cand, max_length, tokenizer, add_s):
    idx = len(cand) - 1
    while idx >= 0:
        length = len(tokenizer(cand[idx])["input_ids"])
        if add_s: length += 2
        if length <= max_length:
            return cand[idx]
        idx -= 1
    return cand[0]

