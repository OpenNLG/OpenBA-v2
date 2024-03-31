import re
def make_ppl_input_10_shot(subject, data):
    demo = data['data']
    ASK_TEMPLATE = "Question: {:} Answer: "
    ANS_TEMPLATE = "{:}"
    input_text = ASK_TEMPLATE.format(demo["question"])
    
    decoder_input_text = [ANS_TEMPLATE.format(demo[f"res{i}"]) for i in range(1,3)]
    output = [input_text]
    demos = data["demo"]
    fs_input_text = ""
    for demo in demos:
        ans_dic = {"A": "res1", "B": "res2"}
        fs_input_text += ASK_TEMPLATE.format(demo["question"]) + \
                         ' ' + ANS_TEMPLATE.format(demo[ans_dic[demo[f"ans"]]]) +' '

        output.append(fs_input_text + input_text)
    return output, decoder_input_text

def make_r_ppl_input_10shot(subject, data):
    def write_sentinal(s):
        def replacer(match):
            replacer.counter += 1
            return f"<extra_id_{replacer.counter}>"
        replacer.counter = -1  
        return re.sub(r'\|\*sent_id\*\|', replacer, s)
    def make_data(data):
        return [write_sentinal(data[0]), [write_sentinal(i) for i in data[1]]]
    ASK_TEMPLATE = "Question: {:} Answer: "
    ANS_TEMPLATE = "|*sent_id*| "
    demo = data['data']
    output, decoder_input_texts = [],['|*sent_id*| ' + demo[f"res{i}"] for i in range(1,3)]
    fs_decoder_input_text, fs_input_text = "", ""
    input_text = ASK_TEMPLATE.format( demo["question"]) + ' ' + ANS_TEMPLATE
    output.append(make_data([fs_input_text + input_text, [fs_decoder_input_text + i for i in decoder_input_texts]]))
    for demo in data["demo"]:
        ans_dic = {"A": "res1", "B": "res2"}
        fs_input_text += ASK_TEMPLATE.format( demo["question"]) + \
                         ' ' + ANS_TEMPLATE +' '
        fs_decoder_input_text += "|*sent_id*| " + demo[ans_dic[demo[f"ans"]]]
        output.append(make_data([fs_input_text + input_text, [fs_decoder_input_text + i for i in decoder_input_texts]]))
    return output


def make_ppl_input_0_shot(subject, data):
    demo = data['data']
    ASK_TEMPLATE = "Question: {:} Answer: "
    ANS_TEMPLATE = "{:}"
    input_text = ASK_TEMPLATE.format(demo["question"])

    decoder_input_text = [ANS_TEMPLATE.format(demo[f"res{i}"]) for i in range(1,3)]
    return [input_text], decoder_input_text

def make_ABCD_input_0_shot(subject, data):
    demo = data['data']
    instruction = "Please answer the following multiple choice question based on physical commonsense knowledge\n\n"
    ASK_TEMPLATE = instruction + "Question: {:} Options: (A) {:} (B) {:} Answer: ("
    ANS_TEMPLATE = ""
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"])
    decoder_input_text = ANS_TEMPLATE
    return [input_text], [decoder_input_text]

def make_ABCD_input_10_shot(subject, data):
    # o-shot+q, 1-shot+q, ..., 5-shot+q
    demo = data['data']
    instruction = "Please answer the following multiple choice question based on physical commonsense knowledge\n\n"
    ASK_TEMPLATE = instruction + "Question: {:} Options: (A) {:} (B) {:} Answer: ("
    ANS_TEMPLATE = "{:}"
    
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"]) # origin
    decoder_input_text = ANS_TEMPLATE.format('')
    demos = data["demo"]
    fs_input_text = ""
    input_texts = [input_text]
    for demo in demos:
        fs_input_text += ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"]) + \
                         ANS_TEMPLATE.format(demo[f"ans"]) + ') \n' # origin
        input_texts.append(fs_input_text + input_text)
    return input_texts, [decoder_input_text]


def choose_longest_input(cand, max_length, tokenizer, add_s):
    idx = len(cand) - 1
    while idx >= 0:
        length = len(tokenizer(cand[idx])["input_ids"])
        if add_s: length += 2
        if length <= max_length:
            return cand[idx]
        idx -= 1
    return cand[0]


