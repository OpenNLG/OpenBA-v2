import re


def make_ppl_input_5_shot(subject, data):
    demo = data['data']
    ASK_TEMPLATE = "Question: {:} OPTIONS: - {:} - {:} - {:} - {:} Answer: "
    ANS_TEMPLATE = "{:}"

    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"])

    assert "res5" not in demo
    anss = [ANS_TEMPLATE.format(demo[f"res{i}"]) for i in range(1, 5)]
    output = [input_text]
    demos = data["demo"]
    fs_input_text = "Fisrt, l will give you some examples. You need to answer the last question. Examples:"
    for demo in demos:
        ans_dic = {"A": "res1", "B": "res2", "C": "res3", "D": "res4"}
        fs_input_text += ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"]) + \
                         ' ' + ANS_TEMPLATE.format(demo[ans_dic[demo[f"ans"]]]) + ' '

        output.append(fs_input_text + "Please answer the following question. " + input_text)
    return output, anss

def make_r_ppl_input_5shot(subject, data):
    def write_sentinal(s):
        def replacer(match):
            replacer.counter += 1
            return f"<extra_id_{replacer.counter}>"
        replacer.counter = -1  
        return re.sub(r'\|\*sent_id\*\|', replacer, s)
    def make_data(data):
        return [write_sentinal(data[0]), [write_sentinal(i) for i in data[1]]]
    ASK_TEMPLATE = "Question: {:} Options: - {:} - {:} - {:} - {:}"
    ANS_TEMPLATE = "Answer: |*sent_id*| "
    demo = data['data']
    output, decoder_input_texts = [],['|*sent_id*| ' + demo[f"res{i}"] for i in range(1,5)]
    fs_decoder_input_text, fs_input_text = "", ""
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"]) + ' ' + ANS_TEMPLATE
    output.append(make_data([fs_input_text + input_text, [fs_decoder_input_text + i for i in decoder_input_texts]]))
    for demo in data["demo"]:
        ans_dic = {"A": "res1", "B": "res2", "C":"res3", "D":"res4"}
        fs_input_text += ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"]) + \
                         ' ' + ANS_TEMPLATE +' '
        fs_decoder_input_text += "|*sent_id*| " + demo[ans_dic[demo[f"ans"]]]
        output.append(make_data([fs_input_text + input_text, [fs_decoder_input_text + i for i in decoder_input_texts]]))
    return output

def make_ppl_input_0_shot(subject, data):
    demo = data['data']
    ASK_TEMPLATE = "Question: {:} OPTIONS: - {:} - {:} - {:} - {:} Choose one of the options as an answer to the question."
    ANS_TEMPLATE = "{:}"
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"])
    assert "res5" not in demo
    anss = [ANS_TEMPLATE.format(demo[f"res{i}"]) for i in range(1,5)]
    return [input_text], anss

def make_ABCD_input_0_shot(subject, data):
    demo = data['data']

    ASK_TEMPLATE = "Please answer the following multiple choice question: \n\n Question: {:} Options: A: {:} B: {:} C: {:} D: {:} Answer:"  # origin
    ANS_TEMPLATE = ""  
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"])

    assert "res5" not in demo
    anss = ANS_TEMPLATE 

    return [input_text], [anss]
    
def make_ABCD_input_5_shot(subject, data):
    demo = data['data']
    ASK_TEMPLATE = "Question: {:} Options: A: {:} B: {:} C: {:} D: {:} Answer:"  # origin
    ANS_TEMPLATE = "{:}"
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"]) # origin

    assert "res5" not in demo
    anss = ANS_TEMPLATE 
    demos = data["demo"]
    fs_input_text = "Fisrt, l will give you some examples. You need to answer the last question. Examples: "
    output = [input_text]
    for demo in demos:
        ans_dic = {"A": "res1", "B": "res2", "C":"res3", "D":"res4"}
        fs_input_text += ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"]) + \
                         ANS_TEMPLATE.format(demo[f"ans"]) + ' \n ' # origin
        output.append(fs_input_text +"Please answer the following question. "+ input_text)
    return output, [""]

def choose_longest_input(cand, max_length, tokenizer, add_s):
    # print(cand)
    idx = len(cand) - 1
    while idx >= 0:
        length = len(tokenizer(cand[idx])["input_ids"])
        if add_s: length += 2
        if length <= max_length:
            return cand[idx]
        idx -= 1
    print(cand[0])
    return cand[0]


