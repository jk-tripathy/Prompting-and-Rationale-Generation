import json

def make_prompt(
        question,
        answer,
        visual_tags=None,
        train_prompt=None,
        choices=None,
        rationale=None,
        isTrain=True,
        ):
    prompt = ""

    if train_prompt:
        prompt += train_prompt + "\n"

    if visual_tags:
        prompt += f"Image Context: {visual_tags}\n"

    prompt += f"Question: {question}\n"

    if choices:
        prompt += "Choices: "
        for choice in choices:
            prompt += choice + ", "
        prompt += "\n"

    if isTrain:
        prompt += f"Answer: {answer}\n"
    else: 
        prompt += "Answer: "

    if rationale:
        prompt += f"Rationale: {rationale}\n"

    return prompt

def gen_text(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    outputs = model.generate(input_ids)
    res = str(tokenizer.decode(outputs[0]))
    trunc_res = res[6:-4]
    return trunc_res

def get_ans(model, tokenizer, prompt_header, train_prompt, test_prompt):
    prompt = prompt_header + "\n" + train_prompt + "\n" + test_prompt
    ans = gen_text(model, tokenizer, prompt)
    return prompt, ans

def get_rationale(model, tokenizer, prompt, ans):
    prompt += f"{ans}\n"
    prompt +="Explain the rationale behind the answer:"
    rationale = gen_text(model, tokenizer, prompt)
    return rationale

def preproc_okvqa(val_path, train_num, infer_num):
    val_jsonl = None
    with open(val_path) as f:
        val_jsonl = json.load(f)
    
    preproc_list = []
    for item in val_jsonl[:train_num+infer_num]:
        temp = {}
        temp['question_id'] = item['question_id']
        temp['img_id'] = item['img_id']
        temp['question'] = item['sent']
        temp['choices'] = None
        temp['answer'] = max(item['label'], key=item['label'].get)
        temp['rationale'] = item['explanation']

        preproc_list.append(temp)

    return preproc_list
        
def preproc_aokvqa(val_path, train_num, infer_num):
    val_jsonl = None
    with open(val_path) as f:
        val_jsonl = json.load(f)
    
    preproc_list = []
    for item in val_jsonl[:train_num+infer_num]:
        temp = {}
        temp['question_id'] = item['question_id']
        temp['img_id'] = str(item['image_id']).rjust(12, "0")
        temp['question'] = item['question']
        temp['choices'] = item['choices']
        temp['answer'] = item['choices'][item['correct_choice_idx']]
        temp['rationale'] = item['rationales']

        preproc_list.append(temp)

    return preproc_list
