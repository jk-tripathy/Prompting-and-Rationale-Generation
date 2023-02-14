import os
import json
import pandas as pd


def make_prompt(
        question,
        answer,
        visual_tags=None,
        train_prompt=None,
        choices=None,
        rationale=None,
        isTrain=True,
        ):
    prompt = ''

    if train_prompt:
        prompt += train_prompt + '\n'

    if visual_tags:
        prompt += f'Image Context: {visual_tags}\n'

    prompt += f'Question: {question}\n'

    if choices:
        prompt += 'Choices: '
        for choice in choices:
            prompt += choice + ', '
        prompt += '\n'

    if isTrain:
        prompt += f'Answer: {answer}\n'
    else:
        prompt += 'Answer: '

    if rationale:
        prompt += f'Rationale: {rationale}\n'

    return prompt


def gen_text(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')

    outputs = model.generate(input_ids)
    res = str(tokenizer.decode(outputs[0]))
    trunc_res = res[6:-4]
    return trunc_res


def get_ans(model, tokenizer, prompt_header, train_prompt, test_prompt):
    prompt = prompt_header + '\n' + train_prompt + '\n' + test_prompt
    ans = gen_text(model, tokenizer, prompt)
    return prompt, ans


def get_rationale(model, tokenizer, prompt, ans):
    prompt += f'{ans}\n'
    prompt += 'Explain the rationale behind the answer:'
    rationale = gen_text(model, tokenizer, prompt)
    return rationale


def preproc_okvqa(val_path, train_num=1, infer_num=100, isFull=False):
    val_jsonl = None
    with open(val_path) as f:
        val_jsonl = json.load(f)

    if not isFull:
        val_jsonl = val_jsonl[:train_num+infer_num]

    preproc_list = []
    for item in val_jsonl:
        temp = {}
        temp['question_id'] = item['question_id']
        temp['img_id'] = item['img_id']
        temp['question'] = item['sent']
        temp['choices'] = None
        temp['answer'] = max(item['label'], key=item['label'].get)
        temp['rationale'] = item['explanation']

        preproc_list.append(temp)

    return preproc_list


def preproc_aokvqa(val_path, train_num=1, infer_num=100, isFull=False):
    val_jsonl = None
    with open(val_path) as f:
        val_jsonl = json.load(f)

    if not isFull:
        val_jsonl = val_jsonl[:train_num+infer_num]

    preproc_list = []
    for item in val_jsonl:
        temp = {}
        temp['question_id'] = item['question_id']
        temp['img_id'] = str(item['image_id']).rjust(12, '0')
        temp['question'] = item['question']
        temp['choices'] = item['choices']
        temp['answer'] = item['choices'][item['correct_choice_idx']]
        temp['rationale'] = item['rationales']

        preproc_list.append(temp)

    return preproc_list


def preproc_senmaking(data_paths, train_num=1, infer_num=100, isFull=False):
    statements = pd.read_csv(data_paths['stat'])
    answers = pd.read_csv(data_paths['ans'], header=None)
    rationales = pd.read_csv(data_paths['rat'], header=None)

    if not isFull:
        statements = statements[:train_num+infer_num]
        answers = answers[:train_num+infer_num]
        rationales = rationales[:train_num+infer_num]

    preproc_list = []
    for i in range(len(statements)):
        temp = {}
        concat = '\n1. ' + statements.iloc[i]['sent0'] + \
            '\n2. ' + statements.iloc[i]['sent1']

        temp['question_id'] = str(statements.iloc[i]['id'])
        temp['question'] = concat
        temp['answer'] = str(answers.iloc[i][1] + 1)
        temp['rationale'] = rationales.iloc[i][1:5].tolist()
        temp['choices'] = None

        preproc_list.append(temp)

    return preproc_list


def preproc_esnli(csv_path, train_num=1, infer_num=100, isFull=False):
    df = pd.read_csv(csv_path)
    df = df[['pairID', 'gold_label', 'Sentence1',
             'Sentence2', 'Explanation_1']]

    if not isFull:
        df = df[:train_num+infer_num]

    preproc_list = []
    for i in range(len(df)):
        temp = {}
        concat = '\n1. ' + df.iloc[i]['Sentence1'] +\
                 '\n2. ' + df.iloc[i]['Sentence2']
        temp['question_id'] = str(df.iloc[i]['pairID'])
        temp['question'] = concat
        temp['answer'] = df.iloc[i]['gold_label']
        temp['rationale'] = [str(df.iloc[i]['Explanation_1'])]
        temp['choices'] = None

        preproc_list.append(temp)

    return preproc_list
