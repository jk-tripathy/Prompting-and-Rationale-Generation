import os
import json
import pandas as pd
from tqdm import tqdm
from get_vis_tags import get_visual_tags
from Stage2.img_pipeline import ImagePipelines
from lavis_caption import LavisCaptioningModel


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


def gen_tags(preproc_list, image_root, ctx):
    image_models = ImagePipelines()
    for item in tqdm(preproc_list, desc=f'Generating {ctx}'):
        image_name = item["img_id"] + '.jpg'
        img_path = os.path.join(image_root, image_name)
        results = image_models.get_results(img_path)
        item['tags'] = get_visual_tags(results)

    return preproc_list


def gen_caption(preproc_list, image_root, ctx):
    lavis = LavisCaptioningModel()
    for item in tqdm(preproc_list, desc=f'Generating {ctx}'):
        image_name = item["img_id"] + '.jpg'
        img_path = os.path.join(image_root, image_name)
        item['caption'] = lavis.generate_caption(img_path)

    return preproc_list


def preproc_okvqa(val_path, image_root, ctx,
                  train_num=1, infer_num=100, isFull=False):
    val_jsonl = None
    with open(val_path) as f:
        val_jsonl = json.load(f)

    if not isFull:
        val_jsonl = val_jsonl[:train_num+infer_num]

    preproc_list = []
    print('Preproc OKVQA')
    for i in tqdm(range(len(val_jsonl)), desc='Loading'):
        temp = {}
        temp['question_id'] = val_jsonl[i]['question_id']
        temp['img_id'] = val_jsonl[i]['img_id']
        temp['question'] = val_jsonl[i]['sent']
        temp['choices'] = None
        temp['answer'] = max(val_jsonl[i]['label'], key=val_jsonl[i]['label'].get)
        temp['rationale'] = val_jsonl[i]['explanation']

        preproc_list.append(temp)

    if ctx == 'tags':
        preproc_list = gen_tags(preproc_list, image_root, ctx=ctx)
    elif ctx == 'caption':
        preproc_list = gen_caption(preproc_list, image_root, ctx=ctx)
    elif ctx == 'both':
        preproc_list = gen_tags(preproc_list, image_root, ctx='tags')
        preproc_list = gen_caption(preproc_list, image_root, ctx='caption')

    return preproc_list


def preproc_aokvqa(val_path, image_root, ctx,
                   train_num=1, infer_num=100, isFull=False):
    val_jsonl = None
    with open(val_path) as f:
        val_jsonl = json.load(f)

    if not isFull:
        val_jsonl = val_jsonl[:train_num+infer_num]

    preproc_list = []
    print('Preproc AOKVQA')
    for i in tqdm(range(len(val_jsonl)), desc='Loading'):
        temp = {}
        temp['question_id'] = val_jsonl[i]['question_id']
        temp['img_id'] = str(val_jsonl[i]['image_id']).rjust(12, '0')
        temp['question'] = val_jsonl[i]['question']
        temp['choices'] = val_jsonl[i]['choices']
        temp['answer'] = val_jsonl[i]['choices'][val_jsonl[i]['correct_choice_idx']]
        temp['rationale'] = val_jsonl[i]['rationales']

        preproc_list.append(temp)

    if ctx == 'tags':
        preproc_list = gen_tags(preproc_list, image_root, ctx=ctx)
    elif ctx == 'caption':
        preproc_list = gen_caption(preproc_list, image_root, ctx=ctx)
    elif ctx == 'both':
        preproc_list = gen_tags(preproc_list, image_root, ctx='tags')
        preproc_list = gen_caption(preproc_list, image_root, ctx='caption')

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
    for i in tqdm(range(len(statements)), desc="Preproc SEN-MAKING"):
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
    for i in tqdm(range(len(df)), desc="Preproc SEN-MAKING"):
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
