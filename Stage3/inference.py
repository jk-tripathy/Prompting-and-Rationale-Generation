import os
import json
import torch
from utils import make_prompt, get_ans, get_rationale,\
    preproc_aokvqa, preproc_okvqa, preproc_senmaking, preproc_esnli
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def infer(name, model, tokenizer, infer_dict_list, save_path, prompt_header,
          image_root=None, ctx=None, isFull=False, train_num=1, infer_num=100, useRationale=False):
    train_prompt = ''

    # train prompts
    for i in range(train_num):
        visual_tags = None

        if image_root:
            if ctx == 'tags':
                visual_tags = infer_dict_list[i]['tags']
            elif ctx == 'caption':
                visual_tags = infer_dict_list[i]['caption']
            elif ctx == 'both':
                visual_tags = infer_dict_list[i]['tags'] + ' ' + infer_dict_list[i]['caption']

        ques = infer_dict_list[i]['question']
        gold_answer = infer_dict_list[i]['answer']
        gold_rationale = infer_dict_list[i]['rationale'][0]
        choices = infer_dict_list[i]['choices']

        train_prompt = make_prompt(
            ques,
            gold_answer,
            visual_tags=visual_tags,
            train_prompt=train_prompt,
            choices=choices,
            isTrain=True,
            rationale=gold_rationale if useRationale else None,
        )

    # inference
    endNum = len(infer_dict_list) if isFull else train_num + infer_num

    for i in tqdm(range(train_num, endNum), desc=f'Inference:{name}', ncols=120):
        visual_tags = None

        if image_root:
            if ctx == 'tags':
                visual_tags = infer_dict_list[i]['tags']
            elif ctx == 'caption':
                visual_tags = infer_dict_list[i]['caption']
            elif ctx == 'both':
                visual_tags = infer_dict_list[i]['tags'] + ' ' + infer_dict_list[i]['caption']

        ques = infer_dict_list[i]['question']
        gold_answer = infer_dict_list[i]['answer']
        gold_rationale = infer_dict_list[i]['rationale']
        choices = infer_dict_list[i]['choices']

        test_prompt = make_prompt(
            ques,
            gold_answer,
            visual_tags=visual_tags,
            choices=choices,
            isTrain=False
        )

        prompt, ans = get_ans(model, tokenizer, prompt_header,
                              train_prompt, test_prompt)

        '''
        print()
        print(prompt)
        exit()
        '''

        rationale = get_rationale(model, tokenizer, prompt, ans)

        if image_root:
            gen_results = {
                'ques_id': infer_dict_list[i]['question_id'],
                'img_id': infer_dict_list[i]['img_id'],
                'visual_tags': visual_tags,
                'question': ques,
                'gold_answer': gold_answer,
                'answer': ans,
                'gold_rationale': gold_rationale,
                'rationale': rationale,
            }
        else:
            gen_results = {
                'ques_id': infer_dict_list[i]['question_id'],
                'question': ques,
                'gold_answer': gold_answer,
                'answer': ans,
                'gold_rationale': gold_rationale,
                'rationale': rationale,
            }

        with open(save_path, 'a') as f:
            json.dump(gen_results, f)
            f.write('\n')


if __name__ == '__main__':
    save_dir = 'Stage3/inference'
    model_name = 'google/flan-t5-xxl'
    torch.cuda.empty_cache()

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       device_map='auto')

    '''TEXT '''
    # SEN-MAKING SETUP
    senmaking_prompt_header = 'Which statement of the two is against common sense?'
    senmaking_data_paths = {
        'stat': 'data/sen_making/All data/Dev Data/subtaskA_dev_data.csv',
        'ans': 'data/sen_making/All data/Dev Data/subtaskA_gold_answers.csv',
        'rat': 'data/sen_making/All data/Dev Data/subtaskC_gold_answers.csv'
    }
    senmaking_preproc_path = 'Stage3/preproc/preproc-senmaking.json'

    if not os.path.isfile(senmaking_preproc_path):
        senmaking_list = preproc_senmaking(senmaking_data_paths, isFull=True)
        with open(senmaking_preproc_path, 'w') as f:
            json.dump(senmaking_list, f)

    with open(senmaking_preproc_path, 'rb') as f:
        senmaking_list = json.load(f)

    # e-SNLI SETUP
    esnli_prompt_header = 'What is the relation between the two sentences.\nPossible answers are: contradiction, entailment, neutral.\n'
    esnli_data_path = 'data/esnli/dataset/esnli_dev.csv'
    esnli_preproc_path = 'Stage3/preproc/preproc-esnli.json'

    if not os.path.isfile(esnli_preproc_path):
        esnli_list = preproc_esnli(esnli_data_path, isFull=True)
        with open(esnli_preproc_path, 'w') as f:
            json.dump(esnli_list, f)

    with open(esnli_preproc_path, 'rb') as f:
        esnli_list = json.load(f)

    ''' IMAGE '''
    img_prompt_header = 'Answer the following questions based on the given image context\n'

    # OKVQA SETUP
    okvqa_image_path = 'data/mscoco/val2014'
    okvqa_data_path = 'data/mscoco/okvqa/okvqa_test.json'
    okvqa_preproc_path = 'Stage3/preproc/preproc-okvqa.json'

    if not os.path.isfile(okvqa_preproc_path):
        aokvqa_list = preproc_okvqa(
            okvqa_data_path,
            image_root=okvqa_image_path,
            ctx='both',
            isFull=True
        )
        with open(okvqa_preproc_path, 'w') as f:
            json.dump(aokvqa_list, f)

    with open(okvqa_preproc_path, 'rb') as f:
        okvqa_list = json.load(f)

    # AOKVQA SETUP
    aokvqa_image_path = 'data/mscoco/val2017'
    aokvqa_data_path = 'data/mscoco/aokvqa/aokvqa_v1p0_val.json'
    aokvqa_preproc_path = 'Stage3/preproc/preproc-aokvqa.json'

    if not os.path.isfile(aokvqa_preproc_path):
        aokvqa_list = preproc_aokvqa(
            aokvqa_data_path,
            image_root=aokvqa_image_path,
            ctx='both',
            isFull=True
        )
        with open(aokvqa_preproc_path, 'w') as f:
            json.dump(aokvqa_list, f)

    with open(aokvqa_preproc_path, 'rb') as f:
        aokvqa_list = json.load(f)

    ''' INFERENCE '''
    rat = True
    for i in range(6):
        senmaking_file = f'senmaking_{i}-flant5-rat_{rat}'
        senmaking_save_path = f'{save_dir}/{senmaking_file}.jsonl'
        infer(
            senmaking_file,
            model,
            tokenizer,
            senmaking_list,
            senmaking_save_path,
            senmaking_prompt_header,
            isFull=True,
            train_num=i,
            useRationale=rat,
        )

        for ctx_type in ['both', 'tags', 'caption']:
            okvqa_file = f'okvqa_{i}-flant5-rat_{rat}'
            okvqa_save_path = f'{save_dir}/{okvqa_file}.jsonl'
            infer(
                okvqa_file,
                model,
                tokenizer,
                okvqa_list,
                okvqa_save_path,
                img_prompt_header,
                image_root=okvqa_image_path,
                ctx=ctx_type,
                isFull=True,
                train_num=i,
                useRationale=rat,
            )

            aokvqa_file = f'aokvqa_{i}-flant5-rat_{rat}'
            aokvqa_save_path = f'{save_dir}/{aokvqa_file}.jsonl'
            infer(
                aokvqa_file,
                model,
                tokenizer,
                aokvqa_list,
                aokvqa_save_path,
                img_prompt_header,
                image_root=aokvqa_image_path,
                ctx=ctx_type,
                isFull=True,
                train_num=i,
                useRationale=rat,
            )

    for i in range(6):
        esnli_file = f'esnli_{i}-flant5-rat_{rat}'
        esnli_save_path = f'{save_dir}/{esnli_file}.jsonl'
        infer(
            esnli_file,
            model,
            tokenizer,
            esnli_list,
            esnli_save_path,
            esnli_prompt_header,
            isFull=True,
            train_num=i,
            useRationale=rat,
        )
