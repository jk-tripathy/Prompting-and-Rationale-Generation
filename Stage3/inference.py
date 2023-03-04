import os
import json
import torch
from utils import make_prompt, get_ans, get_rationale,\
        preproc_aokvqa, preproc_okvqa, preproc_senmaking, preproc_esnli
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def infer(infer_dict_list, save_path, prompt_header, image_root=None, ctx=None, isFull=False,
          model_name='google/flan-t5-xxl', train_num=1, infer_num=100, useRationale=False):
    torch.cuda.empty_cache()

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       device_map='auto')

    train_prompt = ''

    # train prompts
    for i in range(train_num):
        visual_tags = None

        if image_root:
            if ctx == 'ctx':
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
    endNum = len(infer_dict_list) if isFull else train_num+infer_num
    
    for i in tqdm(range(train_num, endNum), desc=f'Inference:{ctx}'):
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
    '''TEXT '''

    # SEN-MAKING
    print('----- SEN-MAKING -----')
    senmaking_prompt_header = 'Which statement of the two is against common sense?'
    senmaking_save_path = 'Stage3/senmaking-flant5-100-rationale.jsonl'
    senmaking_data_paths = {
        'stat': 'data/sen_making/All data/Dev Data/subtaskA_dev_data.csv',
        'ans': 'data/sen_making/All data/Dev Data/subtaskA_gold_answers.csv',
        'rat': 'data/sen_making/All data/Dev Data/subtaskC_gold_answers.csv'
    }
    senmaking_preproc_path = 'Stage3/preproc-senmaking-100.json'

    '''
    if not os.path.isfile(senmaking_preproc_path):
        senmaking_list = preproc_senmaking(senmaking_data_paths)
        with open(senmaking_preproc_path, 'w') as f:
            json.dump(senmaking_list, f)

    with open(senmaking_preproc_path, 'rb') as f:
        senmaking_list = json.load(f)

    infer(senmaking_list, senmaking_save_path, senmaking_prompt_header)
    '''

    # e-SNLI
    print('----- e-SNLI -----')
    esnli_prompt_header = 'What is the relation between the two sentences.\nPossible answers are: contradiction, entailment, neutral.\n'
    esnli_data_path = 'data/esnli/dataset/esnli_dev.csv'
    esnli_save_path = 'Stage3/esnli-flant5-100-rationale.jsonl'
    esnli_preproc_path = 'Stage3/preproc-esnli-100.json'

    '''
    if not os.path.isfile(esnli_preproc_path):
        esnli_list = preproc_esnli(esnli_data_path)
        with open(esnli_preproc_path, 'w') as f:
            json.dump(esnli_list, f)

    with open(esnli_preproc_path, 'rb') as f:
        esnli_list = json.load(f)

    infer(esnli_list, esnli_save_path, esnli_prompt_header)
    '''

    ''' IMAGE '''
    img_prompt_header = 'Answer the following questions based on the given image context\n'
    img_ctx_types = ['both', 'tags', 'caption']
    rationale = [False, True]

    # OKVQA
    print('----- OKVQA -----')
    okvqa_image_path = 'data/mscoco/val2014'
    okvqa_data_path = 'data/mscoco/okvqa/okvqa_test.json'
    okvqa_preproc_path = 'Stage3/preproc-okvqa-full.json'

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

    ctx_type = 'caption'
    rat = False
    okvqa_save_path = f'Stage3/okvqa-flant5-{ctx_type}-rat_{rat}-full.jsonl'
    infer(
        okvqa_list,
        okvqa_save_path,
        img_prompt_header,
        image_root=okvqa_image_path,
        ctx=ctx_type,
        isFull=True,
        useRationale=rat,
    )

    # AOKVQA
    print('----- AOKVQA -----')
    aokvqa_image_path = 'data/mscoco/val2017'
    aokvqa_save_path = 'Stage3/aokvqa-flant5-tags-rationale-100.jsonl'
    aokvqa_data_path = 'data/mscoco/aokvqa/aokvqa_v1p0_val.json'
    aokvqa_preproc_path = 'Stage3/preproc-aokvqa-full.json'

    '''
    if not os.path.isfile(aokvqa_preproc_path):
        aokvqa_list = preproc_aokvqa(
            aokvqa_data_path,
            image_root=aokvqa_image_path,
            ctx='tags',
        )
        with open(aokvqa_preproc_path, 'w') as f:
            json.dump(aokvqa_list, f)

    with open(aokvqa_preproc_path, 'rb') as f:
        aokvqa_list = json.load(f)

    infer(
        aokvqa_list,
        aokvqa_save_path,
        img_prompt_header,
        image_root=aokvqa_image_path,
        ctx='tags',
        useRationale=True,
    )
    '''
