import os
import json
import pandas as pd
from utils import *
from tqdm import tqdm
from get_vis_tags import get_visual_tags
from Stage2.img_pipeline import ImagePipelines
from transformers import T5Tokenizer, T5ForConditionalGeneration

def image_infer(infer_dict_list, data_path, save_path, model_name="google/flan-t5-xxl", train_num=1, infer_num=100):
    image_models = ImagePipelines()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    prompt_header = "Answer the following questions based on the given image context\n"

    train_prompt = ''

    # train prompts
    for i in range(train_num):
        image_name = infer_dict_list[i]["img_id"] + '.jpg'
        img_path = os.path.join(data_path, image_name)
        results = image_models.get_results(img_path)

        visual_tags = get_visual_tags(results)

        ques = infer_dict_list[i]["question"]
        gold_answer = infer_dict_list[i]["answer"]
        gold_rationale = infer_dict_list[i]["rationale"][0]
        choices = infer_dict_list[i]["choices"]

        train_prompt = make_prompt(
                ques,
                gold_answer,
                visual_tags=visual_tags,
                train_prompt=train_prompt,
                choices=choices,
                isTrain=True,
                rationale=gold_rationale,
                )

    # inference
    for i in tqdm(range(train_num, train_num+infer_num)):
        image_name = infer_dict_list[i]["img_id"] + '.jpg'
        img_path = os.path.join(data_path, image_name)
        results = image_models.get_results(img_path)

        visual_tags = get_visual_tags(results)

        ques = infer_dict_list[i]["question"]
        gold_answer = infer_dict_list[i]["answer"]
        gold_rationale = infer_dict_list[i]["rationale"]
        choices = infer_dict_list[i]["choices"]

        test_prompt = make_prompt(
                ques,
                gold_answer,
                visual_tags=visual_tags,
                choices=choices,
                isTrain=False
                )

        prompt, ans = get_ans(model, tokenizer, prompt_header, train_prompt, test_prompt)

        rationale = get_rationale(model, tokenizer, prompt, ans)
    
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

        with open(save_path, 'a') as f:
            json.dump(gen_results, f)
            f.write('\n')

def senmaking_infer(save_path, model_name="google/flan-t5-xxl", train_num=1, infer_num=100):
    statements = pd.read_csv('data/sen_making/All data/Dev Data/subtaskA_dev_data.csv')
    answers = pd.read_csv('data/sen_making/All data/Dev Data/subtaskA_gold_answers.csv', header=None)
    rationales = pd.read_csv('data/sen_making/All data/Dev Data/subtaskC_gold_answers.csv', header=None)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    prompt_header = "Which statement of the two is against common sense?\n"

    # train promtps
    train_prompt = ""
    for i in range(train_num):
        concat = "\n1. " + statements.iloc[i]['sent0'] + "\n2. " + statements.iloc[i]['sent1']
        gold_answer = str(answers.iloc[i][1] + 1)
        gold_rationale = rationales.iloc[i][1:5].tolist()
        train_prompt = make_prompt(
                concat,
                gold_answer,
                train_prompt=train_prompt,
                isTrain=True,
                rationale=gold_rationale[0]
                )
    
    #infer
    for i in tqdm(range(train_num, train_num+infer_num)):
        concat = "\n1. " + statements.iloc[i]['sent0'] + "\n2. " + statements.iloc[i]['sent1']
        gold_answer = str(answers.iloc[i][1] + 1)
        gold_rationale = rationales.iloc[i][1:5].tolist()

        test_prompt = make_prompt(
                concat,
                gold_answer,
                isTrain=False
                )

        prompt, ans = get_ans(model, tokenizer, prompt_header, train_prompt, test_prompt)

        rationale = get_rationale(model, tokenizer, prompt, ans)
    
        gen_results = {
                'ques_id': str(statements.iloc[i]['id']),
                'question': concat,
                'gold_answer': gold_answer,
                'answer': ans,
                'gold_rationale': gold_rationale,
                'rationale': rationale,
                }

        with open(save_path, 'a') as f:
            json.dump(gen_results, f)
            f.write('\n')

def esnli_infer(save_path, model_name="google/flan-t5-xxl", train_num=1, infer_num=100):
    df = pd.read_csv('data/esnli/dataset/esnli_dev.csv')
    df = df[['pairID', 'gold_label', 'Sentence1', 'Sentence2', 'Explanation_1']]

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    prompt_header = "What is the relation between the two sentences.\nPossible answers are: contradiction, entailment, neutral.\n"

    # train promtps
    train_prompt = ""
    for i in range(train_num):
        concat = "\n1. " + df.iloc[i]['Sentence1'] + "\n2. " + df.iloc[i]['Sentence2']
        gold_answer = df.iloc[i]['gold_label']
        train_prompt = make_prompt(
                concat,
                gold_answer,
                train_prompt=train_prompt,
                isTrain=True
                )
    
    #infer
    for i in tqdm(range(train_num, train_num+infer_num)):
        concat = "\n1. " + df.iloc[i]['Sentence1'] + "\n2. " + df.iloc[i]['Sentence2']
        gold_answer = str(df.iloc[i]['gold_label'])
        gold_rationale = str(df.iloc[i]['Explanation_1'])

        test_prompt = make_prompt(
                concat,
                gold_answer,
                isTrain=False
                )

        prompt, ans = get_ans(model, tokenizer, prompt_header, train_prompt, test_prompt)
        rationale = get_rationale(model, tokenizer, prompt, ans)
    
        gen_results = {
                'ques_id': str(df.iloc[i]['pairID']),
                'question': concat,
                'gold_answer': gold_answer,
                'answer': ans,
                'gold_rationale': gold_rationale,
                'rationale': rationale,
                }

        with open(save_path, 'a') as f:
            json.dump(gen_results, f)
            f.write('\n')


if __name__ == "__main__":
    train_num = 1
    infer_num = 100

    # OKVQA 

    okvqa_data_path = 'data/mscoco/val2014'
    okvqa_save_path = 'Stage3/okvqa-flant5-imctx-100-rationale.jsonl'
    okvqa_list = preproc_okvqa('data/mscoco/okvqa/okvqa_val.json', train_num, infer_num)
    image_infer(okvqa_list, okvqa_data_path, okvqa_save_path, train_num=train_num, infer_num=infer_num)

    # AOKVQA 
    
    aokvqa_data_path = 'data/mscoco/val2017'
    aokvqa_save_path = 'Stage3/aokvqa-flant5-imctx-100-rationale.jsonl'
    aokvqa_list = preproc_aokvqa('data/mscoco/aokvqa/aokvqa_v1p0_val.json', train_num, infer_num)
    image_infer(aokvqa_list, aokvqa_data_path, aokvqa_save_path, train_num=train_num, infer_num=infer_num)

    # SEN-MAKING

    senmaking_save_path = 'Stage3/senmaking-flant5-100-rationale.jsonl'
    senmaking_infer(senmaking_save_path)

    # e-SNLI

    esnli_save_path = 'Stage3/esnli-flant5-100-rationale.jsonl'
    esnli_infer(esnli_save_path)

