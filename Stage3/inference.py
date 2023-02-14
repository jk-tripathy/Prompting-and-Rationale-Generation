import os
import json
from utils import make_prompt, get_ans, get_rationale,\
        preproc_aokvqa, preproc_okvqa, preproc_senmaking, preproc_esnli
from tqdm import tqdm
from get_vis_tags import get_visual_tags
from Stage2.img_pipeline import ImagePipelines
from transformers import T5Tokenizer, T5ForConditionalGeneration


def infer(infer_dict_list, save_path, prompt_header, image_root=None,
          model_name="google/flan-t5-xxl", train_num=1, infer_num=100):

    image_models = ImagePipelines()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       device_map="auto")

    train_prompt = ''

    # train prompts
    for i in range(train_num):
        visual_tags = None

        if image_root:
            image_name = infer_dict_list[i]["img_id"] + '.jpg'
            img_path = os.path.join(image_root, image_name)
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
        visual_tags = None

        if image_root:
            image_name = infer_dict_list[i]["img_id"] + '.jpg'
            img_path = os.path.join(image_root, image_name)
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

        prompt, ans = get_ans(model, tokenizer, prompt_header,
                              train_prompt, test_prompt)

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


if __name__ == "__main__":
    '''TEXT '''

    # SEN-MAKING
    senmaking_prompt_header = 'Which statement of the two is against common sense?'
    senmaking_save_path = 'Stage3/senmaking-flant5-100-rationale.jsonl'
    senmaking_data_paths = {
        'stat':  'data/sen_making/All data/Dev Data/subtaskA_dev_data.csv',
        'ans': 'data/sen_making/All data/Dev Data/subtaskA_gold_answers.csv',
        'rat': 'data/sen_making/All data/Dev Data/subtaskC_gold_answers.csv'
        }

    senmaking_list = preproc_senmaking(senmaking_data_paths)
    infer(senmaking_list, senmaking_save_path, senmaking_prompt_header)

    # e-SNLI
    esnli_prompt_header = "What is the relation between the two sentences.\nPossible answers are: contradiction, entailment, neutral.\n"
    esnli_data_path = 'data/esnli/dataset/esnli_dev.csv'
    esnli_save_path = 'Stage3/esnli-flant5-100-rationale.jsonl'

    esnli_list = preproc_esnli(esnli_data_path)
    infer(esnli_list, esnli_save_path, esnli_prompt_header)

    ''' IMAGE '''
    img_prompt_header = "Answer the following questions based on the given image context\n"

    # OKVQA
    okvqa_image_path = 'data/mscoco/val2014'
    okvqa_save_path = 'Stage3/okvqa-flant5-imctx-100-rationale.jsonl'
    okvqa_data_path = 'data/mscoco/okvqa/okvqa_val.json'

    okvqa_list = preproc_okvqa(okvqa_data_path)
    infer(okvqa_list, okvqa_save_path, img_prompt_header, okvqa_image_path)

    # AOKVQA
    aokvqa_image_path = 'data/mscoco/val2017'
    aokvqa_save_path = 'Stage3/aokvqa-flant5-imctx-100-rationale.jsonl'
    aokvqa_data_path = 'data/mscoco/aokvqa/aokvqa_v1p0_val.json'

    aokvqa_list = preproc_aokvqa(aokvqa_data_path)
    infer(aokvqa_list, aokvqa_save_path, img_prompt_header, aokvqa_image_path)
