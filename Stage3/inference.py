import os
import json
from utils import *
from tqdm import tqdm
from Stage2.img_pipeline import ImagePipelines
from transformers import T5Tokenizer, T5ForConditionalGeneration

def make_prompt(visual_tags, question, answer, rationale=None, isTrain=True):
    prompt = f"Image Context: {visual_tags}\n"
    prompt += f"Question: {question}\n"
    if isTrain:
        if rationale:
            prompt += f"Rationale: {rationale}\n"
            
        prompt += f"Answer: {answer}\n"
    else: 
        prompt += "Answer: "

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


if __name__ == "__main__":
    val_x = None
    with open('data/mscoco/val_x.json') as f:
        val_x = json.load(f)
    
    image_models = ImagePipelines()

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")

    image_name = val_x[0]["img_id"] + '.jpg'
    results = {}
    img_path = os.path.join('data/mscoco/val2014', image_name)
    for task_label in image_models.tasks:
        result = image_models.get_results(img_path, task_label)
        results[task_label] = result

    visual_tags = "This is a image with  person,  boat, kite," #get_visual_tags(results)

    train_prompt = make_prompt(
            visual_tags,
            val_x[0]["sent"],
            max(val_x[0]["label"], key=val_x[0]["label"].get),
            isTrain=True
            )
    
        
    prompt_header = "Answer the following questions based on the image context given.\n"

    for i in tqdm(range(1, 101)):
        image_name = val_x[i]["img_id"] + '.jpg'
        results = {}
        img_path = os.path.join('data/mscoco/val2014', image_name)
        for task_label in image_models.tasks:
            result = image_models.get_results(img_path, task_label)
            results[task_label] = result

        visual_tags = get_visual_tags(results)

        ques = val_x[i]["sent"]
        gold_answer = max(val_x[i]["label"], key=val_x[i]["label"].get)
        gold_rationale = val_x[i]["explanation"]

        test_prompt = make_prompt(
                visual_tags,
                ques,
                gold_answer,
                isTrain=False
                )

        prompt, ans = get_ans(model, tokenizer, prompt_header, train_prompt, test_prompt)

        rationale = get_rationale(model, tokenizer, prompt, ans)
    
        gen_results = {
                'okvqa_q_id': val_x[i]['question_id'],
                'img_id': image_name,
                'visual_tags': visual_tags,
                'question': ques,
                'gold_answer': gold_answer,
                'answer': ans,
                'gold_rationale': gold_rationale,
                'rationale': rationale,
                }

        with open('Stage3/flant5-imctx-okvqa-100.jsonl', 'a') as f:
            json.dump(gen_results, f)
            f.write('\n')




















