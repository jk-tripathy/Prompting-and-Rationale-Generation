import json
from evaluate import load

if __name__ == "__main__":
    
    with open('Stage3/flant5-imctx-okvqa-100.jsonl', 'r') as json_file:
        json_list = list(json_file)
    
    
    label_preds = []
    label_gold = []
    rationale_preds = []
    rationale_gold = []
    
    for json_str in json_list:
        res = json.loads(json_str)
        label_preds.append(res["answer"])
        label_gold.append(res["gold_answer"])
        rationale_preds.append(res["rationale"])
        rationale_gold.append(res["gold_rationale"])
    
    acc = load('exact_match')
    bleu = load('bleu')
    rouge = load('rouge')
    meteor = load('meteor')

    acc_res = acc.compute(predictions=label_preds, references=label_gold)
    bleu_res = bleu.compute(predictions=rationale_preds, references=rationale_gold)
    rouge_res = rouge.compute(predictions=rationale_preds, references=rationale_gold)
    meteor_res = meteor.compute(predictions=rationale_preds, references=rationale_gold)

    print(f'Accuracy: {acc_res["exact_match"]}')
    print(f'BLEU: {bleu_res["bleu"]}')
    print(f'ROUGE: {rouge_res["rougeL"]}')
    print(f'METEOR: {meteor_res["meteor"]}')

