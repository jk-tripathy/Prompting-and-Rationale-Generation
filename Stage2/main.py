import sys
sys.path.insert(0, "../")
import os
import json
from Stage1.txt_pipeline import text_pipeline
from img_pipeline import image_pipeline
import pandas as pd

if __name__=='__main__':
    image_models = {
        'object-detection' : ['facebook/detr-resnet-101'],
        'image-classification' : ['vincentclaes/mit-indoor-scenes', 'Rajaram1996/FacialEmoRecog'],
        'image-to-text' : ['nlpconnect/vit-gpt2-image-captioning']
       #'OFA-Sys/ofa-huge' : 'image-to-text',
    }
    dataset = pd.read_csv('../data/MAMI/trial.csv', delimiter='\t')
    dataset = dataset[:20]

    for _, row in dataset.iterrows():
        mami_results = {
            'file' : row[0],
            'task_label' : [dataset.columns[i] for i in range(1, 6) if row[i] == 1],
            'text transcription' : row[-1],
        }

        img_path = os.path.join('../data/MAMI/images', row[0])
        
        for task in image_models:
            mami_results[task] = []
            for model in image_models[task]:
                result = image_pipeline(img_path, model, task)
                mami_results[task].append({'model_id': model, 'output': result})

        with open('mami_results.jsonl', 'a') as f:
            json.dump(mami_results, f)
            f.write('\n')
