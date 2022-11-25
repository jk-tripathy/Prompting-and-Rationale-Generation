from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForObjectDetection
import torch
from PIL import Image
import requests
from transformers import pipeline

def image_pipeline(image, model_id, task):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    if task == 'object-detection':
        model = AutoModelForObjectDetection.from_pretrained(model_id)
    elif task == 'image-classification':
        model = AutoModelForImageClassification.from_pretrained(model_id)
    pipe = pipeline(
        task,
        model=model,
        feature_extractor=feature_extractor,
    )
    results = pipe(image)
    return results

if __name__=='__main__':
    image = 'https://farm4.staticflickr.com/3253/2605202065_d1bcaa15b3_z.jpg'
    image2 = 'http://farm9.staticflickr.com/8357/8311566234_48993d3629_z.jpg'
    model_id = "vincentclaes/mit-indoor-scenes"
    task = 'image-classification'

    results = image_pipeline(image2, model_id, task)
    print(results)

    #for item in results:
        #if item['score'] > 0.9:
            #print(
                #f"Detected {item['label']} with confidence "
                #f"{round(item['score'], 2)} at location {item['box']}"
            #)