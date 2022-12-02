from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForObjectDetection, AutoModelForVision2Seq, AutoTokenizer

def image_pipeline(image, model_id, task):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    if task == 'object-detection':
        model = AutoModelForObjectDetection.from_pretrained(model_id)
    elif task == 'image-classification':
        model = AutoModelForImageClassification.from_pretrained(model_id)
    elif task == 'image-to-text':
        model = AutoModelForVision2Seq.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        task,
        model = model,
        feature_extractor = feature_extractor,
        tokenizer = tokenizer if task == 'image-to-text' else None,
    )
    results = pipe(image)
    return results

if __name__=='__main__':
    image = 'https://farm4.staticflickr.com/3253/2605202065_d1bcaa15b3_z.jpg'
    model_id = 'nlpconnect/vit-gpt2-image-captioning' 
    task = 'image-to-text'

    results = image_pipeline(image, model_id, task)
    print(results)