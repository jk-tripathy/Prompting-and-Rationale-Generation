import sys
sys.path.insert(0, "../Stage1")
from txt_pipeline import text_pipeline
from img_pipeline import image_pipeline

if __name__=='__main__':
    text_model_id = 'EleutherAI/gpt-j-6B'
    image_model_ids = ["facebook/detr-resnet-101", "vincentclaes/mit-indoor-scenes", "Rajaram1996/FacialEmoRecog"]
    image_tasks = ['object-detection', 'image-classification', 'image-classification']
    urls = [
        'https://farm4.staticflickr.com/3253/2605202065_d1bcaa15b3_z.jpg',
        'https://farm2.staticflickr.com/1229/1430469166_ebbb5f7b73_z.jpg',
    ]

    test_urls = [
        'https://farm9.staticflickr.com/8040/8030395008_1a6dce67df_z.jpg',
        'http://farm7.staticflickr.com/6041/6232839129_d1c3175917_z.jpg',
        'http://farm9.staticflickr.com/8357/8311566234_48993d3629_z.jpg'
    ]

    captions = [
        'three people on a couch in the living room playing games loking angry and sad',
        'dog walking on the road infront of motorcycles',
    ]
    test_captions = [
        'girl sitting on a chair with a laptop',
        'man with airplane overhead',
        'woman smiling and eating food'
    ]
    prompts = []
    for url, caption in zip(urls, captions):
        tags = []
        for model_id, task in zip(image_model_ids, image_tasks):
            result = image_pipeline(url, model_id, task)
            if task == 'image-classification':
                tags.append(result[0]['label'])
            elif task == 'object-detection':
                for item in result: tags.append(item['label'])
        prompts.append(f'Tags:{",".join(tags)}\nCaption:{caption}')
    #print(prompts) 
    for url, caption in zip(test_urls, test_captions):
        tags = []
        for model_id, task in zip(image_model_ids, image_tasks):
            result = image_pipeline(url, model_id, task)
            if task == 'image-classification':
                tags.append(result[0]['label'])
            elif task == 'object-detection':
                for item in result: tags.append(item['label'])
        test_prompts = [*prompts, f'Tags:{",".join(tags)}\nCaption:']        
        prompt = "\n".join(test_prompts)
        print(prompt)
        resp = text_pipeline(text_model_id, prompt, max_new_tokens=16)
        print(resp)
        print()