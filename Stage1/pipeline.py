from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from time import time
from transformers import pipeline


def local_inf(model_id, prompts, temperature=0.9, max_new_tokens=32, do_sample=True, batch_size=5):
    #model_id = 'bigscience/bloom'
    model_id = 'EleutherAI/gpt-j-6B'
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", offload_folder="offload")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        'text-generation',
        model=model, 
        tokenizer=tokenizer, 
        device=0, 
        batch_size=batch_size, 
    )
    pipe.pad_token_id = pipe.model.config.eos_token_id
    response = pipe(
        prompts,
        #temperature = temperature, # 0 to 1
        #max_new_tokens = max_new_tokens, # up to 2047 theoretically
        #do_sample = do_sample, # True: use sampling, False: Greedy decoding.
    )
    return response
    #else:
        #model = AutoModelForCausalLM.from_pretrained(model_id)
        #tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        #tokenizer.pad_token = tokenizer.eos_token

        #responses = []
        #for prompt in prompts:
            #input_ids = tokenizer(prompt, return_tensors="pt", padding=True)
            #gen_tokens = model.generate(
                #**input_ids,
                #do_sample=True,
                #temperature=0.9,
                #max_new_tokens=max_new_tokens
            #)
            #responses.append(tokenizer.batch_decode(gen_tokens)[0])
        #return responses

if __name__=='__main__':
    model_id = 'EleutherAI/gpt-j-6B'
    inp = "Can cats fly?"
    t = time()
    resp = local_inf(model_id, inp, max_new_tokens=64)
    delta = time() - t
    print("Inference took %0.2f s." % delta)
    print(resp)