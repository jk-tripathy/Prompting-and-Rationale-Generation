from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from time import time
from transformers import pipeline

#model_id = 'bigscience/bloom'
model_id = 'EleutherAI/gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", offload_folder="offload", torch_dtype=torch.float16)

pipe = pipeline('text-generation',model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device=0)

def local_inf(prompt, temperature=0.7, top_p=None, max_new_tokens=32, repetition_penalty=None, do_sample=False, num_return_sequences=1):
    response = pipe(
        f"{prompt}",
        temperature = temperature, # 0 to 1
        top_p = top_p, # None, 0-1
        max_new_tokens = max_new_tokens, # up to 2047 theoretically
        return_full_text = False, # include prompt or not.
        repetition_penalty = repetition_penalty, # None, 0-100 (penalty for repeat tokens.
        do_sample = do_sample, # True: use sampling, False: Greedy decoding.
        num_return_sequences = num_return_sequences
    )
    return print(prompt + response[0]['generated_text']), response[0]['generated_text']

inp = "Can cats fly?"
t = time()
resp = local_inf(inp, max_new_tokens=64)
delta = time() - t
print("Inference took %0.2f s." % delta)
print(resp)