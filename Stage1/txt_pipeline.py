from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


def text_pipeline(model_id, prompts, temperature=0.7, max_new_tokens=32, do_sample=False, batch_size=5):
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
        temperature = temperature, # 0 to 1
        max_new_tokens = max_new_tokens, # up to 2047 theoretically
        do_sample = do_sample, # True: use sampling, False: Greedy decoding.
    )
    return response

if __name__=='__main__':
    model_id = 'EleutherAI/gpt-j-6B'
    inp = "Can cats fly?"
    resp = text_pipeline(model_id, inp, max_new_tokens=64)
    print(resp)