import sys
sys.path.insert(0, "../")
from data.load_PCL import load_binary, load_multi
from txt_pipeline import text_pipeline

def split_data(df):
    if df['label'].dtype != object:
        return [y for x, y in df.groupby(['label'], axis=0)]
    else:
        pass

def prompt_gen(dfs, class_dict, prompt_len=1, val_len=1):
    train_prompts = 'Answer with Yes or No.\n'
    val_prompts = []
    for i, df in enumerate(dfs):
        train_df = df[:int(len(df)*0.8)]
        val_df = df[-val_len:]
        for j in range(prompt_len):
            inp = f'Prompt: {train_df.iloc[j]["text"]} Is this Patronizing?\n'
            inp = inp + f'Answer: {class_dict[i]}.\n'
            train_prompts = train_prompts + inp
        for j in range(val_len):
            inp = f'Prompt: {val_df.iloc[j]["text"]} Is this Patronizing?\n'
            inp = inp + f'Answer: '
            val_prompts.append(inp)
    prompts = []
    for elem in val_prompts:
        prompts.append(train_prompts + f'{elem}')
    return prompts
        
if __name__=='__main__':
    df = load_binary('../data/PCL')
    df_split = split_data(df)
    model_id = 'EleutherAI/gpt-j-6B'
    #for i in [1, 2, 5, 10, 15, 20]:
        #prompts = prompt_gen(df_split, {0:"No", 1:"Yes"}, prompt_len=1, val_len=5)
        #responses = local_inf(model_id, prompts, max_new_tokens=1, batch_size=5)
        #with open(f'outputs/main_prompt_batched_prompt_len_{i}.json', 'w', encoding='utf-8') as f:
            #json.dump(responses, f, ensure_ascii=False, indent=4)
        #for resp in responses:
            #print(resp[0]['generated_text'])

    prompt = [
        "Prompt: We 're living in times of absolute insanity , as I 'm pretty sure most people are aware . For a while , waking up every day to check the news seemed to carry with it the same feeling of panic and dread that action heroes probably face when they 're trying to decide whether to cut the blue or green wire on a ticking bomb -- except the bomb 's instructions long ago burned in a fire and imminent catastrophe seems the likeliest outcome . It 's hard to stay that on-edge for that long , though , so it 's natural for people to become inured to this constant chaos , to slump into a malaise of hopelessness and pessimism . Is this Patronizing?\nAnswer: No.",
        "Prompt: Arshad said that besides learning many new aspects of sports leadership he learnt how fast-developing nations were using sports as a tool of development and in this effort the disabled and the underprivileged were not left behind at any stage . Is this Patronizing?\nAnswer: Yes.",
        "Prompt: He added that the AFP will continue to bank on the application of the whole of nation -- whole of government approach , which involves the use of sustainable and peaceful engagements with vulnerable communities to protect them from further NPA exploitation . Is this Patronizing?\nAnswer: "
    ]
    resp = text_pipeline(model_id, prompt, max_new_tokens=1, batch_size=1)
    print(resp)
    

