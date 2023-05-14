import json
import random

if __name__ == "__main__":
    path = 'inference/okvqa-flant5-both_5-.jsonl'
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    choices = []
    random.shuffle(json_list)
    for idx, json_str in enumerate(json_list[:25]):
        res = json.loads(json_str)
        print('-' * 25)
        print(f'TAGS: {res["visual_tags"]}')
        print(f'QUESTION {idx}: {res["question"]}')
        print(f'ANSWER {idx}: {res["gold_answer"]}')

        gold = res['gold_rationale'][0]
        gen = res['rationale']

        if random.random() < 0.5:
            print(f'CHOICE 1: {gold}')
            print(f'CHOICE 2: {gen}')

            choice = int(input('Enter your choice: '))
            if choice == 2:
                choices.append(1)
            else:
                choices.append(0)
        else:
            print(f'CHOICE 1: {gen}')
            print(f'CHOICE 2: {gold}')

            choice = int(input('Enter your choice: '))
            if choice == 1:
                choices.append(1)
            else:
                choices.append(0)

    print(sum(choices) / len(choices))
