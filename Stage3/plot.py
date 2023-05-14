import pandas as pd
import matplotlib.pyplot as plt


def data(row):
    datasets = ['aokvqa', 'esnli', 'okvqa', 'senmaking']
    for ds in datasets:
        if ds in row['file']:
            return ds


def model(row):
    models = ['dolly', 'flant5', 'glm', 'opt', 't0pp']
    for m in models:
        if m in row['file']:
            return m


def batch(row):
    for i in range(6):
        if f'_{i}' in row['file']:
            return int(i)


def visualTags(row):
    vistag = ['both', 'caption', 'tags']
    for vt in vistag:
        if vt in row['file']:
            return vt
    return None


if __name__ == "__main__":
    full_df = pd.read_json('results.jsonl', lines=True)
    full_df['dataset'] = full_df.apply(lambda row: data(row), axis=1)
    full_df['model'] = full_df.apply(lambda row: model(row), axis=1)
    full_df['batch'] = full_df.apply(lambda row: batch(row), axis=1)
    full_df['vistag'] = full_df.apply(lambda row: visualTags(row), axis=1)

    full_df.drop(['file'], inplace=True, axis=1)
    full_df = full_df[['dataset', 'model', 'batch', 'vistag', 'accuracy', 'bleu', 'meteor', 'rouge']]

    data_df = full_df.groupby('dataset')
    datasets = {
        'aokvqa': 'AOKVQA',
        'esnli': 'E-SNLI',
        'okvqa': 'OKVQA',
        'senmaking': 'SEN-MAKING'
    }
    metrics = {
        'accuracy': 'Accuracy',
        'bleu': 'BLEU',
        'meteor': 'METEOR',
        'rouge': 'ROUGE'
    }
    models = {
        'dolly': 'Dolly-v2-12b',
        'flant5': 'FLAN-T5-xxl',
        'glm': 'GLM-10b',
        'opt': 'OPT-66b',
        't0pp': 'T0pp'
    }

    img_full_df = pd.concat([data_df.get_group('aokvqa'), data_df.get_group('okvqa')])
    img_dfs = img_full_df.groupby('dataset')

    for data_key in img_dfs.groups.keys():
        model_dfs = data_df.get_group(data_key).groupby('model')
        for metric in metrics.keys():
            print(f'{datasets[data_key]} - {metrics[metric]}')
            fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 3), nrows=1, ncols=3)
            vistag = ['both', 'caption', 'tags']
            for col, model_key in enumerate(model_dfs.groups.keys()):
                ax[col].title.set_text(f'{models[model_key]}')
                tag_df = model_dfs.get_group(model_key).groupby('vistag')
                for tag_key in tag_df.groups.keys():
                    ax[col].plot(range(6), tag_df.get_group(tag_key)[metric], label=tag_key)
                    ax[col].set(xlabel='In-context Samples', ylabel=f'{metrics[metric]}')
                    ax[col].set_xticks(range(6))
            ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
            plt.savefig(f'plots/{data_key}_{metric}.png')
    text_full_df = pd.concat([data_df.get_group('esnli'), data_df.get_group('senmaking')])
    text_dfs = text_full_df.groupby('dataset')
