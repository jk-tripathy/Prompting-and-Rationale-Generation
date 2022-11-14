from PCL.dont_patronize_me import DontPatronizeMe

def load_binary():
    dpm = DontPatronizeMe('data/PCL', 'dontpatronizeme_pcl.tsv')
    # This method loads the subtask 1 data
    dpm.load_task1()
    # which we can then access as a dataframe
    df = dpm.train_task1_df
    df.drop(['par_id', 'art_id', 'country', 'orig_label'], axis=1, inplace=True)
    return df

def load_multi():
    dpm = DontPatronizeMe('data/PCL', 'dontpatronizeme_pcl.tsv')
    # This method loads the subtask 1 data
    dpm.load_task2()
    # which we can then access as a dataframe
    df = dpm.train_task2_df
    df.drop(['par_id', 'art_id', 'country',], axis=1, inplace=True)
    return df

if __name__=='__main__':
    df = load_binary()
    print(df.head())
    df = load_multi()
    print(df.head())