from sklearn.utils import resample
import pandas as pd
import numpy as np
import json
from keras.preprocessing import sequence

with open('../data/output/tripadvisor/vocabolario.json', 'r') as f:
    vocab_index = json.load(f)
    
with open('../data/output/twitter-sentpol/vocabolario_twitter.json', 'r') as f:
    vocab_index_tweet = json.load(f)


def create_balanced_validation(df, majority_class, percentage = 5):
    """ This function returns the validation set built s.t. the
    number of items that belong to class 0 is the same of those
    belonging to class 1
    
    INPUT:
    @majority_class: int 0 or 1
    @percentage: percentage choosen for the validation
    
    Returns:
    @df_validation
    @df_majority
    @df_minority"""

    # Separiamo le due classi
    df_majority = df[df.review_rating_01 == majority_class]
    df_minority = df[df.review_rating_01 == 1-majority_class]

    # Facciamo sampling senza ripetizione
    df_minority_sample_validation = resample(df_minority, 
                                     replace=False,     # sample with replacement
                                     n_samples=int(len(df)/100*percentage/2),    # to match majority class
                                     random_state=123)

    df_majority_sample_validation = resample(df_majority, 
                                     replace=False,     # sample with replacement
                                     n_samples=int(len(df)/100*percentage/2),    # to match majority class
                                     random_state=123)


    df_validation = pd.concat([df_minority_sample_validation, df_majority_sample_validation])
    
    # Remove the items in the validation
    df_majority = df_majority.drop(df_majority_sample_validation.index)
    df_minority = df_minority.drop(df_minority_sample_validation.index)
    
    return df_validation, df_majority, df_minority


def train_validation_tripadvisor(df_downsampled, df_validation):
    """This function returns the matrices of the train and the validation set.
    In particular, the x matrices are padded. Moreover, we get the length of the
    longest text and the max id of the words.
    
    Input:
    @df_downsampled: df obtained by `downsample_majority_class` function
    @df_validation: validation det"""
    
    x_train = np.array(df_downsampled['review_text_token_flag_index_list'])
    y_train = np.array(df_downsampled['review_rating_01'])

    x_validation = np.array(df_validation['review_text_token_flag_index_list'])
    y_validation = np.array(df_validation['review_rating_01'])
    
    # Check the max length
    max_len_seq = max([len(x) for x in x_train])
    print('max len seq {}'.format(max_len_seq))
    max_idx = max(vocab_index.values())
    print('max id {}'.format(max_idx))
    
    
    # Pad the sequences
    x_train_pad = sequence.pad_sequences(x_train, maxlen=max_len_seq, padding='post')
    x_validation_pad = sequence.pad_sequences(x_validation, maxlen=max_len_seq, padding='post')

    
    return x_train_pad, y_train, x_validation_pad, y_validation, max_len_seq, max_idx


def train_validation_twitter(df_downsampled, df_validation):
    """This function returns the matrices of the train and the validation set.
    In particular, the x matrices are padded. Moreover, we get the length of the
    longest text and the max id of the words.
    
    Input:
    @df_downsampled: df obtained by `downsample_majority_class` function
    @df_validation: validation det"""
    
    x_train = np.array(df_downsampled['review_text_token_flag_index_list'])
    y_train = np.array(df_downsampled['review_rating_01'])

    x_validation = np.array(df_validation['review_text_token_flag_index_list'])
    y_validation = np.array(df_validation['review_rating_01'])
    
    # Check the max length
    max_len_seq = max([len(x) for x in x_train])
    print('max len seq {}'.format(max_len_seq))
    max_idx = max(vocab_index_tweet.values())
    print('max id {}'.format(max_idx))
    
    
    # Pad the sequences
    x_train_pad = sequence.pad_sequences(x_train, maxlen=max_len_seq, padding='post')
    x_validation_pad = sequence.pad_sequences(x_validation, maxlen=max_len_seq, padding='post')

    
    return x_train_pad, y_train, x_validation_pad, y_validation, max_len_seq, max_idx