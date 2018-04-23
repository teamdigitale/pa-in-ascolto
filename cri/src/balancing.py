import pandas as pd
from sklearn.utils import resample


def downsample_majority_class(df_majority, df_minority):
    """This function does the downsampling of the majority class to
    rebalance the dataset. In particular, it randomly (without replacement) 
    draws from the majority class dataset a number of samples that corresponds 
    to the number of samples belonging to the minority class."""
    
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,    # sample without replacement
                                     n_samples=len(df_minority),     # to match minority class
                                     random_state=123) # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # Display new class counts
    print (df_downsampled.review_rating_01.value_counts())
    
    return df_downsampled