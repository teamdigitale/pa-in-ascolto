""" This file gathers useful modules to preprocess the 
trip advisor reviews dataset"""

import pandas as pd


##### Filter out the reviews not written in italian
def keep_italian_reviews(PATH = '../data/raw/tripadvisor/reviews.csv', separator = ','):
    """This function keep the rows related to italian reviews. Moreover, 
    it preserves only the column of interest:
    'review_title': title of the review
    'review_text': text of the review
    'review_rating': rating given by the reviewer
    
    --------------------------------------------------------------------
    The file to process is hardcoded. Thus, to run it substiture the right
    path.
    """
   
    # Import dataset
    df = pd.read_csv(PATH, sep = separator)

    # Keep reviews in italian
    df_it = df[df['review_language']=='it'][['review_title', 'review_text', 'review_rating']]

    print ('Numero di recensioni totali in italiano: ', df_it.shape[0], '\nNumero di recensioni totali:', df.shape[0])
    
    df_it.to_csv('data/raw/tripadvisor/reviews.csv', sep = ',')
    
    return df_it


def label(x):
    """The function, applied to the ratings db's column, returns a score between -1 and 1. 
    Specifically we do this mapping:
    1,2 ---> -1: negative
    3 ---> 0: neutro
    4,5 ---> 1: positive
    
    -----------------------------------------------------------------------------------
    The assumption behind the mapping is that the review reflects the """
    
    if x == 1 or  x == 2:
        return -1
    elif x == 3:
        return 0
    elif x == 4 or x == 5:
        return 1
    