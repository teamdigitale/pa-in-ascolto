""" In this file you find the code to process the sentix.csv file, which
includes the polarity of a long list of words coming from italian language.

----------------------------------------------------------------------------
The dataset stores both single words and expressions. At this point, we only
take into account the words (not even caring about the gender (i.e. ottimo,
ottima: the dataset contains only ottimo)). It would be a next step the one
of considering both the expressions and the gender. 
----------------------------------------------------------------------------

In the wake of this, would be nice to ask the community to make these steps 
further.


The result of this preprocessing is a json file that gathers the words and 
their polarity (-1, 0, 1) which respectively mean negative, neutral, positive.


----------------------------------------------------------------------------
A general note is that the recalled file are still hardcoded. Thus, if you
want to run it, you need to modify the path of the input file.
-------------------------------------------------------------------------"""

import pandas as pd
import numpy as np

PATH = '../data/sentix'

def assign_label(x):
    """To each word we assign the label:
          * `$POS` if 0.25<`value`<=1
          * `$NEU` if -0.25<`value`<=0.25
          * `$NEG` if -1<`value`<-0.25"""
    
    if 0.25 < x <= 1:
        return "$POS"
    elif -0.25< x <= 0.25:
        return "$NEU"
    elif -1<= x <= -0.25:
        return "$NEG"

vocabolario = pd.read_csv(PATH, sep = '\t', header = None)[[0,5]]

# Assign the labels accordn to our settings
vocabolario[5] = vocabolario[5].apply(assign_label)

zip_vocabolario = {i:j for i,j in list(zip(vocabolario[0], vocabolario[5]))}

import json
with open('../data/lexicon_polarity.json', 'w') as f:
    json.dump(zip_vocabolario, f)

