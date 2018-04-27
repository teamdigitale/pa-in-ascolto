# First push without data. Probs with git
# Sentiment Analysis

The aim of this model is to be able to classify italian tweets as positive or negative.
With this purpose, a set of dataset have been selected to train the model. 
Underlining the difficulty of finding huge collection of tweets in italian, we managed to get dataset labeled according to the sentiment that vary among reviews and tweets.

At this initial step, we take into account two dataset:

1) A set of `TripAdvisor` reviews written in italian.

2) A set of *politic* tweets of 2016.

__Note:__ for the sake of Git and the user that clone the repository, datasets are compressed (you need to decompress before executing). On the other side we specidy here that the modules used in the ipynb are stored in the folder `src`.

## 1. Results

Results are shown in the `.ipynb`. Namely:

* `Convolutional Neural Network on Tripadvisor`
> So far, *no fine tuning* of parameters has been done. The parameters have been choosen as reasonable for the problem setting. The table below reports the scores of the model on a set of validation samples that have not been seen by the model in any step of the training phase. Everything is reported [here](notebooks/tripadvisor_cnn.ipynb).

|               Model              	| Accuracy 	| Precision 	| Recall 	|  F1  	|
|:--------------------------------:	|:--------:	|:---------:	|:------:	|:----:	|
| Pre-processing + Embedding + CNN 	|   0.92   	|    0.91   	|  0.93  	| 0.92 	|




* `Convolutional Neural Network on Twitter`
> As before, no fine tuning so far. Below the metrics related to this model. As we point out in the notebook, the low amount of data we train the model on implies a bad robustness of the model. Everything is reported [here](tweet_2016_preproc_and_model.ipynb).

|               Model              	| Accuracy 	| Precision 	| Recall 	|  F1  	|
|:--------------------------------:	|:--------:	|:---------:	|:------:	|:----:	|
| Pre-processing + Embedding + CNN 	|   0.73   	|    0.74   	|  0.72  	| 0.73 	|



## 2. Pre-processing

In this section we explain the pre-processing strategies we apply.


### 2.1 TripAdvisor
Before illustrate the process done for the TripAdvisor dataset, we briefely describe the dataset. In particular, it is composed by ~220k reviews about BnB in Rome. Among those we only keep the ones in italian (~60k). For each review we know:

* The rating given by the reviewer
* The text of the review
* The title

#### 2.1.1 Labels pre-processing
The original [dataset](data/raw/tripadvisor/reviews.csv) provides a rating between 1 and 5 for each review. 
Thus, we suppose that the reviews with rating, *r*:

* Equal to 1 and 2 are `negative`
* Equal to 3 are `neutral`
* Equal to 4 and 5 are `positive`

For the model, we use only the samples defined as `positive` or `negative`.

#### 2.1.2 Dataset pre-processing

First of all we normalize the text removing the punctuation and putting everything to the lower case.

Afterwords, we exploit a dataset that indicates the polarity of a large list of italian words and expressions. In particular, each word of the reviews has been mapped to the flags `$POS` or `$NEG`. We decide to do that in order to allow the model to identify *"immediately"* the precence of positive/negative words in the reviews. 

The preprocessing is in [this](notebooks/tripadvisor_preprocessing.ipynb) notebook.

### 2.2 Twitter Sentipolc 2016

The Twitter [dataset](data/raw/twitter-sentpo) is made by ~6k tweets that are labeled according to their irony and polarity. For each tweet we know the text.

#### 2.2.1 Labels pre-processing

The label pre-processing for this dataset implies the reduction of the number of tweets to use. In particular, due to the fact that the labels refer both to the polarity and the irony, a tweet is not necessarely labeled as positive or negative. Thus, we keep only the tweets that are labeled either as `positive` or `negative` (the procedure to identify these tweets is reported in the [notebooks](notebooks/tweet_2016_preproc_and_model.ipynb)).

#### 2.2.2 Dataset pre-processing

The pre-process strategy is the same of the TripAdvisor dataset. We just add one step, which is specific for tweets and consists in substituting words with tags when the are `URL, MENTIONS and HASHTAGS`.


## Give us your support! 

As we stated there are few things that might be done to make better our model. We can distinguish among three branches:

1) `Increase the lexicon`: we ask you to increase the `lexicon_polarity.json` file adding to the file new words and the respective polarity or .

2) `Regex`: the lexicon contains some regular expressions. To make them part of our preprocessing we need to define the regular expression for them.

3) `Label tweets`: to improve the performances of the model, we need more labeled data! 


