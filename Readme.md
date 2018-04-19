# Sentiment Classification Model for Twitter

## Datasets

1. [sentipolc / sentiment polarity classification 2016](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/index.html)

## Preprocessing

### Sentipolc 2016

The steps are described at [sentipolc-preprocessing](./sentipolc-preprocessing.ipynb). We have preprocessed the tweets with [p/processor](https://github.com/s/preprocessor#available-options). We replaced URL, MENTION, HASHTAG, EMOJi, and NUMBER with keywords. The list of positive and negative emoticons.
Final results are stored into the folder data:

1. `sentipolc.npz` contains data preprocesed and stored as a dictionary with keys x_train, y_train, x_test, y_test
2. `sentipolc_seq.npz` contains the same data processed replacing the words with ids. Its word index dictionary is stored at `sentipolc_word_index.json`
3. `sentipolc_char_seq.npq` contains the same data processed at character level. Its char index dictionary is stored at `sentipolc_char_index.json`

Further details can be found in the notbook.
