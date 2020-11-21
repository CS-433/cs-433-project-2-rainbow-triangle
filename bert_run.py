import pandas as pd
from classes.bert import Bert
from constants import *
from time import strftime

if __name__ == '__main__':
  # Instantiating the classifier (be sure to create the folder!)
  classifier = Bert(BERT_WEIGHTS_PATH)

  # Training the model
  for i in range(N_SPLITS):

    train_preprocessed = pd.read_csv(
      f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TRAIN_DATA_PREFIX_BERT}{i}{PREPROCESSED_TRAIN_DATA_SUFFIX_BERT}',
      usecols=['text', 'label'])

    train_preprocessed.dropna(inplace=True)

    X = train_preprocessed['text'].values
    Y = train_preprocessed['label'].values

    classifier.fit(X, Y, batch_size=64)

  # Making the predictions
  test_preprocessed = pd.read_csv(
    f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TEST_DATA_BERT}',
    usecols=['ids', 'text'])

  ids = test_preprocessed['ids'].values
  X = test_preprocessed['text'].values

  classifier.predict(ids, X, f'{SUBMISSION_PATH_BERT}submission-{strftime("%Y-%m-%d %H:%M:%S")}.csv')