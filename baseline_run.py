import pandas as pd

from classes.baseline import Baseline
from constants import *
from time import strftime


def main():
  classifier = Baseline()
  # Read preprocessed data
  train = pd.read_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TRAIN_DATA_CLASSICAL}')
  test = pd.read_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TEST_DATA_CLASSICAL}')
  train = train.dropna().reset_index(drop=True)
  train = classifier.feature_extraction(train)
  # Fit classifier
  features = train.columns.difference(['text', 'label'], sort=False)
  X = train[features].values
  Y = train['label'].values
  classifier.fit(X, Y)
  # Make predictions
  test['text'] = test['text'].fillna('EMPTY') # this is a sad bug
  test = classifier.feature_extraction(test, istest=True)
  ids = test['ids'].values
  X = test[features].values
  classifier.predict(ids, X, f'{SUBMISSION_PATH_CLASSICAL}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')


if __name__ == '__main__':
  main()
