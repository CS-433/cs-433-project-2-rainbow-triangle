import pandas as pd

from classes.baseline import Baseline
from constants import *


def main():
  classifier = Baseline()
  # Read preprocessed data
  train = pd.read_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TRAIN_DATA_CLASSICAL}')
  test = pd.read_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TEST_DATA_CLASSICAL}')
  X, Y = classifier.feature_extraction(train)
  # Fit classifier
  print(X.shape)
  classifier.fit(X, Y)
  # Make predictions
  X, ids = classifier.feature_extraction(test, istest=True)
  print('Test', X.shape)
  classifier.predict(ids, X, f'{SUBMISSION_PATH_CLASSICAL}')
  # Save models
  classifier.save_best_models(CLASSICAL_WEIGHTS_PATH)


if __name__ == '__main__':
  main()
