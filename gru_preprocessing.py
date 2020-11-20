from preprocessing import Preprocessing
from constants import *
import numpy as np


def gru_preprocessing(preprocessing, istest=False):
  """
  Define the preprocessing operations to train the model with Bert.

  :param preprocessing: specifies data
  :type preprocessing: Preprocessing
  :param istest: specifies if it is test data or not
  :type istest: bool
  :return: preprocessed data
  :rtype: Preprocessing
  """
  if not istest:
    preprocessing.drop_duplicates()
  preprocessing.to_lower()
  preprocessing.remove_tags()
  preprocessing.remove_punctuation()
  preprocessing.correct_spacing_indexing()

  return preprocessing


# Preprocessing the train data
train_preprocessing = Preprocessing([TRAIN_DATA_NEGATIVE_FULL, TRAIN_DATA_POSITIVE_FULL], submission=False)
train_preprocessing = gru_preprocessing(train_preprocessing)
train_df = train_preprocessing.get()

train_df = train_df.sample(frac=1)
train_df.to_csv(f'{PREPROCESSED_FILES_PATH}{PREPROCESSED_TRAIN_DATA_GRU}')

# Preprocessing the test data
test_preprocessing = Preprocessing([TEST_DATA], submission=True)
test_preprocessing = gru_preprocessing(test_preprocessing, istest=True)
test_preprocessing.get().to_csv(f'{PREPROCESSED_FILES_PATH}{PREPROCESSED_TEST_DATA_GRU}')