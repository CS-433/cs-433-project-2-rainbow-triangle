from classes import MyPreprocessing
from constants import *
import numpy as np


def bert_preprocessing(preprocessing):
  """
  Define the preprocessing operations to train the model with Bert.

  :param preprocessing: specifies data
  :type preprocessing: MyPreprocessing
  :return: preprocessed data
  :rtype: MyPreprocessing
  """
  preprocessing.drop_duplicates()
  preprocessing.to_lower()
  preprocessing.remove_tags()
  preprocessing.correct_spacing()

  return preprocessing


# Preprocessing the train data
train_preprocessing = MyPreprocessing([TRAIN_DATA_NEGATIVE_FULL, TRAIN_DATA_POSITIVE_FULL], submission=False)
train_preprocessing = bert_preprocessing(train_preprocessing)
train_df = train_preprocessing.get()

train_df = train_df.sample(frac=1)

for i, df in enumerate(np.array_split(train_df, N_SPLITS)):
  df.to_csv(f'{PREPROCESSED_TRAIN_DATA_PREFIX}{i}{PREPROCESSED_TRAIN_DATA_SUFFIX}')

# Preprocessing the test data
test_preprocessing = MyPreprocessing([TEST_DATA], submission=True)
train_preprocessing = bert_preprocessing(test_preprocessing)
test_preprocessing.get().to_csv(PREPROCESSED_TEST_DATA)
