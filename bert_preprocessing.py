from classes.preprocessing import Preprocessing
from constants import *
import numpy as np


def bert_preprocessing(preprocessing, istest=False):
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
  preprocessing.correct_spacing_indexing()

  return preprocessing


# Preprocessing the train data
train_preprocessing = Preprocessing([TRAIN_DATA_NEGATIVE_FULL, TRAIN_DATA_POSITIVE_FULL], submission=False)
train_preprocessing = bert_preprocessing(train_preprocessing)
train_df = train_preprocessing.get()

train_df = train_df.sample(frac=1)

for i, df in enumerate(np.array_split(train_df, N_SPLITS)):
  df.to_csv(f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TRAIN_DATA_PREFIX_BERT}{i}{PREPROCESSED_TRAIN_DATA_SUFFIX_BERT}')

# Preprocessing the test data
test_preprocessing = Preprocessing([TEST_DATA], submission=True)
test_preprocessing = bert_preprocessing(test_preprocessing, istest=True)
test_preprocessing.get().to_csv(f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TEST_DATA_BERT}')
