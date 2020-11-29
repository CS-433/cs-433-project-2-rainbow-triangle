from classes.preprocessing import Preprocessing
from constants import *


def gru_preprocessing(preprocessing, istest=False):
  """
  Define the preprocessing operations to train the model with Bert.

  :param preprocessing: specifies data
  :type preprocessing: Preprocessing
  :param istest: specifies if it is test data or not
  :type istest: bool
  """
  if not istest:
    preprocessing.drop_duplicates()
  preprocessing.remove_endings()
  preprocessing.emoticons_to_tags()
  preprocessing.final_paranthesis(use_glove=True)
  preprocessing.numbers_to_tags()
  preprocessing.hashtags_to_tags()
  preprocessing.repeat_to_tags()
  preprocessing.elongs_to_tags()
  preprocessing.to_lower()
  preprocessing.correct_spacing_indexing()


def main():
  # Preprocessing the train data
  train_preprocessing = Preprocessing([TRAIN_DATA_NEGATIVE, TRAIN_DATA_POSITIVE], submission=False)
  gru_preprocessing(train_preprocessing)
  train_df = train_preprocessing.get()
  #train_df = train_df.sample(frac=1)
  train_df.to_csv(f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TRAIN_DATA_GRU}bbb',
                  index=False)
  # Preprocessing the test data
  test_preprocessing = Preprocessing([TEST_DATA], submission=True)
  gru_preprocessing(test_preprocessing, istest=True)
  test_preprocessing.get().to_csv(f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TEST_DATA_GRU}bbb',
                                  index=False)

if __name__ == '__main__':
  main()
