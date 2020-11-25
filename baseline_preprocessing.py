"""Reads raw data and preprocesses it for applying classical ML."""

from constants import *
from classes.preprocessing import Preprocessing


def run_preprocessing(data_preprocessing, istest=False):
  """
  Runs preprocessing on data.

  If data is test data then no duplicate is dropped since we have to make
  predictions on all data.

  :param data_preprocessing: specifies data
  :type data_preprocessing: Preprocessing
  :param istest: specifies if it is test data or not
  :type istest: bool
  """
  if not istest:
    data_preprocessing.drop_duplicates()
  data_preprocessing.remove_tags()
  data_preprocessing.convert_hashtags()
  data_preprocessing.slangs_to_words()
  data_preprocessing.remove_parenthesis()
  data_preprocessing.emoticons_to_sentiment()
  data_preprocessing.remove_numbers()
  data_preprocessing.remove_punctuation()
  data_preprocessing.to_lower()
  data_preprocessing.correct_spelling()
  data_preprocessing.lemmatize()
  data_preprocessing.remove_stopwords()
  data_preprocessing.correct_spacing_indexing()


def main():
  # Read data
  train_preprocessing = Preprocessing([TRAIN_DATA_NEGATIVE, TRAIN_DATA_POSITIVE], submission=False)
  test_preprocessing = Preprocessing([TEST_DATA], submission=True)
  # Preprocess it
  run_preprocessing(train_preprocessing)
  run_preprocessing(test_preprocessing)
  # Save it
  train_preprocessing.get().to_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TRAIN_DATA_CLASSICAL}')
  test_preprocessing.get().to_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TEST_DATA_CLASSICAL}')


if __name__ == '__main__':
  main()
  
