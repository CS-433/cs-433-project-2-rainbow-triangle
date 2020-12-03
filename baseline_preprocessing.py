"""Reads raw data and preprocesses it for applying classical ML."""

from constants import *
from classes.preprocessing import Preprocessing


def run_preprocessing(preprocessing, istest=False):
  """
  Runs preprocessing on data.

  If data is test data then no duplicate is dropped since we have to make
  predictions on all data.

  :param preprocessing: specifies data
  :type preprocessing: Preprocessing
  :param istest: specifies if it is test data or not
  :type istest: bool
  """
  # Save the raw tweet for later feature engineering
  preprocessing.save_raw()
  if not istest:
    preprocessing.drop_duplicates()

  preprocessing.remove_tags()
  preprocessing.convert_hashtags()
  preprocessing.slangs_to_words()
  preprocessing.emoticons_to_tags()
  preprocessing.final_paranthesis(use_glove=True)
  preprocessing.remove_numbers()
  preprocessing.remove_punctuation()
  preprocessing.to_lower()
  preprocessing.correct_spelling()
  preprocessing.lemmatize()
  preprocessing.remove_stopwords()
  preprocessing.empty_tweets()
  preprocessing.correct_spacing_indexing()


def main():
  # Read data
  train_preprocessing = Preprocessing([TRAIN_DATA_NEGATIVE, TRAIN_DATA_POSITIVE], submission=False)
  test_preprocessing = Preprocessing([TEST_DATA], submission=True)
  # Preprocess it
  run_preprocessing(train_preprocessing)
  run_preprocessing(test_preprocessing, istest=True)
  # Save it
  train_df = train_preprocessing.get()
  train_df = train_df.sample(frac=1)
  train_df.to_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TRAIN_DATA_CLASSICAL}',
      index=False)
  test_preprocessing.get().to_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TEST_DATA_CLASSICAL}',
      index=False)


if __name__ == '__main__':
  main()
  
