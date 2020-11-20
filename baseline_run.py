"""Reads raw data, preprocesses it and does feature extraction."""

import numpy as np

from constants import *
from preprocessing import Preprocessing


def run_preprocessing(data_preprocessing, istest=False):
  """Runs preprocessing on data.

  If data is test data then no duplicate is dropped since we have to make
  predictions on all data.

  Args:
    data_preprocessing: A preprocessing object which encapsulates the data.
    istest: A bool specifying if the data is test data or not.
  """
  if not istest:
    data_preprocessing.drop_duplicates()
  data_preprocessing.slangs_to_words()
  # TODO: there are emoticons not in the list
  # Should we put them in tags rather than words? So tha we don't affect
  # context based features.
  data_preprocessing.emoticons_to_words()
  #TODO: should deal with time: 10:30- 2:30
  data_preprocessing.to_lower()
  data_preprocessing.emoticons_to_words()
  data_preprocessing.remote_hashtags()
  data_preprocessing.remove_tags()
  data_preprocessing.remove_numbers()
  data_preprocessing.remove_stopwords()
  data_preprocessing.lemmatize()
  data_preprocessing.correct_spacing_indexing()


def run_feature_extraction(data_preprocessing):
  data_preprocessing.add_tfidf()


def main():
  # Read data
  train_preprocessing = Preprocessing([TRAIN_DATA_NEGATIVE, TRAIN_DATA_POSITIVE], submission=False)
  test_preprocessing = Preprocessing([TEST_DATA], submission=True)
  # Preprocess it
  run_preprocessing(train_preprocessing)
  run_preprocessing(test_preprocessing)
  # Features
  run_feature_extraction(train_preprocessing)
  #TODO: vocabulary needs to be the same
  run_feature_extraction(test_preprocessing)
  # Save it
  # train_preprocessing.get().to_csv(PREPROCESSED_TRAIN_DATA)
  # test_preprocessing.get().to_csv(PREPROCESSED_TEST_DATA)


if __name__ == '__main__':
  main()
  
