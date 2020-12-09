import argparse
from argparse import RawTextHelpFormatter
from enum import Enum
from classes.abstract_model import AbstractModel
from classes.bert import Bert
from classes.preprocessing import Preprocessing
from constants import *
import pandas as pd
from time import strftime


class Models(Enum):
  bert = "bert"
  gru = "gru"
  mlp = "mlp"
  knn = "knn"
  nbc = "nbc"
  rf = "rf"
  lr = "lr"

  def __str__(self):
    return self.value


def run_preprocessing(csr : AbstractModel, train_preprocessed_path, test_preprocessed_path):
  # Read data
  train_preprocessing = Preprocessing(
    [TRAIN_DATA_NEGATIVE, TRAIN_DATA_POSITIVE],
    submission=False)

  test_preprocessing = Preprocessing([TEST_DATA], submission=True)

  # Preprocess it
  for method in csr.get_preprocessing_methods(istest=False):
    getattr(train_preprocessing, method)()

  for method in csr.get_preprocessing_methods(istest=True):
    getattr(test_preprocessing, method)()

  # Save it
  train_df = train_preprocessing.get()
  train_df = train_df.sample(frac=1)

  train_df.to_csv(train_preprocessed_path, index=False)
  test_preprocessing.get().to_csv(test_preprocessed_path, index=False)


def get_train_test(train_preprocessed_path, test_preprocessed_path):
  train_preprocessed = pd.read_csv(train_preprocessed_path,
                                   usecols=['text', 'label'])

  # Making the predictions
  test_preprocessed = pd.read_csv(test_preprocessed_path,
                                  usecols=['ids', 'text'])

  train_preprocessed.dropna(inplace=True)

  X = train_preprocessed['text'].values
  Y = train_preprocessed['label'].values

  test_ids = test_preprocessed['ids'].values
  X_test = test_preprocessed['text'].values

  return X, Y, X_test, test_ids


if __name__ == '__main__':

  # For the classical ML methods, when the "-lt" (load trained) option
  # is not specificied, we automatically perform a cross validation to find
  # the best hyperparameters for the model
  parser = argparse.ArgumentParser(
    description="This script performs a classification task to predict if "\
                "a tweet message used to contain a positive :) or negative "\
                ":( smiley,by considering only the remaining text.",
    formatter_class=RawTextHelpFormatter)

  # Required argument
  parser.add_argument(
    "model",
    type=Models,
    choices=list(Models),
    help="Specify the model you want to run.\nNote: for classical ML models "\
         "(every model excluded Bert and GRU), if -lt is not specified, "\
         "before the training phase we perform the hyperparameters tuning\n"\
         "  bert: performs the classification with a Bert model (we suggest "\
         "you to train this model on a cloud platform).\n  gru: performs the "\
         "classification with a GRU bidirectional model.\n  mlp: performs the "\
         "classification with a multi-layer perceptron neural network \n"\
         "  knn: performs the classification with a K-nearest neighbors "\
         "classifier\n  nbc: performs the classification with a Naive Bayes "\
         "classifier\n  rf: performs the classification with a Random Forest "\
         "classifier\n  lr: performs the classification with Logistic Regression")

  # Optional arguments
  parser.add_argument(
    "-lp",
    action='store_true',
    help="Load already preprocessed data for a specified model")

  parser.add_argument(
    "-lt",
    action='store_true',
    help="Load an already trained model")

  args = parser.parse_args()
  # parser.print_help()

  if args.model == Models.bert:
    classifier = Bert(BERT_WEIGHTS_PATH)

    train_preprocessed_path = f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TRAIN_DATA_BERT}'
    test_preprocessed_path = f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TEST_DATA_BERT}'

    if not args.lp:
      run_preprocessing(classifier,
                        train_preprocessed_path,
                        test_preprocessed_path)

    X, Y, X_test, test_ids = get_train_test(train_preprocessed_path,
                                            test_preprocessed_path)

    if args.lt:
      classifier.predict(
        test_ids, X_test,
        f'{SUBMISSION_PATH_BERT}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv',
        from_weights=True)

    else:
      classifier.fit_predict(
        X, Y, test_ids, X_test,
        f'{SUBMISSION_PATH_BERT}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')


