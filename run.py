import argparse
from argparse import RawTextHelpFormatter
from enum import Enum
from classes.abstract_model import AbstractModel
from classes.bert import Bert
from classes.gru import Gru
from classes.baseline import Baseline
from classes.ensemble import Ensemble
from classes.preprocessing import Preprocessing
from constants import *
import pandas as pd
from time import strftime


class Models(Enum):
  bert = "bert"
  gru = "gru"
  ensemble = "ensemble"
  mlp = "mlp"
  knn = "knn"
  nbc = "nbc"
  rf = "rf"
  lr = "lr"
  svm = "svm"

  def __str__(self):
    return self.value

  def get_model_name(self):
    list_model = {
      Models.bert: Bert,
      Models.gru: Gru,
      Models.mlp: "Neural Network",
      Models.knn: "KNN",
      Models.nbc: "Naive Bayes",
      Models.rf: "Random Forest",
      Models.lr: "Logistic Regression",
      Models.svm: "SVM",
      Models.ensemble: None
    }

    return list_model[self]


def run_preprocessing(csr: AbstractModel, train_preprocessed_path,
                      test_preprocessed_path):
  # Read data
  train_preprocessing = Preprocessing(
    [TRAIN_DATA_NEGATIVE_FULL, TRAIN_DATA_POSITIVE_FULL],
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


def execute(args, weights_path, train_preprocessed_path, test_preprocessed_path,
            submission_path, **kwargs):

  # Is a classical method is more parameters are specified
  is_classical = len(kwargs) > 0

  if is_classical:
    classifier = Baseline(weights_path)
  else:
    classifier = args.model.get_model_name()(weights_path)

  if not args.lp:
    run_preprocessing(classifier,
                      train_preprocessed_path,
                      test_preprocessed_path)

  if is_classical:
    train_preprocessed = pd.read_csv(train_preprocessed_path,
                                     usecols=['text', 'label', 'raw'])

    # Making the predictions
    test_preprocessed = pd.read_csv(test_preprocessed_path,
                                    usecols=['ids', 'text', 'raw'])
  else:
    train_preprocessed = pd.read_csv(train_preprocessed_path,
                                     usecols=['text', 'label'])

    # Making the predictions
    test_preprocessed = pd.read_csv(test_preprocessed_path,
                                    usecols=['ids', 'text'])

  train_preprocessed.dropna(inplace=True)

  if is_classical:
    X, Y = classifier.feature_extraction(train_preprocessed)
    X_test, test_ids = classifier.feature_extraction(test_preprocessed, istest=True)
  else:
    X, Y = train_preprocessed['text'].values, train_preprocessed['label'].values
    X_test, test_ids = test_preprocessed['text'].values, test_preprocessed['ids'].values

  if args.model == Models.gru:
    # Updating the vocabulary of the classifier according to the training data
    classifier.update_vocabulary(X)

  if args.lt:
    classifier.predict(
      test_ids, X_test,
      f'{submission_path}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv',
      **kwargs)

  else:
    classifier.fit_predict(
      X, Y, test_ids, X_test,
      f'{submission_path}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv',
      **kwargs)


if __name__ == '__main__':

  # For the classical ML methods, when the "-lt" (load trained) option
  # is not specificied, we automatically perform a cross validation to find
  # the best hyperparameters for the model
  parser = argparse.ArgumentParser(
    description="This script performs a classification task to predict if " \
                "a tweet message used to contain a positive :) or negative " \
                ":( smiley,by considering only the remaining text.",
    formatter_class=RawTextHelpFormatter)

  # Required argument
  parser.add_argument(
    "model",
    type=Models,
    choices=list(Models),
    help="Specify the model you want to run.\nNote: for classical ML models " \
         "(every model excluded Bert and GRU), if -lt is not specified, " \
         "before the training phase we perform the hyperparameters tuning\n" \
         "  bert: performs the classification with a Bert model (we suggest " \
         "you to train this model on a cloud platform).\n  gru: performs the " \
         "classification with a GRU bidirectional model.\n  mlp: performs the " \
         "classification with a multi-layer perceptron neural network \n" \
         "  knn: performs the classification with a K-nearest neighbors " \
         "classifier\n  nbc: performs the classification with a Naive Bayes " \
         "classifier\n  rf: performs the classification with a Random Forest " \
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
    execute(args,
            BERT_WEIGHTS_PATH,
            f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TRAIN_DATA_BERT}',
            f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TEST_DATA_BERT}',
            SUBMISSION_PATH_BERT)

  elif args.model == Models.gru:
    execute(args,
            GRU_WEIGHTS_PATH,
            f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TRAIN_DATA_GRU}',
            f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TEST_DATA_GRU}',
            SUBMISSION_PATH_BERT)

  elif args.model == Models.ensemble:
    # Names of the models you want to use for ensembling
    model_names = ["Gru", "Bert_no_prep", "Bert_with_prep", "KNN", "Logistic_Regression", "Naive_Bayes", "Random_Forest", "Multilayer_Perceptron", "SVM"]
    # Dictionary with the submissions of those models and their respective validation accuracy
    model_accuracies = {
      f"{SUBMISSION_PATH_GRU}submission-2020-12-03_16:55:08.csv": 0.857,
      f"{SUBMISSION_PATH_BERT}submission-2020-12-06_16:48:30.csv": 0.894,
      f"{SUBMISSION_PATH_BERT}submission-2020-12-03_20:24:31.csv": 0.888,
      f"{SUBMISSION_PATH_CLASSICAL}submission-KNN-2020-12-08_23:37:01.csv": 0.674,
      f"{SUBMISSION_PATH_CLASSICAL}submission-Logistic Regression-2020-12-09_07:56:20.csv":0.765,
      f"{SUBMISSION_PATH_CLASSICAL}submission-Naive Bayes-2020-12-08_20:28:39.csv":0.642,
      f"{SUBMISSION_PATH_CLASSICAL}submission-Random Forest-2020-12-09_09:30:11.csv": 0.766,
      f"{SUBMISSION_PATH_CLASSICAL}submission-Neural Network-2020-12-09_04:42:17.csv": 0.776,
      f"{SUBMISSION_PATH_CLASSICAL}submission-SVM-2020-12-08_20:03:39.csv": 0.765
    }
    # Instantiating the model
    ensemble_model = Ensemble(model_accuracies, model_names)

    # Predicting
    ensemble_model.predict(f'{SUBMISSION_PATH_ENSEMBLE}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')

  else:
    execute(args,
            CLASSICAL_WEIGHTS_PATH,
            f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TRAIN_DATA_CLASSICAL}',
            f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TEST_DATA_CLASSICAL}',
            SUBMISSION_PATH_CLASSICAL, model_name=args.model.get_model_name())
