import pandas as pd

from classes.baseline import Baseline
from constants import *


def main():
  classifier = Baseline(CLASSICAL_WEIGHTS_PATH)
  # Read preprocessed data
  train = pd.read_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TRAIN_DATA_CLASSICAL}')
  test = pd.read_csv(
      f'{PREPROCESSED_DATA_PATH_CLASSICAL}{PREPROCESSED_TEST_DATA_CLASSICAL}')
  X, Y = classifier.feature_extraction(train)
  X_test, ids = classifier.feature_extraction(test, istest=True)
  # Fit classifier and predict
  #classifier.fit_predict(X, Y, ids, X_test, SUBMISSION_PATH_CLASSICAL, 'KNN') 
  #classifier.fit_predict(X, Y, ids, X_test, SUBMISSION_PATH_CLASSICAL, 'Naive Bayes') 
  classifier.fit_predict(X, Y, ids, X_test, SUBMISSION_PATH_CLASSICAL, 'Logistic Regression') 
  #classifier.fit_predict(X, Y, ids, X_test, SUBMISSION_PATH_CLASSICAL, 'SVM') 
  classifier.fit_predict(X, Y, ids, X_test, SUBMISSION_PATH_CLASSICAL, 'Random Forest') 
  #classifier.fit_predict(X, Y, ids, X_test, SUBMISSION_PATH_CLASSICAL, 'Neural Network') 


if __name__ == '__main__':
  main()
