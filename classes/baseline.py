import numpy as np
import pandas as pd

from classes.abstract_model import AbstractModel
from constants import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class Baseline(AbstractModel):
  """Does feature extraction, training and inference for various classic models.

  The non-deep-learning models used are:
    * Naive Bayes
    * Logistic Regression
    * SVM with rbf kernel
    * Random Forest
    * NN - multi layer perceptron
  """
  def __init__(self):
    super().__init__(None) 
    self.init_models()

  def init_models(self):
    # Initialize the sklearn models here and their respective hyperparameters
    # grid for grid search with cross validation in training.
    # Some defaults are mentioned since they are important.
    self.__best_models = []
    self.__models = {
      'Naive Bayes': (GaussianNB(), {'var_smoothing': np.logspace(-12, 0, 11)}),
      'Logistic Regression': (
          LogisticRegression(penalty='l2',
                             dual=False,
                             tol=1e-4,
                             fit_intercept=True,
                             class_weight='balanced',
                             random_state=SEED,
                             solver='sag', # fast for large dataset
                             max_iter=10000,
                             n_jobs=NJOBS,
                             verbose=1),
          {'C': np.logspace(-3, 3, 11)}),
      'SVM': (
          SVC(kernel='rbf',
              gamma='scale',
              class_weight='balanced', # random folds so class frequencies are unexpected
              random_state=SEED,
              verbose=1),
          {'C': np.logspace(-3, 3, 11)}),
      'Random Forest': ( 
          RandomForestClassifier(criterion='gini',
                                 bootstrap=True,
                                 verbose=1,
                                 max_features='auto'),# will do sqrt at each split
          {
            'n_estimators': [10, 50, 100, 500, 1000],
            'max_depth': [10, 25, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]}), # powers of 2
      'Neural Network': ( 
          MLPClassifier(solver='adam',
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        max_iter=10000,
                        random_state=SEED,
                        verbose=1,
                        early_stopping=True),
          {
            'hidden_layer_sizes': [(size,) for size in [1, 5, 20, 80, 320, 1280]],
            'activation': ['relu', 'tanh'], # relu might be the best anyway
            'alpha': np.logspace(-3, 3, 11),
          }),
    }

  def feature_extraction(self, data, istest=False):
    """Does in place feature_extraction for data.
     
    If the data passed is train data, then some states need to be saved.
    Example: for tfidf, use the same vocabulary from train to test data.
    """
    return self._add_tfidf(data, istest)

  def fit(self, X, Y):
    print('Fit...')
    for model_name, (model, param_grid) in self.__models.items():
      print(f'Grid searching for {model_name}')
      if model_name == 'Neural Network':
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   n_jobs=NJOBS,
                                   verbose=1)
        grid_search.fit(X, Y)
        best_model = grid_search.best_estimator_
        print(f'Done for {model_name}')
        self.__best_models.append(best_model)
        print(best_model.classes_)
        # For debugging
        break


  def predict(self, ids, X, path):
    for best_model in self.__best_models:
      predictions = best_model.predict(X)
      AbstractModel._create_submission(ids, predictions, path)

  def _add_tfidf(self, data, istest):
    """Adds tfidf vectorization to the data."""
    print('Vectorize with TFIDF...')
    if istest:
      vectorizer = TfidfVectorizer(vocabulary=self.__vocabulary)
    else:
      vectorizer = TfidfVectorizer(max_features=1000)
    x = vectorizer.fit_transform(data['text'])
    if not istest:
      self.__vocabulary = vectorizer.get_feature_names()
    feature_names = ['TFIDF_' + name.upper()
                     for name in vectorizer.get_feature_names()]
    tfidf_features = pd.DataFrame(x.toarray(), columns=feature_names) \
      .reset_index(drop=True)
    data = pd.concat([data, tfidf_features], axis=1)
    return data
