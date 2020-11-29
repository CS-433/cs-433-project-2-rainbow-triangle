import nltk
import numpy as np
import pandas as pd

from classes.abstract_model import AbstractModel
from constants import *
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
    data = self._add_tfidf_lsi(data, istest)
    data = self.__add_stats(data)
    data = self.__add_vader(data)
    data = self.__add_morpho_stats(data)
    

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

  def _add_tfidf_lsi(self, data, istest):
    """Adds tfidf vectorization to the data with latent semantic indexing."""
    print('Vectorize with TFIDF-LSI...')
    if not istest:
      self.__vectorizer = TfidfVectorizer()
      x = self.__vectorizer.fit_transform(data['text'])
      self.__svd_model = TruncatedSVD(n_components=500,
                                      algorithm='randomized',
                                      n_iter=10,
                                      random_state=SEED)
      x = self.__svd_model.fit_transform(x)
      # Save the feature names
      words = self.__vectorizer.get_feature_names()
      self.__feature_names = [
          '+'.join([f'{coef:.1f}{word}' for coef, word in zip(component, words)])
          for component in self.__svd_model.components_]
    else:
      # Reuse the training representation
      x = self.__vectorizer.transform(data['text'])
      x = self.__svd_model.transfrom(x)
    tfidf_features = pd.DataFrame(x, columns=self.__feature_names) \
      .reset_index(drop=True)
    data = pd.concat([data, tfidf_features], axis=1)
    return data

  def _add_morpho_stats(self, data):
    nltk_tagged = nltk.pos_tag(data['text']) 


  def _add_vader(self, data):
    """Adds scores from Vader Sentiment Analysis."""
    analyzer = SentimentIntensityAnalyzer()
    data['VADER'] = data['raw'].str.apply(
        lambda raw_tweet: analyzer.polarity_scores(raw_tweet).['compound'])
