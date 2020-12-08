import collections
import nltk
import numpy as np
import pandas as pd

from classes.abstract_model import AbstractModel
from constants import *
from joblib import dump, load
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from time import strftime


class Baseline(AbstractModel):
  """Does feature extraction, training and inference for various classic models.

  The non-deep-learning models used are:
    * KNN
    * Naive Bayes
    * Logistic Regression
    * SVM with linear kernel
    * Random Forest
    * NN - multi layer perceptron
  """
  def __init__(self, weights_path):
    super().__init__(weights_path) 
    self.init_models()

  def init_models(self):
    # Initialize the sklearn models here and their respective hyperparameters
    # grid for grid search with cross validation in training.
    # Some defaults are mentioned since they are important.
    self.__best_models = {}
    self.__models = {
      'KNN': (KNeighborsClassifier(weights='uniform',
                                   algorithm='auto',
                                   p=2,
                                   metric='minkowski'),
             {'n_neighbors': [3, 5, 7]}),
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
                             n_jobs=5,
                             verbose=1),
          {'C': np.logspace(-3, 3, 11)}),
      'SVM': (
          LinearSVC(class_weight='balanced', # random folds so class frequencies are unexpected
              dual=False, # n_samples > n_features
              random_state=SEED,
              max_iter=10000,
              verbose=1),
          {'C': np.logspace(-3, 3, 11)}),
      'Random Forest': ( 
          RandomForestClassifier(criterion='gini',
                                 bootstrap=True,
                                 verbose=1,
                                 n_jobs=5,
                                 max_depth=25,
                                 min_samples_split=2,
                                 min_samples_leaf=4,
                                 random_state=SEED,
                                 max_features='auto'),# will do sqrt at each split
          { 'n_estimators': [10, 50, 100, 500, 1000] }),
      'Neural Network': ( 
          MLPClassifier(solver='adam',
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        max_iter=10000,
                        random_state=SEED,
                        verbose=True,
                        activation='relu',
                        early_stopping=True),
          {
            'hidden_layer_sizes': [(size,) for size in [1, 5, 20, 80, 320, 1280]],
            'alpha': np.logspace(-3, 3, 11),
          }),
    }

  def feature_extraction(self, data, istest=False):
    """Does in place feature_extraction for data.
     
    If the data passed is train data, then some states need to be saved.
    Example: for tfidf, use the same vocabulary from train to test data.
    """
    initial_columns = data.columns
    data = self._add_tfidf_lsi(data, istest)
    self._add_vader(data)
    self._add_morpho_stats(data)
    if istest:
      labels_or_ids = data['ids'].values
    else:
      labels_or_ids = data['label'].values
    features = data.columns.difference(initial_columns, sort=False)
    data = data[features]
    data = self._standardize_data(data, istest)
    return data, labels_or_ids
    
  def fit_predict(self, X, Y, ids_test, X_test, prediction_path, model_name=None):
    print('Fit...')
    model, param_grid = self.__models[model_name]
    print(f'Grid searching for {model_name}...')
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='accuracy',
                               n_jobs=14,
                               verbose=10)
    grid_search.fit(X, Y)
    print(f'Done for {model_name}!')
    predictions = grid_search.best_estimator_.predict(X_test)
    file_path = f'{prediction_path}submission-{model_name}-{strftime("%Y-%m-%d_%H:%M:%S")}.csv'
    AbstractModel._create_submission(ids_test, predictions, file_path)
    print(f'[{model_name}]') 
    print(f'CV Accuracy: {grid_search.best_score_}') 
    print(f'Params: {grid_search.best_params_}')
    self.__best_models[model_name] = grid_search.best_estimator_
    print(f'Saving {model_name}')
    dump(grid_search.best_estimator_,
         f'{self._weights_path}model-{model_name}.joblib')

  def predict(self, ids, X, path, model_name=None):
    try:
      print('Trying to load', f'{self._weights_path}model-{model_name}.joblib')
      model = load(f'{self._weights_path}model-{model_name}.joblib')
    except:
      if model_name not in self.__best_models:
        print(f'{model_name} Not fitted! Please fit {model_name} before predicting')
        return
      model = self.__best_models[model_name]
    file_path = f'{path}submission-{model_name}-{strftime("%Y-%m-%d_%H:%M:%S")}.csv'
    print(f'Making predictions made with {model_name}...') 
    predictions = model.predict(X)
    AbstractModel._create_submission(ids, predictions, file_path)
    print(f'Predictions made with {model_name}!') 

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
      x = self.__svd_model.transform(x)
    tfidf_features = pd.DataFrame(x, columns=self.__feature_names) \
      .reset_index(drop=True)
    data = pd.concat([data, tfidf_features], axis=1)
    # Need to return the new df since pd.concat is not an inplace method
    return data

  def _add_morpho_stats(self, data):
    """Adds part-of-speech weighted count by clustering in pos tag type."""
    psw_tag_counter = data['text'].apply(lambda text: 
        collections.Counter(
          [Baseline.__psw_category(nltk_tag) 
           for _, nltk_tag in nltk.pos_tag(text.split())]))
    strengths = {
        'E': 2,
        'N': 1.5,
        'R': 1,
    }
    for tag_type in strengths:
      data[f'PSW_{tag_type}'] = strengths[tag_type] * psw_tag_counter.apply(
          lambda counter: counter[tag_type])

  def _add_vader(self, data):
    """Adds scores from Vader Sentiment Analysis."""
    analyzer = SentimentIntensityAnalyzer()
    data['VADER'] = data['raw'].apply(
        lambda raw_tweet: analyzer.polarity_scores(raw_tweet)['compound'])

  def _standardize_data(self, data, istest=False):
    """Standardize data."""
    if not istest:
      self.__standardizer = StandardScaler()
      self.__standardizer.fit(data)
    data = self.__standardizer.transform(data)
    return data
  
  @staticmethod
  def __psw_category(nltk_tag):
    """
    Cluster the part-of-speech tags into categories:
    E (emotion): verb, adjectiv, adverb
    N (normal): noun
    R (remaining): everything else
    """
    if nltk_tag.startswith('V') or nltk_tag.startswith('J') or nltk_tag.startswith('R'):
      return 'E'
    elif nltk_tag.startswith('N'):
      return 'N'
    else:
      return 'R'
