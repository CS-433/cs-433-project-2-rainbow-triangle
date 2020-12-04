import pandas as pd
from classes.bert import Bert
from constants import *
from time import strftime

if __name__ == '__main__':
  # Instantiating the classifier (be sure to create the folder!)
  classifier = Bert(BERT_WEIGHTS_PATH)

  # Training the model

  train_preprocessed = pd.read_csv(
    f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TRAIN_DATA_BERT}',
    usecols=['text', 'label'])

    # Making the predictions
  test_preprocessed = pd.read_csv(
    f'{PREPROCESSED_DATA_PATH_BERT}{PREPROCESSED_TEST_DATA_BERT}',
    usecols=['ids', 'text'])
  
  train_preprocessed.dropna(inplace=True)

  X = train_preprocessed['text'].values
  Y = train_preprocessed['label'].values

  test_ids = test_preprocessed['ids'].values
  X_test = test_preprocessed['text'].values

  # Fitting the model and making the prediction
  classifier.fit_predict(X, Y, test_ids, X_test, f'{SUBMISSION_PATH_BERT}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')
  
  #Decomment to only make a prediction with a saved model. Comment the previous line in this case.
  #classifier.predict(test_ids, X_test, f'{SUBMISSION_PATH_BERT}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv, from_weights = True)