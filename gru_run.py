import pandas as pd
from classes.gru import Gru
from constants import *
from time import strftime

if __name__ == '__main__':
    classifier = Gru(GRU_WEIGHTS_PATH, GLOVE_PATH)

    train_preprocessed = pd.read_csv(
        f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TRAIN_DATA_GRU}',
        usecols=['text', 'label'])

    test_preprocessed = pd.read_csv(
        f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TEST_DATA_GRU}',
        usecols=['ids', 'text'])

    train_preprocessed.dropna(inplace=True)

    X = train_preprocessed['text'].values
    Y = train_preprocessed['label'].values

    test_ids = test_preprocessed['ids'].values
    X_test = test_preprocessed['text'].values

    # Fitting the model and making the prediction
    classifier.fit_predict(X, Y, test_ids, X_test, f'{SUBMISSION_PATH_GRU}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')

    
