import pandas as pd
from classes.gru import Gru
from constants import *
from time import strftime

if __name__ == '__main__':
    classifier = Gru(GRU_WEIGHTS_PATH, GLOVE_PATH)

    train_preprocessed = pd.read_csv(
        f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TRAIN_DATA_GRU}',
        usecols=['text', 'label'])

    train_preprocessed.dropna(inplace=True)

    X = train_preprocessed['text'].values
    Y = train_preprocessed['label'].values

    classifier.fit(X, Y)

    # Making the predictions
    test_preprocessed = pd.read_csv(
        f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TEST_DATA_GRU}',
        usecols=['ids', 'text'])

    ids = test_preprocessed['ids'].values
    X = test_preprocessed['text'].values

    classifier.predict(ids, X, f'{SUBMISSION_PATH_GRU}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')
