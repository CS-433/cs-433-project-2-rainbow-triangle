import pandas as pd
from gru import Gru
from constants import *

if __name__ == '__main__':
    classifier = Gru(GRU_WEIGHTS_PATH)

    train_preprocessed = pd.read_csv(
        f'{PREPROCESSED_FILES_PATH}{PREPROCESSED_TRAIN_DATA_GRU}',
        usecols=['text', 'label'])

    train_preprocessed.dropna(inplace=True)

    X = train_preprocessed['text'].values
    Y = train_preprocessed['label'].values

    classifier.fit(X, Y)

    # Making the predictions
    test_preprocessed = pd.read_csv(
        f'{PREPROCESSED_FILES_PATH}{PREPROCESSED_TEST_DATA_GRU}',
        usecols=['ids', 'text'])

    ids = test_preprocessed['ids'].values
    X = test_preprocessed['text'].values

    classifier.predict(ids, X, f'{SUBMISSION_PATH}gru_submission.csv')
