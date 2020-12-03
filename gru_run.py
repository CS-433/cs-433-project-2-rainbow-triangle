import pandas as pd
from classes.gru import Gru
from constants import *
from time import strftime

if __name__ == '__main__':

    # Instanciating the classifier with a pre-trained GLOVE embedding
    classifier = Gru(GRU_WEIGHTS_PATH, GLOVE_PATH)

    # Reading the preprocessed training and test files for this model
    train_preprocessed = pd.read_csv(
        f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TRAIN_DATA_GRU}',
        usecols=['text', 'label'])

    test_preprocessed = pd.read_csv(
        f'{PREPROCESSED_DATA_PATH_GRU}{PREPROCESSED_TEST_DATA_GRU}',
        usecols=['ids', 'text'])

    # Dropping NaN values from the train dataset, that could be present after the preprocessing
    train_preprocessed.dropna(inplace=True)

    # Creating the training matrix with the related labels
    X = train_preprocessed['text'].values
    Y = train_preprocessed['label'].values

    # Creating the test matrix with the associated ids
    X_test = test_preprocessed['text'].values
    test_ids = test_preprocessed['ids'].values

    # Updating the vocabulary of the classifier according to the training data
    classifier.update_vocabulary(X)

    # Fitting the model and making the prediction
    classifier.fit_predict(X, Y, test_ids, X_test, f'{SUBMISSION_PATH_GRU}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')



    #Decomment to only make a prediction with a saved model. Comment the previous line in this case.
    #classifier.predict(test_ids, X_test, f'{SUBMISSION_PATH_GRU}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv', from_weights = True)