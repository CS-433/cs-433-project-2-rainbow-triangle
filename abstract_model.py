import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class AbstractModel(ABC):

  def __init__(self, weights_path):
    self._weights_path = weights_path

  @abstractmethod
  def fit(self, X, Y, batch_size, epochs):
    pass

  @abstractmethod
  def predict(self, ids, X, path):
    pass

  @staticmethod
  def _create_submission(ids, predictions, path):
    # Generating the submission file
    submission = pd.DataFrame(columns=['Id', 'Prediction'],
                              data={'Id': ids, 'Prediction': predictions})

    submission['Prediction'].replace(0, -1, inplace=True)

    submission.to_csv(path, index=False)
    submission.head(10)

  @staticmethod
  def _split_data(X, Y, split_size=0.2):
    print('Splitting data in train and test set...')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=split_size)

    return X_train, X_test, Y_train, Y_test
