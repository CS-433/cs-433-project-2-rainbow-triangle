from abstract_model import AbstractModel

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers


class Gru(AbstractModel):

  def __init__(self, weights_path, max_tweet_length=120, embedding_dim=16):
    super().__init__(weights_path)

    self.__tokenizer = Tokenizer()
    self.__model = tf.keras.Sequential()
    self.__max_tweet_length = max_tweet_length
    self.__embedding_dim = embedding_dim

    # Size of the vocabulary, it will be updated according to the input data
    self.__vocab_size = 0

  def __update_vocabulary(self, X):
    print('Updating vocabulary...')

    # Updates the default internal vocabulary according to the words in X
    self.__tokenizer.fit_on_texts(X)

    # Updating the vocabulary length
    self.__vocab_size = len(self.__tokenizer.word_index) + 1

  def __convert_data(self, X):
    print('Converting data...')

    # Creating the numerical tokens and padding each tweet to max_tweet_length 
    X_tokens = self.__tokenizer.texts_to_sequences(X)

    # Note: padding = 'post' means that the pad is after each sequence
    # (each tweet) and not before
    X_pad = pad_sequences(
      X_tokens,
      maxlen=self.__max_tweet_length,
      padding='post')

    return X_pad

  def __build_model(self):
    print('Building model...')

    # Note: mask_zero must be true because 0 is a special character
    # used as padding
    self.__model.add(layers.Embedding(
      input_dim=self.__vocab_size,
      output_dim=self.__embedding_dim,
      input_length=self.__max_tweet_length,
      mask_zero=True))

    # Note: since GRU is a RNN, we need to define two types of dropouts: the
    # first one is used for the first operation on the inputs (when data
    # "enters" in GRU) the second one is used for the recurrences Units:
    # dimensionality of the output space
    self.__model.add(layers.GRU(units=16, dropout=0.2, recurrent_dropout=0.2))
    self.__model.add(layers.Dense(1, activation='sigmoid'))

    self.__model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

    print(self.__model.summary())

  def fit(self, X, Y, batch_size=128, epochs=1):
    # Updating vocabulary
    self.__update_vocabulary(X)

    # Splitting train and test data
    X_train, X_test, Y_train, Y_test = AbstractModel._split_data(X, Y)

    # Converting train and test data to sequences
    X_train_pad = self.__convert_data(X_train)
    X_test_pad = self.__convert_data(X_test)

    print(X_train_pad[0])

    self.__build_model()

    print('Training the model...')
    self.__model.fit(X_train_pad, Y_train, batch_size, epochs,
                     validation_data=(X_test_pad, Y_test))

    print('Saving the model...')

    self.__model.save_weights(f'{self._weights_path}model')

  def predict(self, ids, X, path):
    """
    Performs the predictions.

    :param ids: ids of new data
    :param X: new data to predict
    :param path: specifies where to store the submission file
    """

    # Loading weights
    self.__model.load_weights(f'{self._weights_path}model')

    # Converting input data
    X_pad = self.__convert_data(X)
    predictions = self.__model.predict(X_pad)
    print(predictions)

    AbstractModel._create_submission(ids, X, path)
