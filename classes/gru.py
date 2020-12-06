import tensorflow as tf
import numpy as np
from classes.abstract_model import AbstractModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from tensorflow.keras import layers



class Gru(AbstractModel):

  def __init__(self, weights_path, glove_path, max_tweet_length=120, embedding_dim=100):
    super().__init__(weights_path)

    self.__tokenizer = Tokenizer(oov_token = '<unk>')
    self.__model = tf.keras.Sequential()
    self.__max_tweet_length = max_tweet_length
    self.__embedding_dim = embedding_dim
    self.__glove_path = glove_path

    # Size of the vocabulary, it will be updated according to the input data
    self.__vocab_size = 0


  def update_vocabulary(self, X):
    """
    Method used to update (create) the vocabulary of the tokenizer.
    
    :param X: A matrix. Each row is a document, in our case a tweet.
    :type X: list 
    """

    print('Updating vocabulary...')

    # Updates the default internal vocabulary according to the words in X
    self.__tokenizer.fit_on_texts(X)

    # Updating the vocabulary length. 
    # NOTE: the +2 is due to some special reserved tokens that are in the vocabulary
    # but not in the tweets
    self.__vocab_size = len(self.__tokenizer.word_index) + 2


  def __convert_data(self, X):
    """
    Converts the tweets in numerical tokens.
      Each word in the tweet is substituted with its index in the vocabulary,
      in a bag of words fashion. Each tweet is padded to 120 words at maximum,
      with 0 as special padding character.

    param X: A matrix. Each row is a document, in our case a tweet.
    :type X: list 
    
    :return: Numpy array with shape (len(X), max_tweet_length)
    :rtype: numpy.ndarray
    """  

    print('Converting data...')

    # Creating the numerical tokens and padding each tweet to max_tweet_length 
    X_tokens = self.__tokenizer.texts_to_sequences(X)

    # NOTE: padding = 'post' means that the pad is after each sequence
    # (each tweet) and not before
    X_pad = pad_sequences(
      X_tokens,
      maxlen=self.__max_tweet_length,
      padding='post')

    return X_pad


  def __generate_embedding_matrix(self):
    """
    Generates the word embedding matrix according to the words in the vocabulary. 
      Each word is represented by a vector with length equal to embedding_dim. 
      The embedding is done according to a model pretrained on twitter data 
      (https://nlp.stanford.edu/projects/glove/). Only the words in the vocabulary that
      are found in the pretrained model are taken into account.

    :return: The embedding matrix. Each row corresponds to a word in the vocabulary.
      The index of the row is the index of the word in the voc.
    :rtype: numpy.ndarray
    """

    print('Generating embedding matrix...')

    # Getting the vocabulary from the tokenizer
    word_index = self.__tokenizer.word_index

    # Creating a dictionary the embedding file. Keys = words in the embedding file,
    # Values = their respective vector
    embeddings_index = {}

    with open(self.__glove_path) as f:
      for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

    # Printing the number of words found in the file
    print("Found %s word vectors." % len(embeddings_index))

    # Generating the embedding matrix
    embedding_matrix = np.zeros((self.__vocab_size, self.__embedding_dim))

    # These two variables will hold the number of words in the vocabulary
    # That are found in the file, and the number of the ones that are not.
    hits = 0
    misses = 0

    for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)

      # Words not found in embedding index will be represented as a zero-vector.
      # This includes the representation for "padding" and "OOV"
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
          hits += 1
      else:
          misses += 1

    # Printing the number of found / not found words
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix


  def __build_model(self, embedding_matrix):
    """
    Method used to build and compile the GRU (Bidirectional) model.

    :param embedding_matrix: The embedding matrix used for the Embedding layer of the model.
      The embedding happens according to the matrix. The matrix is built in the previous method.
    :type: numpy.ndarray:
    """

    print('Building model...')

    # Creating the model with all its layers.
    # NOTE: mask_zero must be true because 0 is a special character
    # used as padding, as mentioned before. 
    # The Embedding layer is not trainable since we loaded the vectors from a pre-trained file, 
    # as mentioned before
    self.__model.add(layers.Embedding(
      input_dim=self.__vocab_size,
      output_dim=self.__embedding_dim,
      embeddings_initializer=Constant(embedding_matrix),
      input_length=self.__max_tweet_length,
      mask_zero=True,
      trainable = False))

    # NOTE: since we are using GRU as a RNN, we need to define two types of dropouts: the
    # first one is used for the first operation on the inputs (when data
    # "enters" in GRU) the second one is used for the recurrences Units
    self.__model.add(layers.Bidirectional(layers.GRU(units=100, dropout=0.2, recurrent_dropout=0, activation = 'tanh', \
      recurrent_activation='sigmoid', unroll=False, use_bias = True, reset_after = True)))
    self.__model.add(tf.keras.layers.Dense(100, activation='relu')),
    self.__model.add(layers.Dense(1, activation='sigmoid'))

    # Compiling the model. The optimizer is Adam with standard lr (0.001)
    self.__model.compile(
      loss='binary_crossentropy',
      optimizer= tf.keras.optimizers.Adam(),
      metrics=['accuracy'])

    # Printing model's summary
    print(self.__model.summary())


  def fit_predict(self, X, Y, ids_test, X_test, prediction_path, batch_size=128, epochs=10):
    """
    Fits (train) the model, and makes a prediction on the test data. TODO TODO TODO
    """
    
    # Splitting train and validation data
    X_train, X_val, Y_train, Y_val = AbstractModel._split_data(X, Y)

    # Converting train and validation data to sequences
    X_train_pad = self.__convert_data(X_train)
    X_val_pad = self.__convert_data(X_val)

    print(list(self.__tokenizer.word_index.keys())[:5])
    # Generating the embedding matrix from the training data
    embedding_matrix = self.__generate_embedding_matrix()

    self.__build_model(embedding_matrix)

    print('Training the model...')
    self.__model.fit(X_train_pad, Y_train, batch_size, epochs,
                     validation_data=(X_val_pad, Y_val))

    print('Saving the model...')

    self.__model.save(f'{self._weights_path}model')

    print('Making the prediction...')

    self.predict(ids_test, X_test, prediction_path)


  def predict(self, ids, X, path, from_weights = False):
    """
    Performs the predictions. Usually called within the fit_predict method.

    :param ids: ids of new data
    :param X: new data to predict
    :param path: specifies where to store the submission file
    """
    if from_weights:
      # Loading weights
      self.__model = tf.keras.models.load_model(f'{self._weights_path}model')
    
    # Converting input data
    X_pad = self.__convert_data(X)
    predictions = self.__model.predict(X_pad).squeeze()
    preds = np.where(predictions >= 0.5, 1, -1)
    print(preds)

    AbstractModel._create_submission(ids, preds, path)
