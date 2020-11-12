import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from constants import *

from classes import *

if __name__ == '__main__':

  """
    Other methods available
    
    preprocessing.remove_numbers()
    preprocessing.remove_stopwords()
    preprocessing.remove_frequent_words()
    preprocessing.remove_rare_words()
    preprocessing.lemmatize() # or preprocessing.stemming()
      
  """

  # Performing preprocessing
  preprocessing = MyPreprocessing([FILE_NEGATIVE, FILE_POSITIVE])

  preprocessing.to_lower()
  preprocessing.remove_tags()
  preprocessing.slangs_to_words()
  preprocessing.emoticons_to_words()
  preprocessing.correct_spelling()
  preprocessing.remove_punctuation()
  preprocessing.correct_spacing()

  # Saving preprocessed data
  preprocessing.get().to_csv('out.csv')

  # Preprocessing takes much time (expecally for correcting spelling), so read from previous CSV
  # preprocessing = MyPreprocessing(['out.csv'])

  tweets = preprocessing.get()

  # Some rows are composed only by <user> and <url> tags, so after preprocessing are NaN
  tweets.dropna(inplace=True)
  tweets.info()

  X = tweets['text'].values
  Y = tweets['label'].values

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

  vectorize_layer = TextVectorization()
  vectorize_layer.adapt(X_train)

  model = tf.keras.Sequential([
    vectorize_layer,

    tf.keras.layers.Embedding(
      input_dim=len(vectorize_layer.get_vocabulary()),
      output_dim=64,
      # Use masking to handle the variable sequence lengths
      mask_zero=True),

    layers.Bidirectional(layers.GRU(64, return_sequences=True)),
    layers.Bidirectional(layers.GRU(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1)])

  model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer=optimizers.Adam(),
    metrics=['accuracy'])

  epochs = 10
  history = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_test, Y_test),
    epochs=epochs)
