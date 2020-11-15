import pkg_resources
import nltk
import string
import re
import pandas as pd
import numpy as np
from emo_unicode import EMOTICONS
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from symspellpy import SymSpell
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')


class Bert:

  def __init__(self, weights_path):
    """
    Instantiates the model for Bert classification.

    :param weights_path: specifies where load/store weights of the model
    """
    self.__model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    self.__tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    self.__weights_path = weights_path
    self.__n = 0

  def fit(self, X, Y, batch_size):
    """
    Fits the model.

    :param X: data
    :param Y: labels
    :param batch_size: specifies how many datapoints to use for each step
    :param load_weights: specifies whether to load the weights
    """
    # Converting the tweets to have a good input for BERT
    train_input_examples, validation_input_examples = self.__convert_data_to_examples(X=X, Y=Y, split_size=0.2)
    train_data = self.__convert_examples_to_tf_dataset(list(train_input_examples))
    train_data = train_data.shuffle(100).batch(batch_size).repeat(2)

    validation_data = self.__convert_examples_to_tf_dataset(list(validation_input_examples))
    validation_data = validation_data.batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    self.__model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    if self.__n > 0:
      self.__model.load_weights(f'{self.__weights_path}model_{self.__n - 1}')

    self.__model.fit(train_data, epochs=1, validation_data=validation_data)

    self.__model.save_weights(f'{self.__weights_path}model_{self.__n}')
    self.__n += 1

  def predict(self, ids, X, path):
    """
    Performs the predictions.

    :param ids: ids of new data
    :param X: new data to predict
    :param path: specifies where to store the submission file
    """

    if self.__n > 0:
      self.__model.load_weights(f'{self.__weights_path}model_{self.__n - 1}')

    predictions = []

    for i, tweet in enumerate(X):
      feature = self.__tokenizer.encode_plus(text=tweet, return_tensors="tf")
      output = self.__model(feature)[0].numpy().squeeze().argmax()
      predictions.append(output)

      if i % 100 == 0:
        print(f"Step: {i}")

    submission = pd.DataFrame(columns=['Id', 'Prediction'], data={'Id': ids, 'Prediction': predictions})
    submission['Prediction'].replace(0, -1, inplace=True)

    submission.to_csv(path, index=False)
    submission.head(10)

  def __convert_examples_to_tf_dataset(self, data, max_length=128):
    """
    Performs the tokenization where each word of each document has a max_length.

    :param data: input data
    :param max_length: length of the tokenization
    :return: precessed data
    """
    features = []  # -> will hold InputFeatures to be converted later

    for e in data:
      # Documentation is really strong for this method, so please take a look at it
      input_dict = self.__tokenizer.encode_plus(
        e.text_a,
        add_special_tokens=True,
        max_length=max_length,  # truncates if len(s) > max_length
        return_token_type_ids=True,
        return_attention_mask=True,
        padding='max_length',  # pads to the right by default
        truncation=True
      )

      input_ids, token_type_ids, attention_mask = (
        input_dict["input_ids"], input_dict["token_type_ids"], input_dict['attention_mask'])

      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
        )
      )

    def gen():
      for f in features:
        yield (
          {
            "input_ids": f.input_ids,
            "attention_mask": f.attention_mask,
            "token_type_ids": f.token_type_ids,
          },
          f.label,
        )

    return tf.data.Dataset.from_generator(
      gen,
      ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
      (
        {
          "input_ids": tf.TensorShape([None]),
          "attention_mask": tf.TensorShape([None]),
          "token_type_ids": tf.TensorShape([None]),
        },
        tf.TensorShape([]),
      ),
    )

  @staticmethod
  def __convert_data_to_examples(X, Y, split_size=0.2):
    """
    Function to transform the data in a format suitable for BERT.

    :param X: input data
    :param Y: input labels
    :param split_size: specifies the ratio to split data in train/test
    :return: transformed data
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    train_input_examples = []

    for text, label in zip(X_train, Y_train):
      train_input_examples.append(InputExample(guid=None, text_a=text, text_b=None, label=label))

    validation_input_examples = []

    for text, label in zip(X_test, Y_test):
      validation_input_examples.append(InputExample(guid=None, text_a=text, text_b=None, label=label))

    return train_input_examples, validation_input_examples


class MyPreprocessing:
  def __init__(self, list_: list, submission=False):
    if not submission:
      if len(list_) == 2:
        self.__data = pd.DataFrame(columns=['text', 'label'])

        for i, file_name in enumerate(list_):
          with open(file_name) as f:
            content = f.read().splitlines()
          df = pd.DataFrame(columns=['text', 'label'], data={'text': content, 'label': np.ones(len(content)) * i})
          self.__data = self.__data.append(df).reset_index(drop=True)

    else:
      if len(list_) == 1:
        with open(list_[0]) as f:
          content = f.read().splitlines()

        ids = [line.split(',')[0] for line in content]
        texts = [",".join(line.split(',')[1:]) for line in content]

        self.__data = pd.DataFrame(columns=['ids', 'text'], data={'ids': ids, 'text': texts})

  def get(self):
    return self.__data

  def drop_duplicates(self):
    self.__data = self.__data.drop_duplicates(subset=['text'])

  def to_lower(self):
    print("Converting to lowercase...")
    self.__data['text'] = self.__data['text'].str.lower()

  def remove_punctuation(self):
    print("Removing punctuation...")
    self.__data['text'] = self.__data['text'].str.replace('[^\w\s]', '')

  def remove_numbers(self):
    print("Removing numbers...")
    self.__data['text'] = self.__data['text'].str.replace('\d', '')

  def remove_stopwords(self):
    print("Removing stopwords...")
    stopwords_ = set(stopwords.words('english'))
    self.__data['text'] = self.__data['text'].apply(
      lambda text: " ".join([word for word in str(text).split() if word not in stopwords_]))

  def remove_frequent_words(self):
    print("Removing frequent words...")
    cnt = Counter()
    freqwords = set([w for (w, wc) in cnt.most_common(10)])
    self.__data['text'] = self.__data['text'].apply(
      lambda text: " ".join([word for word in str(text).split() if word not in freqwords]))

  def remove_rare_words(self):
    print("Removing rare words...")
    cnt = Counter()
    n_rare_words = 10
    rarewords = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])
    self.__data['text'] = self.__data['text'].apply(
      lambda text: " ".join([word for word in str(text).split() if word not in rarewords]))

  def stemming(self):
    print("Performing stemming...")
    stemmer = PorterStemmer()
    self.__data['text'] = self.__data['text'].apply(
      lambda text: " ".join([stemmer.stem(word) for word in text.split()]))

  def lemmatize(self):
    print("Performing lemmatization...")
    lemmatizer = WordNetLemmatizer()
    self.__data['text'] = self.__data['text'].apply(
      lambda text: " ".join([lemmatizer.lemmatize(word) for word in text.split()]))

  def emoticons_to_words(self):
    print("Converting emoticons to words...")

    def convert_emoticons(text):
      for emot in EMOTICONS:
        text = re.sub(u'(' + emot + ')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
      return text

    self.__data['text'] = self.__data['text'].apply(lambda text: convert_emoticons(str(text)))

  def remove_tags(self):
    print("Removing tags...")
    self.__data['text'] = self.__data['text'].str.replace("<[\w]*>", "")

  def slangs_to_words(self):
    print("Converting slangs to words...")
    with open('slang.txt') as f:
      chat_words_str = f.read().splitlines()

    chat_words_map_dict = {}
    chat_words_list = []

    for line in chat_words_str:
      cw = line.split("=")[0]
      cw_expanded = line.split("=")[1]
      chat_words_list.append(cw)
      chat_words_map_dict[cw] = cw_expanded

    chat_words_list = set(chat_words_list)

    def chat_words_conversion(text):
      new_text = []
      for w in text.split():
        if w.upper() in chat_words_list:
          new_text.append(chat_words_map_dict[w.upper()])
        else:
          new_text.append(w)
      return " ".join(new_text)

    self.__data['text'] = self.__data['text'].apply(lambda text: chat_words_conversion(str(text)))

  def correct_spelling(self):
    print("Correcting spelling...")
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    self.__data['text'] = self.__data['text'].apply(
      lambda text: sym_spell.lookup_compound(text, max_edit_distance=2)[0].term)

  def correct_spacing(self):
    '''Deletes double or more spaces.

    Must be called after calling the above methods.
    Most of the above methods just delete a token. However since tokens are
    surrounded by whitespaces, they will often result in having more than one
    space between words.
    '''
    self.__data['text'] = self.__data['text'].str.replace('\s{2,}', ' ')

  def logging(self):
    '''Prints the first 10 rows in the dataframe stored in self.__data.'''
    print('Logging:')
    print(self.__data['text'].head(10))
