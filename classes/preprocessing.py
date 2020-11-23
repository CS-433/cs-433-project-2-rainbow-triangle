import pkg_resources
import nltk
import re
import pandas as pd
import numpy as np

from collections import Counter
from utility.emo_unicode import EMOTICONS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from symspellpy import SymSpell

nltk.download('stopwords')
nltk.download('wordnet')


class Preprocessing:
  """Preprocesses the data and can even perform feature extraction.

  Attributes:
    __data: A pandas dataframe with the data (at least one column called text).
  """

  def __init__(self, list_: list, submission=False):
    if not submission:
      if len(list_) == 2:
        self.__data = pd.DataFrame(columns=['text', 'label'])

        for i, file_name in enumerate(list_):
          with open(file_name) as f:
            content = f.read().splitlines()
          df = pd.DataFrame(columns=['text', 'label'],
                            data={'text': content,
                                  'label': np.ones(len(content)) * i})
          self.__data = self.__data.append(df).reset_index(drop=True)
    else:
      if len(list_) == 1:
        with open(list_[0]) as f:
          content = f.read().splitlines()

        ids = [line.split(',')[0] for line in content]
        texts = [','.join(line.split(',')[1:]) for line in content]
        self.__data = pd.DataFrame(columns=['ids', 'text'],
                                   data={'ids': ids, 'text': texts})

  def get(self):
    return self.__data

  def logging(self):
    """Prints the first 10 rows in the dataframe stored in self.__data."""
    print('Logging:')
    print(self.__data['text'].head(10))

  def drop_duplicates(self):
    self.__data = self.__data.drop_duplicates(subset=['text'])

  def to_lower(self):
    print('Converting to lowercase...')
    self.__data['text'] = self.__data['text'].str.lower()

  def remove_punctuation(self):
    print('Removing punctuation...')
    self.__data['text'] = self.__data['text'].str.replace('[^\w\s]', '')

  def remove_elongs(self):
    print('Removing elongs...')

    self.__data['text'].apply(
      lambda text: str(re.sub(r'([a-zA-Z])\1{2,}', r'\1', text)))

  def remove_numbers(self):
    print('Removing numbers...')
    self.__data['text'] = self.__data['text'].str.replace('\d', '')

  def remove_stopwords(self):
    print('Removing stopwords...')
    stopwords_ = set(stopwords.words('english'))
    self.__data['text'] = self.__data['text'].apply(
      lambda text: ' '.join(
        [word for word in str(text).split() if word not in stopwords_]))

  def remove_frequent_words(self):
    print('Removing frequent words...')
    cnt = Counter()
    freqwords = set([w for (w, wc) in cnt.most_common(10)])
    self.__data['text'] = self.__data['text'].apply(
      lambda text: ' '.join(
        [word for word in str(text).split() if word not in freqwords]))

  def remove_rare_words(self):
    print('Removing rare words...')
    cnt = Counter()
    n_rare_words = 10
    rarewords = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])
    self.__data['text'] = self.__data['text'].apply(
      lambda text: ' '.join(
        [word for word in str(text).split() if word not in rarewords]))

  def stemming(self):
    print('Performing stemming...')
    stemmer = PorterStemmer()
    self.__data['text'] = self.__data['text'].apply(
      lambda text: ' '.join([stemmer.stem(word) for word in text.split()]))

  def lemmatize(self):
    print('Performing lemmatization...')
    lemmatizer = WordNetLemmatizer()
    self.__data['text'] = self.__data['text'].apply(
      lambda text: ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split()]))

  def emoticons_to_words(self):
    print('Converting emoticons to words...')

    def inner(text):
      for emo in EMOTICONS:
        emo_with_spaces = " ".join(
          [re.escape(ch) for ch in emo if not ch == '\\'])

        text = re.sub(
          f'( {emo}( |$))|( {emo_with_spaces}( |$))|( < \\\ 3( |$))',
          ' ' + EMOTICONS[emo].lower() + ' ',
          text)

      return text

    self.__data['text'] = self.__data['text'].apply(
      lambda text: inner(str(text)))

  def remove_tags(self):
    print('Removing tags...')

    def inner(text):
      text = re.sub('<[\w]*>', '', text).strip()
      text = re.sub('\.{3}$', '', text).strip()
      text = re.sub(' (\(|\))$', r' :\1', text).strip()

      return text

    self.__data['text'] = self.__data['text'].apply(
      lambda text: inner(str(text)))

  def remove_parenthesis(self):
    print('Removing parenthesis...')
    self.__data['text'] = self.__data['text'].apply(
      lambda text: str(re.sub('(\(|\))', '', text)))

  def slangs_to_words(self):
    print('Converting slangs to words...')
    with open('../utility/slang.txt') as f:
      chat_words_str = f.read().splitlines()
    chat_words_map_dict = {}
    chat_words_list = []
    for line in chat_words_str:
      cw = line.split('=')[0]
      cw_expanded = line.split('=')[1]
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
      return ' '.join(new_text)

    self.__data['text'] = self.__data['text'].apply(
      lambda text: chat_words_conversion(str(text)))

  def convert_hashtags(self):
    print('Converting hashtags...')
    self.__data['text'] = self.__data['text'].apply(
      lambda text: str(re.sub(
        '(#)(\w+)',
        lambda x: Preprocessing.__word_segmentation(str(x.group(2)),
                                                    correct_words=False),
        text)))

  def convert_negation(self):
    print('Converting negations...')

    # a sentence without any spaces
    self.__data['text'] = self.__data['text'].apply(
      lambda text: str(re.sub("n't", ' not', text)))

  @staticmethod
  def __word_segmentation(text, correct_words):
    max_dictionary_edit_distance = 0
    if correct_words:
      max_dictionary_edit_distance = 2

    sym_spell = SymSpell(
      max_dictionary_edit_distance=max_dictionary_edit_distance)

    dictionary_path = pkg_resources.resource_filename(
      'symspellpy',
      'frequency_dictionary_en_82_765.txt')
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    bigram_path = pkg_resources.resource_filename(
      'symspellpy',
      'frequency_bigramdictionary_en_243_342.txt')
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    result = sym_spell.word_segmentation(text)

    return result.corrected_string

  def word_segmentation(self):
    print('Splitting words...')
    self.__data['text'] = self.__data['text'].apply(
        lambda text: Preprocessing.__word_segmentation(text, correct_words=False))

  def final_paranthesis(self):
    print('Substituting final final paranthesis...')
    self.__data['text'] = self.__data['text'].str.replace('\)\)+$', ':))')
    self.__data['text'] = self.__data['text'].str.replace('\)$', ':)')
    self.__data['text'] = self.__data['text'].str.replace('\(\(+$', ':((')
    self.__data['text'] = self.__data['text'].str.replace('\($', ':(')

  def emoticons_to_sentiment(self):
    print('Substituting emoticons with sentiment...')
    pass

  def correct_spacing_indexing(self):
    print('Correcting spacing...')

    """Deletes double or more spaces and corrects indexing.

    Must be called after calling the above methods.
    Most of the above methods just delete a token. However since tokens are
    surrounded by whitespaces, they will often result in having more than one
    space between words.
    """
    self.__data['text'] = self.__data['text'].str.replace('\s{2,}', ' ')
    self.__data['text'] = self.__data['text'].apply(lambda text: text.strip())

    self.__data.reset_index(inplace=True, drop=True)

  def add_tfidf(self):
    """Adds tfidf vectorization to the data"""
    print('Vectorize with TFIDF...')
    vectorizer = TfidfVectorizer(max_features=2000)
    x = vectorizer.fit_transform(self.__data['text'])
    feature_names = ['TFIDF_' + name.upper()
                     for name in vectorizer.get_feature_names()]

    tfidf_features = pd.DataFrame(x.toarray(), columns=feature_names) \
      .reset_index(drop=True)

    self.__data = pd.concat([self.__data, tfidf_features], axis=1)
