import pkg_resources
import nltk
import re
import pandas as pd
import numpy as np

from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from utility.emoticons import SENTIMENT_EMOTICONS
from utility.emoticons_glove import EMOTICONS_GLOVE
from symspellpy import SymSpell

nltk.download('stopwords')
nltk.download('wordnet')


class Preprocessing:
  """Preprocesses the data and can even perform feature extraction.

  Attributes:
    __data: A pandas dataframe with the data (at least one column called text).
  """

  symspell = None

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
    self.__data['text'] = self.__data['text'].apply(__lemmatize)
    lemmatizer = WordNetLemmatizer()
    self.__data['text'] = self.__data['text'].apply(
      lambda text: ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split()]))

  def emoticons_to_words(self):
    print('Converting emoticons to words...')

    def inner(text):
      for emo in EMOTICONS:
        emo_with_spaces = ' '.join(
          [re.escape(ch) for ch in emo if not ch == '\\'])
        text = re.sub(
          f'( {emo}( |$))|( {emo_with_spaces}( |$))|( < \\\ 3( |$))',
          ' ' + EMOTICONS[emo].lower() + ' ',
          text)
      return text

    self.__data['text'] = self.__data['text'].apply(
      lambda text: inner(str(text)))

  def emoticons_to_sentiment(self):
    print('Substituting emoticons with sentiment...')

    def inner(text):
      for emo in SENTIMENT_EMOTICONS:
        emo_with_spaces = ' '.join(
          [re.escape(ch) for ch in emo if not ch == '\\'])
        emo_escaped = ''.join([re.escape(ch) for ch in emo if not ch == '\\']) 
        text = re.sub(
          f'{emo_escaped}|{emo_with_spaces}',
          ' ' + SENTIMENT_EMOTICONS[emo].lower() + ' ',
          text)
      return text

    self.__data['text'] = self.__data['text'].apply(
      lambda text: inner(str(text)))

  def remove_tags(self):
    print('Removing tags...')
    self.__data['text'] = self.__data['text'].str.replace('<[\w]*>', '')
    self.__data['text'] = self.__data['text'].str.replace('\.{3}$', '')

  def remove_parenthesis(self):
    print('Removing parenthesis...')
    self.__data['text'] = self.__data['text'].apply(
      lambda text: str(re.sub('(\(|\))', '', text)))

  def slangs_to_words(self):
    print('Converting slangs to words...')
    with open('./utility/slang.txt') as f:
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
    
  def correct_spelling(self):
    print('Correcting spelling...')
    self.__data['text'] = self.__data['text'].apply(
      lambda text: Preprocessing.__correct_spelling(text))

  def convert_hashtags(self):
    print('Converting hashtags...')
    self.__data['text'] = self.__data['text'].str.replace('(#)(\w+)',
        lambda text: Preprocessing.__word_segmentation(str(text.group(2))))

  def convert_negation(self):
    print('Converting negations...')
    # a sentence without any spaces
    self.__data['text'] = self.__data['text'].apply(
      lambda text: str(re.sub("n't", ' not', text)))
  
  def final_paranthesis(self):
    """Separates :) meaning smile from :)) meaning laugh.
    
    Distiction might not be good as some people do not thing there is a
    difference so use with caution.
    """
    print('Substituting final paranthesis...')
    self.__data['text'] = self.__data['text'].str.replace('\)\)+$', ':))')
    self.__data['text'] = self.__data['text'].str.replace('\)$', ':)')
    self.__data['text'] = self.__data['text'].str.replace('\(\(+$', ':((')
    self.__data['text'] = self.__data['text'].str.replace('\($', ':(')

  def correct_spacing_indexing(self):
    """Deletes double or more spaces and corrects indexing.

    Must be called after calling the above methods.
    Most of the above methods just delete a token. However since tokens are
    surrounded by whitespaces, they will often result in having more than one
    space between words.
    """
    print('Correcting spacing...')
    self.__data['text'] = self.__data['text'].str.replace('\s{2,}', ' ')
    self.__data['text'] = self.__data['text'].apply(lambda text: text.strip())
    self.__data.reset_index(inplace=True, drop=True)

  def empty_tweets(self):
    print('Marking empty tweets...')
    self.__data['text'] = self.__data['text'].str.replace('^\s*$', '<EMPTY>')

  def save_raw(self):
    """Must be called before anything else!"""
    print('Saving raw tweet...')
    self.__data['raw'] = self.__data['text']

  @staticmethod
  def __get_symspell():
    if Preprocessing.symspell is None:
      Preprocessing.symspell = SymSpell()
      dictionary_path = pkg_resources.resource_filename(
        'symspellpy',
        'frequency_dictionary_en_82_765.txt')
      Preprocessing.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)
      bigram_path = pkg_resources.resource_filename(
        'symspellpy',
        'frequency_bigramdictionary_en_243_342.txt')
      Preprocessing.symspell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    return Preprocessing.symspell

  @staticmethod
  def __word_segmentation(text):
    result = Preprocessing.__get_symspell().word_segmentation(text, max_edit_distance=0)
    return result.segmented_string
 
  @staticmethod
  def __correct_spelling(text):
    result = Preprocessing.__get_symspell().lookup_compound(text, max_edit_distance=2)
    return result[0].term

  @staticmethod
  def __get_wordnet_tag(wordnet_tag):
    if wordnet_tag.startswith('V'):
      return wordnet_tag.VERB
    elif wordnet_tag.startswith('N'):
      return wordnet.NOUN
    elif wordnet_tag.startswith('J'):
      return wordnet.ADJ
    elif wordnet_tag.startswith('R'):
      return wordnet.ADV
    else:
      return None

  @staticmethod
  def __lematize(text):
    nltk_tagged = nltk.pos_tag(text.split())
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, __get_wordnet_tag(nltk_tag)) 
            for w, nltk_tag in nltk_tagged]

  # GRU STUFF

  def emoticons_to_tags(self):
    """
    Convert emoticons (with or without spaces) into tags according to the pretrained stanford glove model
    e.g.: :) ---> <smile> and so on
    """
    print('Converting emoticons to tags...')
    union_re = {}
    for tag, emo_list in EMOTICONS_GLOVE.items():
      re_emo_with_spaces = '|'.join(re.escape(' '.join(emo)) for emo in emo_list)
      re_emo = '|'.join(re.escape(emo) for emo in emo_list)
      union_re[tag] = f'{re_emo_with_spaces}|{re_emo}'

    def inner(text, union_re):
      for tag, union_re in union_re.items():
        text = re.sub(union_re, ' ' + tag + ' ', text)
      return text

    self.__data['text'] = self.__data['text'].apply(
      lambda text: inner(str(text), union_re))
  
  def hashtags_to_tags(self):
    """
    Convert hashtags.
    e.g.: #hello ---> <hashtag> hello
    """
    print('Converting hashtags to tags...')
    self.__data['text'] = self.__data['text'].str.replace(r'#(\S+)', r'<hashtag> \1')
  
  def numbers_to_tags(self):
    """
    Convert numbers into tags.
    e.g.: 34 ---> <number>
    """
    print('Converting numbers to tags...')
    self.__data['text'] = self.__data['text'].str.replace(r'[-+]?[.\d]*[\d]+[:,.\d]*', r'<number>')
  
  def repeat_to_tags(self):
    """
    Convert repetitions of '!' or '?' or '.' into tags.
    e.g.: ... ---> . <repeat>
    """
    print('Converting repetitions of symbols to tags...')
    self.__data['text'] = self.__data['text'].str.replace(r'([!?.]){2,}', r'\1 <repeat>')
  
  def elongs_to_tags(self):
    """
    Convert elongs into tags.
    e.g.: hellooooo ---> hello <elong>
    """
    print('Converting elongated words to tags...')
    self.__data['text'] = self.__data['text'].str.replace(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <elong>')

  def remove_endings(self):
    """
    Remove ... <url> which represents the ending of tweet
    """
    print('Removing tweet ending when the tweet is cropped...')
    self.__data['text'] = self.__data['text'].str.replace(r'\.{3} <url>$', '')
