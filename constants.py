# Data files paths
TRAIN_DATA_NEGATIVE = './data/train_neg.txt'
TRAIN_DATA_POSITIVE = './data/train_pos.txt'
TRAIN_DATA_NEGATIVE_FULL = './data/train_neg_full.txt'
TRAIN_DATA_POSITIVE_FULL = './data/train_pos_full.txt'
TEST_DATA = './data/test_data.txt'

# Constants for BERT
BERT_WEIGHTS_PATH = './weights/bert/'
SUBMISSION_PATH_BERT = './submissions/bert/'
PREPROCESSED_DATA_PATH_BERT = './preprocessed_data/bert/'
PREPROCESSED_TRAIN_DATA_PREFIX_BERT = 'train_preprocessed_full_'
N_SPLITS = 1
PREPROCESSED_TRAIN_DATA_SUFFIX_BERT = '.csv'
PREPROCESSED_TEST_DATA_BERT = 'test_preprocessed.csv'

# Constants for GRU
GRU_WEIGHTS_PATH = './weights/gru/'
SUBMISSION_PATH_GRU = './submissions/gru/'
PREPROCESSED_DATA_PATH_GRU = './preprocessed_data/gru/'
PREPROCESSED_TRAIN_DATA_GRU = 'train_preprocessed.csv'
PREPROCESSED_TEST_DATA_GRU = 'test_preprocessed.csv'

# Constants for baseline (classical ML)
CLASSICAL_WEIGHTS_PATH = './weights/classical/'
SUBMISSION_PATH_CLASSICAL = './submissions/classical/'
PREPROCESSED_DATA_PATH_CLASSICAL = './preprocessed_data/classical/'
PREPROCESSED_TRAIN_DATA_CLASSICAL = 'train_preprocessed.csv'
PREPROCESSED_TEST_DATA_CLASSICAL = 'test_preprocessed.csv'
