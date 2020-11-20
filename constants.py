#  ???
BUFFER_SIZE = 200000
BATCH_SIZE = 64
VALIDATION_SIZE = 20000

# Data files paths
TRAIN_DATA_NEGATIVE = './data/train_neg.txt'
TRAIN_DATA_POSITIVE = './data/train_pos.txt'
TRAIN_DATA_NEGATIVE_FULL = './data/train_neg_full.txt'
TRAIN_DATA_POSITIVE_FULL = './data/train_pos_full.txt'
TEST_DATA = './data/test_data.txt'

# Data preprocessing and training data split
PREPROCESSED_TRAIN_DATA_PREFIX = 'train_preprocessed_full_'
N_SPLITS = 5
PREPROCESSED_TRAIN_DATA_SUFFIX = '.csv'
PREPROCESSED_TEST_DATA = 'test_preprocessed.csv'
PREPROCESSED_TRAIN_DATA = 'train_preprocessed.csv'

SUBMISSION_FILE = 'submission.csv'
