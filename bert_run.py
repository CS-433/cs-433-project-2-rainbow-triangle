import pandas as pd
from classes import Bert
from constants import *

# Instantiating the classifier (be sure to create the folder!)
classifier = Bert('./weights/')

# Training the model
for i in range(N_SPLITS):
  train_preprocessed = pd.read_csv(f'{PREPROCESSED_TRAIN_DATA_PREFIX}{i}{PREPROCESSED_TRAIN_DATA_SUFFIX}',
                                   usecols=['text', 'label'])
  train_preprocessed.dropna(inplace=True)

  X = train_preprocessed['text'].values
  Y = train_preprocessed['label'].values

  classifier.fit(X, Y, batch_size=BATCH_SIZE)

# Making the predictions
test_preprocessed = pd.read_csv(PREPROCESSED_TEST_DATA, usecols=['ids', 'text'])

ids = test_preprocessed['ids'].values
X = test_preprocessed['text'].values

classifier.predict(ids, X, SUBMISSION_FILE)
