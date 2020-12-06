import pandas as pd
import numpy as np
from constants import *
from classes.ensemble import Ensemble
from time import strftime


model_names = ["Gru", "Bert_no_prep", "Bert_with_prep", "KNN", "Logistic_Regression", "Naive_Bayes", "Random_Forest", "Multilayer_Perceptron"]

model_accuracies = {
  f"{SUBMISSION_PATH_GRU}submission-2020-12-03_16:55:08.csv": 0.857,
  f"{SUBMISSION_PATH_BERT}submission-2020-11-24_11:30:15.csv": 0.893,
  f"{SUBMISSION_PATH_BERT}submission-2020-12-03_20:24:31.csv": 0.888,
  f"{SUBMISSION_PATH_CLASSICAL}submission-KNN-2020-12-04_23:54:14.csv": 0.669,
  f"{SUBMISSION_PATH_CLASSICAL}submission-Logistic Regression-2020-12-03_15:48:25.csv":0.775,
  f"{SUBMISSION_PATH_CLASSICAL}submission-Naive Bayes-2020-12-03_15:42:16.csv":0.637,
  f"{SUBMISSION_PATH_CLASSICAL}submission-Random-Forest-2020-12-04_16:14:15.csv": 0.773,
  f"{SUBMISSION_PATH_CLASSICAL}submission-Neural Network-2020-12-04_00:21:48.csv": 0.790
}

ensemble_model = Ensemble(model_accuracies, model_names)

ensemble_model.predict(f'{SUBMISSION_PATH_ENSEMBLE}submission-{strftime("%Y-%m-%d_%H:%M:%S")}.csv')


