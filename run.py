import argparse
from argparse import RawTextHelpFormatter
from enum import Enum
from constants import *

class Models(Enum):
  bert = "bert"
  gru = "gru"
  mlp = "mlp"
  knn = "knn"
  nbc = "nbc"
  rf = "rf"
  lr = "lr"

  def __str__(self):
    return self.value



if __name__ == '__main__':

  # NOTE: for the classical ML methods, when the "-lt" (load trained) option is not specificied, 
  # we automatically perform a cross validation to find the best hyperparameters for the model
  parser = argparse.ArgumentParser(description="This script performs a classification task to predict if \
a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.", formatter_class=RawTextHelpFormatter)

  # Required argument
  parser.add_argument("model", type = Models, choices = list(Models), 
    help="Specify the model you want to run.\n"+
      "Note: for classical ML models (every model excluded Bert and GRU), if -lt is not specified, before the training phase we perform the hyperparameters tuning\n"+
      "  bert: performs the classification with a Bert model (we suggest you to train this model on a cloud platform).\n"+
      "  gru: performs the classification with a GRU bidirectional model.\n"+
      "  mlp: performs the classification with a multi-layer perceptron neural network \n"+
      "  knn: performs the classification with a K-nearest neighbors classifier\n"+
      "  nbc: performs the classification with a Naive Bayes classifier\n"+
      "  rf: performs the classification with a Random Forest classifier\n"+
      "  lr: performs the classification with Logistic Regression")

  # Optional arguments
  parser.add_argument("-lp", help="Load already preprocessed data for a specified model")
  parser.add_argument("-lt", help="Load an already trained model")


  args = parser.parse_args()
  parser.print_help()


