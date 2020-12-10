# Twitter Sentiment Analysis - EPFL course challenge

## Authors (rainbow-triangle 🌈)

* Giorgio Mannarini
* Maria Pandele
* Francesco Posa

## Introduction

## Data

Label -1 - negative  
Label 1 - positive  

## Dependencies
To properly run our code you will have to install some dependencies. Our 
suggestion is to use a Python environment (we used Anaconda). 
GRU and Bert are built on TensorFlow, with Keras as a wrapper, while the 
baseline has been done in scikit-learn. In alphabetical order, you should have:

- joblib 0.17 `pip install joblib`
- nltk 3.5 `pip install nltk`
- numpy 1.18.5 `pip install numpy`
- pandas 1.1.2 `pip install pandas`
- tensorflow 2.3.1 `pip install --upgrade tensorflow`
- transformers 3.4.0  `pip install transformers`
- scikit-learn 0.23.2 `pip install -U scikit-learn`
- setuptools 50.3 `pip install setuptools`
- symspellpy 6.7 `pip install symspellpy`
- vaderSentiment 3.3.2 `pip install vaderSentiment`

## Project structure

This is scheleton we used when developing this project. We recommend this
structure since all the files' locations are based on it.

`classes`: contains all our implementation

`logs`: contains outputed logs during training

`preprocessed_data`: we are saving/loading the preprocessed data here/from here

`submissions`: contains AIcrowd submissions

`utility`: contains helpful resources for preprocessing the tweets

`weights`: contains saved weights

`Exploratory_data_analysis.ipynb`: extracts emoticons from tweets

`constants.py`: defines constants used throughout preprocessing, training and
inference

`run.py`: main script, more details on how to use it in the next section

## How to run

There are several ways to run it. You can either re-run everything from data
preprocessing to training and inference. Or you can just load our already
trained models and make predictions.  
**If, you just want to reproduce our best
submission then skip to [Best submission on AIcrowd](#best-submission-on-AIcrowd)
section.**

### Step 1. Download the raw data
Skip this section if you only want to make predictions.

Download the raw data from [https://www.aicrowd.com/challenges/epfl-ml-text-classification](https://www.aicrowd.com/challenges/epfl-ml-text-classification)
and put it in a new top level folder called `data`.
So you should have something like this:
```
├── data
│   ├── train_pos.txt
│   ├── train_neg.txt
│   ├── train_pos_full.txt
│   ├── train_neg_full.txt
│   └── test_data.txt
```

### Step 2. Download the already preprocessed tweets
Skip this section if you did [Step 1](#step-1-download-the-raw-data) and want
to do your own preprocessing.

If you want to download the preprocessed tweets then download them from 
[this Drive link](https://drive.google.com/drive/folders/16izsD7W0SG3AF094cW0JpcfnPFRF1aXY?usp=sharing)
and save them into the top level [`preprocessed_data`](https://github.com/mapaaa/ml-project2/tree/master/preprocessed_data) folder. So you should have 
something like this:
```
├── preprocessed_data
│   ├── baseline
│   │   ├── test_preprocessed.csv   
│   │   └── train_preprocessed.csv
│   ├── bert
│   │   ├── test_preprocessed.csv   
│   │   └── train_preprocessed.csv
│   ├── gru
│   │   ├── test_preprocessed.csv   
│   │   └── train_preprocessed.csv
│   └── README.md
```

### Step 3. Download the models
Skip this section if you want to re-train the models.

If you want to download the pretrained models (HIGHLY RECOMMENDED for the
deep learning models) then download them from [this Drive link](https://drive.google.com/drive/folders/1o_exDi-gA0X1kSBTl9qUPpEGWZBX-MFy?usp=sharing)
and save them into the top level [`weights`](https://github.com/mapaaa/ml-project2/tree/master/weights) folder.
So you should have something like this:
```
├── weights
│   ├── baseline
│   │   ├── model-KNN.csv   
│   │   ├── model-Logistic-Regression.csv   
│   │   ...
│   │   └── model-SVM.csv
│   ├── bert
│   │   └── model
│   │       ├── config.json
│   │       └── tf_model.h5
│   ├── gru
│   └── README.md
```

### Step 4. The actual run

## Best sumbission on AIcrowd


## Results at a glance
