import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing
import json
import os
from joblib import dump
import pickle

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def train():

    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
      
    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Models training
    
    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)
    
    # Serialize model
    from joblib import dump
    dump(clf_lda, MODEL_PATH_LDA)
        
    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    clf_NN.fit(X_train, y_train)
       
    # Serialize model
    from joblib import dump, load
    dump(clf_NN, MODEL_PATH_NN)

    # Model 
    logit_model = LogisticRegression(max_iter=10000)
    logit_model = logit_model.fit(X_train, y_train)

    # Cross validation
    cv = StratifiedKFold(n_splits=3) 
    val_logit = cross_val_score(logit_model, X_train, y_train, cv=cv).mean()

    # Validation accuracy to JSON
    train_metadata = {
        'validation_acc': val_logit
        }

    # Set path to output (model)
    model_name = 'logit_model.joblib'
    model_path = os.path.join(MODEL_DIR, model_name)

     # Serialize and save model
    dump(logit_model, model_path)


     # Set path to output (metadata)
    RESULTS_DIR = os.environ["RESULTS_DIR"]
    train_results_file = 'train_metadata.json'
    results_path = os.path.join(RESULTS_DIR, train_results_file)

    # Serialize and save metadata
    with open(results_path, 'w') as outfile:
        json.dump(train_metadata, outfile)
        
if __name__ == '__main__':
    train()
