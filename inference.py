import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
import json



def inference():

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
    model_file = 'logit_model.joblib'
    model_path = os.path.join(MODEL_DIR, model_file)
        
    # Load, read and normalize training data
    PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
    test_data_path = os.path.join(PROCESSED_DATA_DIR,testing)
    testing = "test.csv"
    data_test = pd.read_csv(testing)

    # Load data
    df = pd.read_csv(test_data_path, sep=",")

        
    y_test = df['# Letter'].values
    X_test = df.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # Run model
    print(MODEL_PATH_LDA)
    clf_lda = load(MODEL_PATH_LDA)
    print("LDA score and classification:")
    print(clf_lda.score(X_test, y_test))
    print(clf_lda.predict(X_test))
        
    # Run model
    clf_nn = load(MODEL_PATH_NN)
    print("NN score and classification:")
    print(clf_nn.score(X_test, y_test))
    print(clf_nn.predict(X_test))

    # Load model
    logit_model = load(model_path)

    # Predict
    logit_predictions = logit_model.predict(X_test)

     # Compute test accuracy
    test_logit = accuracy_score(y_test,logit_predictions)

     # Test accuracy to JSON
    test_metadata = {
    'test_acc': test_logit
    }


    # Set output path
    RESULTS_DIR = os.environ["RESULTS_DIR"]
    test_results_file = 'test_metadata.json'
    results_path = os.path.join(RESULTS_DIR, test_results_file)

    # Serialize and save metadata
    with open(results_path, 'w') as outfile:
        json.dump(test_metadata, outfile)
    
    
if __name__ == '__main__':
    inference()
