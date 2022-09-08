import json
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from flask import Flask

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
model_file = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

# Loading LDA model
print("Loading model from: {}".format(MODEL_PATH_LDA))
inference_lda = load(MODEL_PATH_LDA)

# loading Neural Network model
print("Loading model from: {}".format(MODEL_PATH_NN))
inference_NN = load(MODEL_PATH_NN)

# Creation of the Flask app
app = Flask(__name__)

# API 1
# Flask route so that we can serve HTTP traffic on that route
@app.route('/line/<Line>')
# Get data from json and return the requested row defined by the variable Line
def line(Line):
    with open('./test.json', 'r') as jsonfile:
       file_data = json.loads(jsonfile.read())
    # We can then find the data for the requested row and send it back as json
    return json.dumps(file_data[Line])
    

# API 2
# Flask route so that we can serve HTTP traffic on that route
@app.route('/prediction/<int:Line>',methods=['POST', 'GET'])
# Return prediction for both Neural Network and LDA inference model with the requested row as input
def prediction(Line):
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    X = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    X_test = X.iloc[Line,:].values.reshape(1, -1)
    
    clf_lda = load(MODEL_PATH_LDA)
    prediction_lda = clf_lda.predict(X_test)
    
    clf_nn = load(MODEL_PATH_NN)
    prediction_nn = clf_nn.predict(X_test)
    
    return {'prediction LDA': int(prediction_lda), 'prediction Neural Network': int(prediction_nn)}

# API 3
# Flask route so that we can serve HTTP traffic on that route
@app.route('/score',methods=['POST', 'GET'])
# Return classification score for both Neural Network and LDA inference model from the all dataset provided
def score():

    data = pd.read_json('./test.json')
    data_test = data.transpose()
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    
    clf_lda = load(MODEL_PATH_LDA)
    score_lda = clf_lda.score(X_test, y_test)
    
    clf_nn = load(MODEL_PATH_NN)
    score_nn = clf_nn.score(X_test, y_test)
    
    return {'Score LDA': score_lda, 'Score Neural Network': score_nn}
# API 4
# Flask route so that we can serve HTTP traffic on that route
@app.route('/test_acc',methods=['POST', 'GET'])
# Return classification score for both Neural Network and LDA inference model from the all dataset provided
def test_acc():

    data = pd.read_json('./test.json')
    data_test = data.transpose()
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    
   # Load model
    logit_model = load(model_path)

    # Predict
    logit_predictions = logit_model.predict(X_test)

     # Compute test accuracy
    test_logit = accuracy_score(y_test,logit_predictions)

     # Test accuracy to JSON
     # test_metadata = {
     # 'test_acc': test_logit
    #}
    
    return {'test_acc':  test_logit}

# API 4
# Flask route so that we can serve HTTP traffic on that route
@app.route('/validation_acc',methods=['POST', 'GET'])
# Return classification score for both Neural Network and LDA inference model from the all dataset provided
def validation_acc():

    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)
    
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
    
    return {'validation_acc': train_metadata}



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    
