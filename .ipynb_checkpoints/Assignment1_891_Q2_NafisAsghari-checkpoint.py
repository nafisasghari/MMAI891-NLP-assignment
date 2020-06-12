# -*- coding: utf-8 -*-
"""
Nafis Asghari
Student Number: 20196621
MMAI
Cohort: 2020
Course: MMAI 891 - Natural Language Processing
June 15, 2020


Submission to Question 2 - Main File
"""

################################
### Import packages and Data ###
################################
from nlp_preprocessing import df_preprocess
from ml_utils import *

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
import os
import argparse

in_dir = 'data'
out_dir = 'out/ml'

df_train = pd.read_csv(os.path.join(in_dir, "sentiment_train.csv"))

print(df_train.info())
print(df_train.head())

df_test = pd.read_csv(os.path.join(in_dir, "sentiment_test.csv"))

print(df_test.info())
print(df_test.head())


###################################
###     ML-BASED APPROACH       ###
###################################

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', help='tfidf, glove', default='glove')
args = parser.parse_args()

if __name__ == "__main__":

    if args.method == 'glove':    
        
        nlp = spacy.load('en_core_web_md')

        x_train , y_train = df_preprocess(df_train, text_col= 'Sentence', nlp_method = 'pre_trained_glove',
                                          target = 'Polarity', feat_eng = True, nlp = nlp)

        x_test , y_test = df_preprocess(df_test, text_col= 'Sentence', nlp_method = 'pre_trained_glove',
                                        target = 'Polarity', feat_eng = True, nlp = nlp)
    
    
    
    if args.method == 'tfidf':
        
        x_train , y_train = df_preprocess(df_train, text_col= 'Sentence', nlp_method = 'tfidf',
                                          target = 'Polarity', feat_eng = True, train = True , no_features = 2000)

        x_test , y_test = df_preprocess(df_test, text_col= 'Sentence', nlp_method = 'tfidf',
                                        target = 'Polarity', train = False , no_features = 2000)
    
    
    print(x_train.shape , x_test.shape)
    assert (x_test.shape[1] == x_train.shape[1])
    assert((x_test.columns == x_train.columns).all())

    
    #Random Forest
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    param_grid_rf = [{'n_estimators': [100, 200, 300], 'max_depth': [5, 10], 'min_samples_split' : [ 10, 20, 30]}]

    rf_model = train_model_func(x=x_train, y=np.ravel(y_train),
                                    estimator=rf_clf, param_grid=param_grid_rf, cv= 5, scoring="f1")


    metrics(df_test, rf_model, x_test, y_test, thrsh= 0.41, out_dir = out_dir) #thrsh= 0.48 for tfidf
        
