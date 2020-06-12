# -*- coding: utf-8 -*-
"""
Nafis Asghari
Student Number: 20196621
MMAI
Cohort: 2020
Course: MMAI 891 - Natural Language Processing
June 15, 2020


Submission to Question 3 - Main file
"""

################################
### Import packages and Data ###
################################
from dl_utils import *

import pandas as pd
import numpy as np
import os
import argparse
from keras.utils import np_utils



in_dir = 'data'
out_dir = 'out/deeplearning'

df_train = pd.read_csv(os.path.join(in_dir, "sentiment_train.csv"))

print(df_train.info())
print(df_train.head())

df_test = pd.read_csv(os.path.join(in_dir, "sentiment_test.csv"))

print(df_test.info())
print(df_test.head())


###################################
###     DL-BASED APPROACH       ###
###################################
MAXLEN = 30

df= pd.concat([df_train , df_test] , axis = 0)

x , embedding_layer = keras_preprocess(df['Sentence'], maxlen = 30)
x_train , x_test = x[:df_train.shape[0]] , x[df_train.shape[0]:]

y_train = np_utils.to_categorical(list(df[:df_train.shape[0]].Polarity))
y_test = np_utils.to_categorical(list(df[df_train.shape[0]:].Polarity))



parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='CNN, Lstm', default='CNN')
args = parser.parse_args()

if __name__ == "__main__":              
        
    clf = dl_model(model = args.model, maxlen = MAXLEN , embedding_layer = embedding_layer, 
                   x = x_train , y = y_train , out_dir = out_dir)
    
    metrics(df_test, clf ,x_test, y_test , thrsh = 0.5, clf_name = args.model, out_dir = out_dir)
         
    
    print(x_train.shape , x_test.shape)
    assert (x_test.shape[1] == x_train.shape[1])
    assert((y_train.shape[1] == 2))
    assert((y_test.shape[1] == 2))