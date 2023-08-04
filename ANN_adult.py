# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:42:59 2019

@author: Faisal Caem
"""

# Import required libraries
import pandas as pd
#import numpy as np 
#import matplotlib.pyplot as plt
#import sklearn
from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
#%%
import os
os.chdir(r'G:\NTUST\Semester 2\Machine Learning')
df = pd.read_csv('adult.csv') 
print(df.shape)
df.describe().transpose()
#%%
data_clean = df.dropna()
for column in df.columns:
    if df[column].dtype == type(object):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
#%%
target_column = ['income'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()
#%%
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)
#%%
#from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(32,16,16,8), 
                    activation='relu',
                    solver='adam', 
                    max_iter=500, 
                    learning_rate_init=0.001,
                    learning_rate='adaptive')
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
#%%
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
#%%
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
