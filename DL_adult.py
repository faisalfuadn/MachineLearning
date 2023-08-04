# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:47:54 2019

@author: Faisal Caem
"""

import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#%%
import os
os.chdir(r'F:\NTUST\Semester 2\Machine Learning')
df = pd.read_csv('adult.csv') 
print(df.shape)
df.describe().transpose()
#%%
data_clean = df.dropna()
for column in df.columns:
    if df[column].dtype == type(object):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
df['hours-per-week']=5374*df['hours-per-week']
#df['work_class_multiply_work_hours']=df['workclass']*df['hours-per-week']
#%%
target_column = ['income'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()
#%%
X = df[predictors].values
y = df[target_column].values
X_train=X
y_train=y
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
#print(X_train.shape); print(X_test.shape)
#%%
model = Sequential()
'''
model.add(Dense(output_dim=64, activation='relu', input_dim=14))
model.add(Dropout(0.3))
model.add(Dense(output_dim=32, activation='relu', input_dim=64))
model.add(Dropout(0.3))'''
model.add(Dense(output_dim=64, activation='relu', input_dim=14))
model.add(Dropout(0.375))
model.add(Dense(output_dim=8, activation='relu', input_dim=32))
model.add(Dropout(0.375))
model.add(Dense(output_dim=1, activation='sigmoid', input_dim=8))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
#%%
from keras.utils import plot_model
import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pyd

#Visualize Model

def visualize_model(model):
  return SVG(model_to_dot(model).create(prog='dot', format='svg'))
#create your model
#then call the function on your model
visualize_model(model)
plot_model(model, to_file='model.png')
#%%
import matplotlib.pyplot as plt

history = model.fit(X_train, y_train, validation_split=0.3, epochs=15, batch_size=128, verbose=1)
#%%
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()