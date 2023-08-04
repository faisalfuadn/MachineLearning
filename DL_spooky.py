# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:46:51 2019

@author: Faisal Caem
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#%%
import os
os.chdir(r'F:\NTUST\Semester 2\Machine Learning')
df=pd.read_csv('spooky.csv')
data_clean = df.dropna()
#%%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
df['text']=df['text'].str.replace('[^\w\s]','')
for i in range (len(df)):
    df['text'][i]=word_tokenize(df['text'][i])
#%%
def remove_stopwords(text):
    words=[w for w in text if w not in stopwords.words('english')]
    return words
df['text']=df['text'].apply(lambda x: remove_stopwords(x))
df['text'].head(10)
#%%
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text=[lemmatizer.lemmatize(i) for i in text]
    return lem_text
df['text'].apply(lambda x:word_lemmatizer(x))
#%%
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def word_stemmer(text):
    stem_text=" ".join([stemmer.stem(i) for i in text])
    return stem_text
df['text']=df['text'].apply(lambda x: word_stemmer(x))
#%%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vector= vectorizer.fit_transform(df['text'].values[:])
vect=vector.toarray()
#%%
frame=pd.DataFrame(df,columns=['author'])
index=[str(i).zfill(2) for i in range (1, len(vect[0])+1)]
frame[index]=pd.DataFrame(vect)
#%%
target_column = ['author'] 
predictors = list(set(list(frame.columns))-set(target_column))
#frame[predictors] = frame[predictors]/frame[predictors].max()
#frame.describe().transpose()
#%%
X = frame[predictors].values
y = frame[target_column].values
#%%
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
Y = onehotencoder.fit_transform(y).toarray() 
X_train=X
y_train=Y

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
#print(X_train.shape); print(X_test.shape)
#%%
#from sklearn.neural_network import MLPClassifier
'''
mlp = MLPClassifier(hidden_layer_sizes=(64,64,32,16), 
                    activation='logistic',
                    solver='adam', 
                    max_iter=1000, 
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
print(classification_report(y_test,predict_test))'''
#%%
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
model = Sequential()

'''model.add(Dense(output_dim=64, activation='relu', input_dim=15645))
model.add(Dropout(0.375))
model.add(Dense(output_dim=32, activation='relu', input_dim=64))
model.add(Dropout(0.375))'''
model.add(Dense(output_dim=32, activation='relu', input_dim=15645))
model.add(Dropout(0.375))
model.add(Dense(output_dim=16, activation='relu', input_dim=32))
model.add(Dropout(0.375))
model.add(Dense(output_dim=3, activation='sigmoid', input_dim=16))

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

history = model.fit(X_train, y_train, validation_split=0.3, epochs=15, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
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