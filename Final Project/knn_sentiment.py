# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 23:33:54 2020

@author: Faisal Caem
"""
import pandas as pd
import os
os.chdir(r'F:\NTUST\Semester 2\Machine Learning\Final Project')
df = pd.read_csv("sst_train.txt", sep='\t', names=['label', 'sentence'], header=None, engine='python')
val_data = pd.read_csv("sst_dev.txt", sep='\t', names=['label', 'sentence'], header=None, engine='python')
test_data = pd.read_csv("sst_test.txt", sep='\t', names=['label', 'sentence'], header=None, engine='python')
#%%
#remove punctuation
import string
df['sentence'].str.replace('[{}]'.format(string.punctuation), '')
 #%%
#remove stopwords and lemmatization
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['sentence'] = df['sentence'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df["sentence"] = df["sentence"].str.lower().str.split()
#%%
#porter stemmer words
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def word_stemmer(text):
    stem_text=" ".join([stemmer.stem(i) for i in text])
    return stem_text
df['sentence']=df['sentence'].apply(lambda x: word_stemmer(x))
#%%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vector= vectorizer.fit_transform(df['sentence'].values[:])
vect=vector.toarray()
#%%
frame=pd.DataFrame(df,columns=['label'])
index=[str(i).zfill(2) for i in range (1, len(vect[0])+1)]
frame[index]=pd.DataFrame(vect)
#%%
target_column = ['label'] 
predictors = list(set(list(frame.columns))-set(target_column))
X = frame[predictors].values
y = frame[target_column].values
#%%
from sklearn.model_selection import train_test_split
'''from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
Y = onehotencoder.fit_transform(y).toarray()
X_train=X
y_train=Y'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
#%%
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))