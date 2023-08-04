import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, GRU, Input, Layer, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
eng_stopwords = set(stopwords.words("english"))
import textblob as tb
import pytreebank
import sys
import os
import torch
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer
from pytorch_pretrained_bert import BertModel
# Import Data
def import_data():
    out_path = os.path.join(sys.path[0], 'sst_{}.txt')
    dataset = pytreebank.load_sst()

    for category in ['train', 'test', 'dev']:
        with open(out_path.format(category), 'w') as outfile:
            for item in dataset[category]:
                outfile.write("{}\t{}\n".format(
                    item.to_labeled_lines()[0][0] + 1,
                    item.to_labeled_lines()[0][1]
                ))


df = pd.read_csv("sst_train.txt", sep='\t', names=['label', 'sentence'], header=None, engine='python')
val_data = pd.read_csv("sst_dev.txt", sep='\t', names=['label', 'sentence'], header=None, engine='python')
test_data = pd.read_csv("sst_test.txt", sep='\t', names=['label', 'sentence'], header=None, engine='python')

Y_train = df['label']
Y_val = val_data['label']
# Y_test = test_data['label']

Y_train = Y_train - 1
Y_val = Y_val - 1
# Y_test = Y_test - 1

Y_train = to_categorical(Y_train.values)
Y_val = to_categorical(Y_val.values)
# Y_test = to_categorical(Y_test.values)

df.drop(['label'], axis=1, inplace=True)
val_data.drop(['label'], axis=1, inplace=True)
# test_data.drop(['label'], axis=1, inplace=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = []
MAX_LEN = 0  # length of each vector
for i in range(0, len(df)):
    df['sentence'][i] = "[CLS] " + df['sentence'][i] + " [SEP]"
    tokenized = tokenizer.tokenize(df['sentence'][i])
    if len(tokenized) > MAX_LEN:
        MAX_LEN = len(tokenized)
    tokenized_text.append(tokenizer.convert_tokens_to_ids(tokenized))

tokenized_val =[]
for i in range(0, len(val_data)):
    df['sentence'][i] = "[CLS] " + df['sentence'][i] + " [SEP]"
    tokenized = tokenizer.tokenize(df['sentence'][i])
    if len(tokenized) > MAX_LEN:
        MAX_LEN = len(tokenized)
    tokenized_val.append(tokenizer.convert_tokens_to_ids(tokenized))

X_train = pad_sequences(tokenized_text, maxlen=MAX_LEN)
X_val = pad_sequences(tokenized_val, maxlen=MAX_LEN)
VOCAB_SIZE = np.amax(X_train) + 1  # size of vocabulary

model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=100, input_length=MAX_LEN, mask_zero=True))
model.add(LSTM(64, activation='sigmoid', dropout=0.2, return_sequences=True))
model.add(LSTM(32, activation='sigmoid', dropout=0.2, return_sequences=False))
model.add(Dense(5, activation='sigmoid'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=128, verbose=1,
          validation_data=(X_val, Y_val))
# Text Converting
'''corpus_text = []
meta_text = []
# For words in training data
for i in range(0, len(df)):
    corpus = df['sentence'][i]
    corpus = corpus.lower()  # convert words into lower case
    corpus = corpus.split()  # split into separate instances
    pstem = PorterStemmer()  # use PorterStemmer to perform stemming
    corpus = [pstem.stem(word) for word in corpus if not word in eng_stopwords]  # words that are not a stop word
    corpus = ' '.join(corpus)
    corpus_text.append(corpus)
    # print(corpus)
    meta_text.append(tb.TextBlob(corpus).sentiment[0])


corpus_val = []
meta_val = []
# For words in validation data
for i in range(0, len(val_data)):
    corpus = val_data['sentence'][i]
    corpus = corpus.lower()  # convert words into lower case
    corpus = corpus.split()  # split into separate instances
    pstem = PorterStemmer()  # use PorterStemmer to perform stemming
    corpus = [pstem.stem(word) for word in corpus if not word in eng_stopwords]  # words that are not a stop word
    corpus = ' '.join(corpus)
    corpus_val.append(corpus)
    meta_val.append(tb.TextBlob(corpus).sentiment[0])

corpus_test = []
meta_test = []
# For words in testing data
for i in range(0, len(test_data)):
    corpus = test_data['sentence'][i]
    corpus = corpus.lower()  # convert words into lower case
    corpus = corpus.split()  # split into separate instances
    pstem = PorterStemmer()  # use PorterStemmer to perform stemming
    corpus = [pstem.stem(word) for word in corpus if not word in eng_stopwords]  # words that are not a stop word
    corpus = ' '.join(corpus)
    corpus_test.append(corpus)
    # meta_test[i] = tb.TextBlob(corpus[i]).sentiment[0]

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(corpus_text))
X = tokenizer.texts_to_sequences(corpus_text)
X_val = tokenizer.texts_to_sequences(corpus_val)
X_test = tokenizer.texts_to_sequences(corpus_test) 

X_train = pad_sequences(X, maxlen=124)
X_val = pad_sequences(X_val, maxlen=125)
X_test = pad_sequences(X_test, maxlen=125)

meta_text = np.reshape(meta_text, (-1,1))
meta_text = meta_text + 1
meta_val = np.reshape(meta_val, (-1,1))
meta_val = meta_val + 1
X_train = np.concatenate((X_train, meta_text), axis=1)
X_val = np.concatenate((X_val, meta_val), axis=1)
print(np.amin(meta_text))

model = Sequential()
model.add(Embedding(20000, 100, mask_zero=True))
model.add(LSTM(64, activation='sigmoid', dropout=0.2, return_sequences=True))
model.add(LSTM(32, activation='sigmoid', dropout=0.2, return_sequences=False))
model.add(Dense(5, activation='sigmoid'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=1024, verbose=1)
# first = Sequential()
#first.add(Embedding(20000, 100, mask_zero=True))
#first.add(LSTM(32, activation='sigmoid', dropout=0.2, return_sequences=False))
#first.summary()
#second = Sequential()
'''

'''model = Sequential()
model.add(Dense(5, activation='sigmoid'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=128, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)'''
'''Y = data['sentiment values']
# Y = np.asarray_chkfinite(Y)
Y = to_categorical(Y.values)
data.drop(['sentiment values'], axis=1, inplace=True)

# Text Converting
corpus_text = []
# For words in training and testing data
for i in range(0, len(data)):
    corpus = data['phrase'][i]
    corpus = corpus.lower()  # convert words into lower case
    corpus = corpus.split()  # split into separate instances
    pstem = PorterStemmer()  # use PorterStemmer to perform stemming
    corpus = [pstem.stem(word) for word in corpus if not word in eng_stopwords]  # words that are not a stop word
    corpus = ' '.join(corpus)
    corpus_text.append(corpus)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(list(corpus_text))
X = tokenizer.texts_to_sequences(corpus_text)
X = pad_sequences(X, maxlen=125)

print(corpus_text[50000])
print(X[50000])
print(tb.TextBlob(data['phrase'][50000]).sentiment[0])'''

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)



#GRU
'''model = Sequential()
model.add(Embedding(20000, 100, mask_zero=True))
model.add(GRU(64, activation='sigmoid', dropout=0.2, return_sequences=True))
model.add(GRU(32, activation='sigmoid', dropout=0.2, return_sequences=True))
model.add(GRU(16, activation='sigmoid', dropout=0.2, return_sequences=False))
model.add(Dense(5, activation='sigmoid'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=1024, verbose=1)'''

'''# CNN
model = Sequential()
model.add(Embedding(20000, 100, mask_zero=True))
model.add(Conv1D(128, 5, activation='sigmoid'))
model.add(GlobalMaxPooling1D())
model.add(Dense(5, activation='sigmoid'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=1024, verbose=1)'''

# LSTM
'''model = Sequential()
model.add(Embedding(20000, 100, mask_zero=True))
model.add(LSTM(64, activation='sigmoid', dropout=0.2, return_sequences=True))
model.add(LSTM(32, activation='sigmoid', dropout=0.2, return_sequences=False))
model.add(Dense(5, activation='sigmoid'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=1024, verbose=1)'''

''''# Dense Neural Network
model = Sequential()
model.add(Embedding(20000, 100, mask_zero=True, input_length=125))
model.add(Flatten())
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=1024, verbose=1)'''