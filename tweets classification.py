#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras as kr


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[51]:


#read training data into DataFrame
df = pd.read_csv('processed_training_data_preprocessed.csv',encoding = "ISO-8859-1" )


# In[52]:


df.head()


# In[47]:


data = df[['clean_tweet','polarity of tweet']]
data.head()


# In[65]:


#data = df[['clean_tweet','polarity of tweet']]
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['clean_tweet'].values)
X = tokenizer.texts_to_sequences(data['clean_tweet'].values)
X = pad_sequences(X)


# In[67]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[68]:


Y = pd.get_dummies(data['polarity of tweet']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[69]:


batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)


# In[70]:


from keras.models import load_model


# In[72]:


model.save('tweets_sentiment_classification.h5')


# In[75]:


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("accuracy: %.2f" % (acc))

