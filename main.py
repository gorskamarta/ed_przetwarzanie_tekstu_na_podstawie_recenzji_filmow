#!/usr/bin/env python
# coding: utf-8

# <h1>Przetwarzanie tekstu na podstawie recenzji filmów - dataset kaggle, IMDB</h1>
# <p>Marta Górska - 22818</p>
# <p>Bartosz Bachórz - 22769</p>
# <p>Marcin Rohde - 22626</p>

# In[1]:


import numpy as np
import os
import pandas as pd
from bs4 import BeautifulSoup             
import re
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


## Pobranie danych z pliku CSV oraz weryfikacja ostatnich 5 pozycji
data = pd.read_csv(r"imdb_master.csv", encoding="latin1", index_col=0)
data.tail()


# In[3]:


## Usunięcie zbędnych kolumn
data = data.drop (['type','file'],axis=1)
data.columns = ["review","label"]
data.tail()


# In[4]:


## Usunięcie danych nie poddających się klasyfikacji - brak etykiet. Zastąpienie pos/neg wartościami 1/0
data = data[data.label != 'unsup']
data['label'] = data['label'].map({'pos': 1, 'neg': 0})
data.tail()


# In[5]:


training = data
training.shape


# In[6]:


## Funkcja "czyszcząca" dane wejściowe
def review_to_words( raw_review ):
    # 1. Usunięcie potencjalnie istniejących znaczników HTML
    review_text = BeautifulSoup(raw_review,).get_text() 
    #
    # 2. Usunięcie ciągów znaków zawierających znaki inne niż same litery
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Konwersja na małe litery - unifikacja
    words = letters_only.lower().split()                                             
    #
    # 4. Konwersja z list na set - szybsze przeszukiwanie
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Usunięcie "stopwords"
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Odwrócenie procesu z punktu 4
    return( " ".join( meaningful_words ))


# In[7]:


# Przygotowanie materiału do machine learning

num_reviews = training["review"].size

# Pusta lista na wyniki
clean_train_reviews = []

# Iteracja po review od 0 do num_reviews
print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # Zwracanie wiadomości co 10000 przetworzonych indeksów
    if( (i+1)%10000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews ))                                                                 
    clean_train_reviews.append( review_to_words( training["review"][i] ))
print("Done")


# In[8]:


## Sprawdzamy dane wynikowe
clean_train_reviews


# In[9]:


print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Wektoryzacja przy pomocy CountVectorizer - bag of words
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 6000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Konwersja wyniku do tablicy (array)
train_data_features = train_data_features.toarray()
print ("Done")


# In[10]:


## Sprawdzamy co jest w danych wynikowych
train_data_features


# In[14]:


## Sprawdzenie datasetu i etykiet. Długość jest podyktowana parametrem max_features
print(len(train_data_features[0]), len(train_data_features), len(training["label"]))


# In[15]:


## tu zaczyna się Frankenstein - spawanie kodu z TF


# In[16]:


## Ładowanie bibliotek - tensorflow i inne potencjalnie potrzebne
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

print(tf.__version__)


# In[17]:


## Dane treningowe - array zawierający listy identyfikatorów wyrazów
train_data = train_data_features
## Etykiety przyjmujące wartość 1/0 - dla pozytywnych i negatywnych recenzji
train_labels = training["label"]


# In[18]:


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


# In[28]:


## Nrzygotowanie modelu do nauki
## 6000 wejść podyktowane rozmiarem max_features, 3 warstwy ukryte po 16 neuronów, 1 neuron wyjścia typu sigmoid
model = keras.Sequential([
    layers.Flatten(input_shape=[6000]),
    layers.Dense(3 , activation=tf.nn.relu),
    #layers.Dense(16 , activation=tf.nn.relu),
    #layers.Dense(16 , activation=tf.nn.relu),
    layers.Dense(1, activation=tf.nn.sigmoid)
])
## wybór optymizera
##model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

## Podsumowanie
model.summary()


# In[29]:


## Przygotowanie danych treningowych - przecięcie zestawu do 25000 i od 25000.

## Zestaw walidacyjny
x_val = train_data[:25000]
y_val = train_labels[:25000]

#x_val = train_data
#y_val = train_labels



## Zestaw treningowy
partial_x_train = train_data[25000:]
partial_y_train = train_labels[25000:]

#partial_x_train = train_data
#partial_y_train = train_labels


# In[30]:


## Uruchomienie szkolenia NN. Dane z przebiegu zapisane do 'history'
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[22]:


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')

  plt.plot(hist['epoch'], hist['loss'],
           label='Loss')
  plt.plot(hist['epoch'], hist['acc'],
           label = 'Acc')
  
  plt.legend()
  maxloss = np.amax([hist['acc'],hist['loss']])
  plt.ylim([0,(maxloss*1.1)])

plot_history(history)


# In[23]:


def plot_history2(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')

  plt.plot(hist['epoch'], hist['loss'], label='Loss')
  plt.plot(hist['epoch'], hist['acc'],  label = 'Acc')
  plt.plot(hist['epoch'], hist['val_loss'], label='V Loss')
  plt.plot(hist['epoch'], hist['val_acc'],  label = 'V Acc')

    
  plt.legend()
  maxloss = np.amax([hist['acc'],hist['loss'],hist['val_acc'],hist['val_loss']])
  plt.ylim([0,(maxloss*1.1)])

plot_history2(history)


# In[27]:


plot_history2(history)


# In[31]:


plot_history2(history)


# In[ ]:




