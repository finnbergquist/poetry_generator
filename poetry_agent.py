from typing import Sequence
import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from tensorflow import keras
from sklearn.model_selection import train_test_split


class PoetryAgent:
    '''
    The poetry agent holds all the important ml models for the poetry generation.
    It implements a tensorflow neural network, using recurrent reinforcement
    learning. 
    '''
    def __init__(self):
        with open('data.txt') as f:
            data = f.readlines()

        self.token =Tokenizer()
        self.token.fit_on_texts(data)
        self.sequential_encoded_text = self.token.texts_to_sequences(data)
        self.X, self.y, self.vocab_size, self.seq_length = self.prepare_data()
        self.model = Sequential()

    def save(self, path):
        '''
        Saves the model to specified path location in directory
        '''
        self.model.save(path)
        return
    
    def load(self, path):
        '''
        Loads model from a specified path, that sas saved using the save() method
        '''
        self.model = keras.models.load_model(path)
        return

    def prepare_data(self):
        datalist = []
        for d in self.sequential_encoded_text:
            if len(d)>1:
                for i in range(2, len(d)):
                    datalist.append(d[:i])
                    #print(d[:i])

        vocab_size = len(self.token.word_counts) + 1
        sequences = pad_sequences(datalist, maxlen=20, padding='pre')
        X = sequences[:, :-1]
        y = sequences[:, -1]
        y = to_categorical(y, num_classes=vocab_size)
        seq_length = X.shape[1]

        return X, y, vocab_size, seq_length

    def train(self):
        self.model.add(Embedding(self.vocab_size, 50, input_length=self.seq_length))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.vocab_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #Epoch -> number of times to go through the training data
        self.model.fit(self.X, self.y, batch_size=32, epochs=20)


    def generate_poetry(self, seed_text, n_lines):
        for i in range(n_lines):
            text = []
            for _ in range(10):#generate 10 lines
                encoded = self.token.texts_to_sequences([seed_text])
                encoded = pad_sequences(encoded, maxlen=self.seq_length, padding='pre')

                y_pred = np.argmax(self.model.predict(encoded), axis=-1)

                predicted_word = ""
                for word, index in self.token.word_index.items():
                    if index == y_pred:
                        predicted_word = word
                        break
    
                #seed_text = seed_text + ' ' + predicted_word
                text.append(predicted_word)

            seed_text = text[-1]
            text = ' '.join(text)
            print(text)
    
    def next_word(self, seed_text):
        encoded = self.token.texts_to_sequences([seed_text])
        encoded = pad_sequences(encoded, maxlen=self.seq_length, padding='pre')
        y_pred = np.random.choice(np.argsort(self.model.predict(encoded), axis=-1)[0][::-1][:4])# top 3 choices
        #y_pred = np.argmax(self.model.predict(encoded), axis=-1)
        print(y_pred)

        predicted_word = ""
        for word, index in self.token.word_index.items():
            if index == y_pred:
                return word

        return None