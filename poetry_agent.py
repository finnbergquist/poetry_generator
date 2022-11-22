'''
Author: Finn Bergquist
This module impelements the training, loading, and querying of a recurrent
neural network for poetry generation.
'''
from typing import Sequence
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random

class PoetryAgent:
    '''
    The poetry agent holds all the important ml models for the poetry generation.
    It implements a tensorflow neural network, using recurrent reinforcement
    learning. 
    '''
    def __init__(self):
        '''Initializes poetry agent class. Prepares data for training'''
        #reading csv for training
        with open('data.txt') as f:
            data = f.readlines()

        #model information
        self.token =Tokenizer()
        self.token.fit_on_texts(data)
        self.sequential_encoded_text = self.token.texts_to_sequences(data)
        self.X, self.y, self.vocab_size, self.seq_length = self.prepare_data()
        self.model = Sequential()

        #Hyperparameters
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.01

        #keeps track of how many times each word occured
        self.word_count = {}

    def save(self, path):
        '''Saves the model to specified path location in directory'''
        self.model.save(path)
        return
    
    def load(self, path):
        '''Loads model from a specified path, that sas saved using the save()
        method'''
        self.model = keras.models.load_model(path)
        return

    def prepare_data(self):
        '''Organize training data into X and y batches. Also this will keep 
        track of vocab_size which is the number of words per token and the 
        seq_length which is the length of each sequenece of words in training
        data. Returns independent data, dependent data, vocabulary size, and 
        the sequence length'''
        #Sequential data formatted for data list to be passed to pad_sequences()
        datalist = []
        for d in self.sequential_encoded_text:
            if len(d)>1:
                for i in range(2, len(d)):
                    datalist.append(d[:i])
        
        #Organize sequences and then split -> X & y
        vocab_size = len(self.token.word_counts) + 1
        sequences = pad_sequences(datalist, maxlen=20, padding='pre')
        X = sequences[:, :-1]#Everything excpet for the last element
        y = sequences[:, -1]#Just the last element
        y = to_categorical(y, num_classes=vocab_size)
        seq_length = X.shape[1]

        return X, y, vocab_size, seq_length

    def train(self):
        '''The training process uses two long short-term memory cells with 100 
        inputs. It also uses a rectified linear activation unit and a softmax
        activation unit. This makes an output vector a probability distribution.
        This output is ideal beacuase then we can select from the top arguments
        which words are the best fit.
        '''
        #Add Neural network layers
        self.model.add(Embedding(self.vocab_size, 50, input_length=self.seq_length))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        
        #rectified linear activation unit
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.vocab_size, activation='softmax'))

        #define loss function 
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt,
        metrics=['accuracy'])

        #train with the data gathered from prepare_data()
        self.model.fit(self.X, self.y, batch_size=self.batch_size, epochs=self.epochs)

    def encoding_to_txt(self, encoding):
        """Returns the string corresponding to an encoded token"""
        ret_word = None
        for word, index in self.token.word_index.items():
            if index == encoding:
                ret_word = word
        return ret_word
    
    def next_word(self, seed_text):
        """This method is used for the generation of poretry, one word at a time. 
        The model makes a prediction based on the seed_text. It chooses a word
        from the top 10 predicted words randomly, and retruns that. Returns the
        selected word and itse deviation from the argmax word"""

        #encode seed_text and randomly select from top 4 predictions
        encoded = self.token.texts_to_sequences([seed_text])
        encoded = pad_sequences(encoded, maxlen=self.seq_length, padding='pre')
        model_prediction = self.model.predict(encoded)
        y_pred = np.random.choice(np.argsort(model_prediction, axis=-1)[0][::-1][:10])
        y_arg_max = np.argmax(model_prediction, axis=-1)[0]
        error = abs(y_pred - y_arg_max)

        #decode selected word
        ret_word = self.encoding_to_txt(y_pred)

        #If word has been used more than 4 times->generate a new one at random
        if ret_word in self.word_count:
            if self.word_count[ret_word] >= 4:
                ret_word = random.choice(list(self.token.word_index.keys()))
            else:
                self.word_count[ret_word] += 1
        else:
            self.word_count[ret_word] = 1
        
        return ret_word, error