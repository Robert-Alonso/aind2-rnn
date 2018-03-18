import re
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import keras


# DONE: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    for i in range(window_size + 1):
        X.append(np.roll(series, -i)[:-window_size])
    X = np.array(X).T
    X, y = X[:,:-1], X[:,-1:]
    
    # reshape each 
    # X = np.asarray(X)
    # X.shape = (np.shape(X)[0:2])
    # y = np.asarray(y)
    # y.shape = (len(y),1)

    return X,y

# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(32, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model

def build_part1_RNN_large(window_size, n_units):
    model = Sequential()
    model.add(LSTM(n_units, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.5))
    model.add(LSTM(n_units))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
              
              
### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    return re.sub(f"[^a-zA-Z{''.join(punctuation)}]+", ' ', text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
