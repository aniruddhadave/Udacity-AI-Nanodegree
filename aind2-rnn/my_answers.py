import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


def window_transform_series(series, window_size):
    """ Transforms the input series and window-size 
        into a set of input/output pairs for use with the RNN model.
    """
    # Length of the input/output pairs
    length = len(series) - window_size
    X = [series[i: i + window_size] for i in range(length)]
    y = [series[i + window_size] for i in range(length)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y


def build_part1_RNN(window_size):
    """ Build's an RNN to perform regression 
        on our time series input/output data.
    """
    model = Sequential()
    
    # Layer 1: LSTM with 5 units 
    model.add(LSTM(5, input_shape= (window_size, 1)))
    
    # Layer 2: Dense Layer 
    model.add(Dense(1))
    
    return model


def cleaned_text(text):
    """ Lists all unique characters in the 
        text and remove any non-english ones.
    """
    # Punctuation to be included
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    # ASCII Lowercase letters
    ascii_lowercase = string.ascii_lowercase
    
    # Set of allowed characters
    alowed_characters = set(ascii_lowercase) | set(punctuation) | set([' '])
    
    # Convert text to lower case
    lower_case_text = text.lower()
    unique_characters = set(lower_case_text)
    
    # Retain only ascii lowercase and the punctuation 
    characters_to_remove = unique_characters - alowed_characters
    for ch in characters_to_remove:
        text = text.replace(ch, ' ')
    
    # Remove any double spaces created
    text.replace('  ', ' ')

    return text


def window_transform_text(text, window_size, step_size):
    """ Transforms the input text and window-size into 
        a set of input/output pairs for use with the RNN model.
    """
    length = len(text) - window_size
    inputs = [text[i: i + window_size] for i in range(0, length, step_size)]
    outputs = [text[i + window_size] for i in range(0, length, step_size)]

    return inputs,outputs

def build_part2_RNN(window_size, num_chars):
    """ Builds the required RNN model: 
        a single LSTM hidden layer with softmax activation
        and categorical_crossentropy loss.
    """
    model = Sequential()
    
    # LSTM layer with 200 units
    model.add(LSTM(200, input_shape= [window_size, num_chars]))
    
    # Output layer with softmax activation
    model.add(Dense(num_chars, activation= 'softmax'))
    
    return model