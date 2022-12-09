import numpy as np
import os.path as path
import pandas as pd

from funcy import partial
from parameter import X_WORDS, Y_WORDS
from transformer import create_encoder, encode


def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', filename), dtype={'Expression': 'string', 'Answer': 'string'}, index_col='ID')


def get_dataset(filename):
    data_frame = get_data_frame(filename)

    x_strings = data_frame['Expression']
    y_strings = data_frame['Answer']

    xs = np.array(tuple(map(partial(encode, create_encoder(X_WORDS), max(map(len, x_strings))), x_strings)), dtype=np.int32)
    ys = np.array(tuple(map(partial(encode, create_encoder(Y_WORDS), max(map(len, y_strings))), y_strings)), dtype=np.int32)

    return (xs, ys[:, :-1]), ys[:, 1:]


def get_train_dataset():
    return get_dataset('train.csv')


def get_test_dataset():
    return get_dataset('test.csv')
