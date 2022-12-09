import numpy as np
import tensorflow as tf

from funcy import partial
from dataset import get_test_dataset
from itertools import starmap
from operator import eq
from random import sample
from parameter import X_WORDS, Y_MAX_LENGTH, Y_WORDS
from transformer import BEGIN_WORD, create_decoder, create_encoder, decode, END_WORD


def translate(model, xs):
    encoder = create_encoder(Y_WORDS)

    ys = np.zeros((len(xs), Y_MAX_LENGTH), dtype=np.int32)
    ys[:, 0] = encoder(BEGIN_WORD)

    is_end = np.zeros((len(xs),), dtype=np.int32)

    for i in range(1, Y_MAX_LENGTH):
        ys[:, i] = np.argmax(model.predict_on_batch((xs, ys[:, :i]))[:, -1], axis=-1)

        is_end |= ys[:, i] == encoder(END_WORD)

        if np.all(is_end):
            break

    return ys[:, 1:]


model = tf.keras.models.load_model('model')

(xs, _), ys_true = get_test_dataset()

decoder_x, decoder_y = map(create_decoder, (X_WORDS, Y_WORDS))

ys_true_string, ys_pred_string = map(lambda ys: tuple(map(partial(decode, decoder_y, ''), ys)), (ys_true, translate(model, xs)))

print(f'Accuracy = {sum(starmap(eq, zip(ys_true_string, ys_pred_string))) / len(xs)}')

for x, y_true_string, y_pred_string in sample(tuple(zip(xs, ys_true_string, ys_pred_string)), 20):
    print(f'{decode(decoder_x, "", x[1:])} {"==" if y_true_string == y_pred_string else "!="} {y_pred_string} ({y_true_string})')
