import numpy as np
import tensorflow as tf

from funcy import concat, count, rcompose, repeat, take, takewhile
from parameter import BEGIN_WORD, END_WORD


def create_op(block_size,
              attention_head_size,
              attention_dimension_size,
              attention_dropout_rate,
              feed_forward_dimension_size,
              feed_forward_dropout_rate,
              x_word_size,
              x_max_length,
              y_word_size,
              y_max_length):

    # 位置エンコーディング

    def create_positional_encoding(length):
        result = np.empty((length, attention_dimension_size), dtype=np.float32)

        angles = np.arange(length)[:, np.newaxis] / np.power(10_000, 2 * np.arange(attention_dimension_size // 2) / attention_dimension_size)

        result[:, 0::2] = np.sin(angles)  # 偶数はsin
        result[:, 1::2] = np.cos(angles)  # 奇数はcos

        return tf.expand_dims(result, 0)

    # EncoderとDecoderに必要な演算を定義します。

    def Linear(dimension_size):
        return tf.keras.layers.Dense(dimension_size)

    def Dropout(rate):
        return tf.keras.layers.Dropout(rate)

    def Embedding(input_dimension_size):
        return tf.keras.layers.Embedding(input_dimension_size, attention_dimension_size)

    def FeedForward():
        return rcompose(Linear(feed_forward_dimension_size),
                        GeLU(),
                        Linear(attention_dimension_size),
                        Dropout(feed_forward_dropout_rate))

    def GeLU():
        return tf.keras.activations.gelu

    def Norm():
        return tf.keras.layers.LayerNormalization()

    def MaskedMultiHeadAttention():
        def op(*xs):
            return MultiHeadAttention()(*xs, use_causal_mask=True)

        return op

    def MultiHeadAttention():
        return tf.keras.layers.MultiHeadAttention(attention_head_size, attention_dimension_size, dropout=attention_dropout_rate)

    def Softmax():
        return tf.keras.layers.Softmax()

    # EncoderとDecoderを定義します。

    def Encoder():
        def op(x):
            x = Embedding(x_word_size)(x) * normalize_factor + positional_encoding[:, :tf.shape(x)[1]]

            for _ in range(block_size):
                x = Norm()(MultiHeadAttention()(x, x, x) + x)
                x = Norm()(FeedForward()(x) + x)

            return x

        normalize_factor = tf.math.sqrt(tf.cast(attention_dimension_size, tf.float32))
        positional_encoding = create_positional_encoding(x_max_length)

        return op

    def Decoder():
        def op(y, z):
            y = Embedding(y_word_size)(y) * normalize_factor + positional_encoding[:, :tf.shape(y)[1]]

            for _ in range(block_size):
                y = Norm()(MaskedMultiHeadAttention()(y, y, y) + y)
                y = Norm()(MultiHeadAttention()(y, z, z) + y)
                y = Norm()(FeedForward()(y) + y)

            return Softmax()(Linear(y_word_size)(y))

        normalize_factor = tf.math.sqrt(tf.cast(attention_dimension_size, tf.float32))
        positional_encoding = create_positional_encoding(y_max_length)

        return op

    # Transformerを作成します。

    def op(inputs):
        x, y = inputs

        return Decoder()(y, Encoder()(x))

    return op


def create_word_encoder(words):
    def op(x):
        return encoder[x]

    encoder = dict(zip(words, count(1)))

    return op


def create_word_decoder(words):
    def op(x):
        return decoder[x] if x != 0 else END_WORD

    decoder = dict(zip(count(1), words))

    return op


def encode_words(encoder, max_length, words):
    return tuple(take(max_length + 2, concat(map(encoder, concat((BEGIN_WORD,), words, (END_WORD,))),
                                             repeat(0))))


def decode_words(decoder, separator, word_numbers):
    return separator.join(takewhile(lambda word: word != END_WORD,
                                    map(decoder, word_numbers)))
