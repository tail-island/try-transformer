from funcy import concat


BEGIN_WORD = '^'
END_WORD = '$'

X_WORDS = tuple(concat((BEGIN_WORD, END_WORD, '+', '-'),
                       map(str, range(10))))
Y_WORDS = tuple(concat((BEGIN_WORD, END_WORD, '-'),
                       map(str, range(10))))

# TODO: チューニング！　多分、問題の複雑さに比べてパラメーターが大きすぎる。

BLOCK_SIZE = 3
ATTENTION_HEAD_SIZE = 4
ATTENTION_DIMENSION_SIZE = 256
ATTENTION_DROPOUT_RATE = 0.1
FEED_FORWARD_DIMENSION_SIZE = 1024
FEED_FORWARD_DROPOUT_RATE = 0.1
X_WORD_SIZE = len(X_WORDS) + 1
X_MAX_LENGTH = 256
Y_WORD_SIZE = len(Y_WORDS) + 1
Y_MAX_LENGTH = 256

LEARNING_RATE = 0.0001
