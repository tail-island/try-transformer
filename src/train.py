import tensorflow as tf

from dataset import get_train_dataset
from parameter import ATTENTION_DIMENSION_SIZE, ATTENTION_DROPOUT_RATE, ATTENTION_HEAD_SIZE, BLOCK_SIZE, FEED_FORWARD_DIMENSION_SIZE, FEED_FORWARD_DROPOUT_RATE, LEARNING_RATE, X_MAX_LENGTH, X_WORD_SIZE, Y_MAX_LENGTH, Y_WORD_SIZE
from sklearn.model_selection import train_test_split
from transformer import create_op
# , LearningRateSchedule, Loss


def loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')(y_true, y_pred) *
                          tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32))


(xs_1, xs_2), ys = get_train_dataset()

train_xs_1, valid_xs_1, train_xs_2, valid_xs_2, train_ys, valid_ys = train_test_split(xs_1, xs_2, ys, test_size=0.1, random_state=0)

inputs = tf.keras.Input(shape=(None,)), tf.keras.Input(shape=(None,))

op = create_op(BLOCK_SIZE,
               ATTENTION_HEAD_SIZE,
               ATTENTION_DIMENSION_SIZE,
               ATTENTION_DROPOUT_RATE,
               FEED_FORWARD_DIMENSION_SIZE,
               FEED_FORWARD_DROPOUT_RATE,
               X_WORD_SIZE,
               X_MAX_LENGTH,
               Y_WORD_SIZE,
               Y_MAX_LENGTH)

model = tf.keras.Model(inputs, op(inputs))
# model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(ATTENTION_DIMENSION_SIZE), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=Loss())
model.compile(tf.keras.optimizers.experimental.AdamW(LEARNING_RATE), loss=loss)
model.summary()

model.fit((train_xs_1, train_xs_2), train_ys, 256, 100, validation_data=((valid_xs_1, valid_xs_2), valid_ys))
model.save('model', include_optimizer=False)