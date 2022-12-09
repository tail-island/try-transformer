import numpy as np
import pandas as pd

from funcy import cat, partial, repeatedly
from operator import add, floordiv, mul, sub


rng = np.random.default_rng(0)


def create_sentence(op_str, op):
    x_1 = rng.integers(1, 1_000)
    x_2 = rng.integers(1, 1_000)
    y = op(x_1, x_2)

    return f'{x_1}{op_str}{x_2}', f'{y}'


def create_data_frame(size):
    expressions, answers = zip(*cat(map(lambda op: repeatedly(partial(create_sentence, *op), size // 2),
                                        (('+', add),
                                         ('-', sub)))))

    data_frame = pd.DataFrame({'Expression': expressions, 'Answer': answers}, dtype='string')
    data_frame.index.name = 'ID'

    return data_frame


create_data_frame(20_000).to_csv('train.csv')
create_data_frame(2_000).to_csv('test.csv')
