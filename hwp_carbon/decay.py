from typing import Callable

import numpy as np
from math import exp, log
# from numba import vectorize, float64
#
# def radioactive_decay_func_numba(half_life: int) -> Callable:
#     '''Standard radioactive decay function
#     use f.accumulate(data) to calculate the remaining product with a decay_rate
#     data -> list of input values'''
#     if half_life == 0:
#         decay_rate = 0
#     elif np.isnan(half_life) or half_life < 0:
#         decay_rate = 1
#     else:
#         decay_rate = exp(-log(2) / half_life)
#
#     @vectorize([float64(float64, float64)])
#     def f(x, y):
#         return decay_rate * (x + y)
#
#     return f.accumulate


def radioactive_decay_func_numpy(half_life: int) -> Callable:
    '''Calculate the remaining product with a decay_rate
    Quicker than numba because won't be used very often and no compilation time'''
    if half_life == 0:
        decay_rate = 0
    elif np.isnan(half_life):
        decay_rate = 1
    else:
        decay_rate = exp(-log(2) / half_life)

    def f(x, y):
        return decay_rate * (x + y)

    return np.frompyfunc(f, nin=2, nout=1).accumulate
