#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Roughly based on: https://gist.github.com/fede-vaccaro/ac737942e233cc31ffb404bb928ee6cf

from __future__ import print_function

import numpy as np
from time import time

size = 3000
A, B = np.random.random((size, size)).astype('float32'), np.random.random((size, size)).astype('float32')

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

print('benchmark\titerations\taverage time')

# Matrix dot
N = 20
t = time()
for i in range(N):
    np.dot(A, B)
delta = time() - t
print('np.dot(A, B)\t%d\t%0.2f ms/op' % (N, delta * 1000.0 / N))

# Matrix divison
N = 30
t = time()
for i in range(N):
    np.divide(A, B)
delta = time() - t
print('np.divide(A, B)\t%d\t%0.2f ms/op' % (N, delta * 1000.0 / N))

# Matrix addition
N = 100
t = time()
for i in range(N):
    np.add(A, B)
delta = time() - t
print('np.add(A, B)\t%d\t%0.2f ms/op' % (N, delta * 1000.0 / N))

# Sigmoid of matrix
N = 50
t = time()
for i in range(N):
    sigmoid(A)
delta = time() - t
print('sigmoid(A)\t%d\t%0.2f ms/op' % (N, delta * 1000.0 / N))


print('')
print('This was obtained using the following Numpy configuration:')
np.__config__.show()
