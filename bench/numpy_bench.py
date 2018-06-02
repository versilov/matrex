#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Roughly based on: https://gist.github.com/fede-vaccaro/ac737942e233cc31ffb404bb928ee6cf
# Logistic regression code from https://github.com/chrismcg/machine_learning_ng

from __future__ import print_function

import numpy as np
from time import time

size = 3000
A, B = np.random.random((size, size)).astype('float32'), np.random.random((size, size)).astype('float32')

theta_t = np.array(range(0, 401)).reshape((401, 1))
X_t = np.c_[np.ones((5000, 1)), np.array(range(0, 2000000)).reshape((5000, 400), order='F') / 10]
y_t = np.random.randint(2, size=5000).reshape(5000, 1)
lambda_t = 3

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def logistic_cost(theta, x, y, reg_lambda):
    m = y.size
    # ensure it's a vector not an array for dot with reshape below
    h = sigmoid(x.dot(theta.reshape((theta.size, 1))))

    y_transpose = y.transpose()
    j = (-y_transpose.dot(np.log(h)) - ((1 - y_transpose).dot(np.log(1 - h)))) / m

    regularization = (reg_lambda / (2 * m)) * (theta[1:] ** 2).sum()
    j = j + regularization

    gradients = (x.transpose().dot(h - y)) / m
    temp = theta.copy().reshape((theta.size, 1))
    temp[0] = 0
    gradients = gradients + ((reg_lambda / m) * temp)

    return (j[0][0], gradients.flatten())


print('benchmark\titerations\taverage time')

# Logistic regression cost function
N = 1000
t = time()
for i in range(N):
    logistic_cost(theta_t, X_t, y_t, lambda_t)
delta = time() - t
print('logistic_cost()\t%d\t%0.2f ms/op' % (N, delta * 1000.0 / N))

# Matrix dot
N = 20
t = time()
for i in range(N):
    np.dot(A, B)
delta = time() - t
print('np.dot(A, B)\t%d\t%0.2f ms/op' % (N, delta * 1000.0 / N))

# Matrix divison
N = 100
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
