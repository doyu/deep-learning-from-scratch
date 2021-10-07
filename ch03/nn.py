#!/usr/bin/env python
# coding: utf-8

import numpy as np


# # NN with input and output lyaer
# ![](./nn1.png)

X = np.array([1, 2])
X.shape


W = np.array([[1, 3, 5], [2, 4, 6]])
W.shape


Y = np.dot(X, W)
Y.shape


# # NN with 2 hidden layer
# ![](./nn2.png)
# ![](./nn3.png)
# ![](./nn4.png)
# ![](./nn5.png)

# Input -> Layer 1
X = np.random.randn(2); print(X)
W1 = np.random.randn(2,3)
B1 = np.arange(3)

A1 = np.dot(X, W1) + B1
A1.shape


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Z1 = sigmoid(A1)

print(np.vstack((A1, Z1)))
print(Z1.shape)


# ![](./nn6.png)

# Layer 1 -> Layer 2
W2 = np.random.randn(3, 2)
B2 = np.random.randn(2)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
Z2.shape


# ![](./nn7.png)

# Layer 2 -> Output

W3 = np.random.randn(2,2)
B3 = np.random.randn(2)
A3 = np.dot(Z2, W3) + B3
Y = np.identity(2) @ A3
Y.shape

