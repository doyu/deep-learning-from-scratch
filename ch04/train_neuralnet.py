#!/usr/bin/env python
# coding: utf-8

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
#from dataset.mnist import load_mnist
#from two_layer_net import TwoLayerNet


# Alternate data since mnist takes too long
def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # クラスごとのサンプル数
    DIM = 2  # データの要素数
    CLS_NUM = 3  # クラス数

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=int)

    for j in range(CLS_NUM):
        for i in range(N):#N*j, N*(j+1)):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t

x, t = load_data()
idx = np.random.permutation(len(x))
x, t = x[idx], t[idx]
plt.scatter(x[:,0], x[:,1], c=t.argmax(axis=1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
]    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad


#%%time
iters_num = 300  # 繰り返しの回数を適宜設定する
hidden_size = 10
batch_size = 30
learning_rate = 1.0

W1 = 0.01 * np.random.randn(len(x[0]), hidden_size)
b1 = np.zeros(1)
W2 = 0.01 * np.random.randn(hidden_size, len(t[0]))
b2 = np.zeros(1)

def predict(x):
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = a2
    return y

def loss(x, t):
    z = predict(x)
    y = softmax(z)
    return cross_entropy_error(t, y)

def update_params(x, t):
    params = [W1, b1, W2, b2]
    f = lambda param: loss(x, t)
    grads = [numerical_gradient(f, p) for p in params]
    for g, p in zip(grads, params):
        p -= learning_rate * g

loss_list = []
for i in range(300):
    idx = np.random.permutation(len(x))
    x, t = x[idx], t[idx]
    for j in range(len(x)//batch_size):
        idx = range(j*batch_size, (j+1)*batch_size)
        xx, tt = x[idx], t[idx]
        _loss = loss(xx, tt)
        loss_list.append(_loss)
        update_params(xx, tt)

print(_loss)
plt.plot(range(len(loss_list)), loss_list)
plt.show()

