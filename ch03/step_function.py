#!/usr/bin/env python
# coding: utf-8

# # Implement Step function

# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# ## Step function only with scalar supported
# The following scalar version of step function works with scalar values but it does not with array values.

# scalar supported
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
step_function(-0.1), step_function(0.2)    


step_function(np.array[1.0, 2.0])


# ## Step function with array supported
# 
# To support array being accepted by step_function(), we will make use of the fact that Numpy can convert numerical values to bool with astype().

a = np.arange(-3, 3).reshape((2,3)).astype(float)
b = a > 0
a, b, b.astype(int)


# array supported
def step_function(x):
    return np.array(x > 0, dtype=int)


X = np.arange(-5.0, 5.0, 0.1)
X[:9]


Y = step_function(X)
Y


plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 図で描画するy軸の範囲を指定
plt.show()

