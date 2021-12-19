import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data.txt', names=['x', 'y'])
x_list = df['x']
y_list = df['y']

def mse(y, y_hat):
    return np.sum((y - y_hat)**2)**0.5

@np.vectorize
def loss(w_hat, b_hat):
    global x
    y_hat = w_hat * x_list + b_hat
    return mse(y_list, y_hat)

w_hat = np.random.normal()
b_hat = np.random.normal()

ww, bb = np.mgrid[-10:20, -10:20]
z = loss(ww, bb)
plt.contourf(ww, bb, z)
plt.scatter([7], [4], s=30, c='red', label='Theoretical')
plt.legend()
plt.savefig('foo')