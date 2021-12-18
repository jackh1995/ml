from tkinter.constants import Y
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
from numpy.linalg import inv

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, type=str)
parser.add_argument('-m', '--mode', required=True, type=str)
parser.add_argument('-e', '--epoch', required=False, type=int, default=100)
args = parser.parse_args()

# np.random.seed(0)

# Data preparation
df = pd.read_csv(args.path, names=['x', 'y'])
x_list = df['x']
y_list = df['y']

# Weight initialization
w_hat = np.random.normal()
b_hat = np.random.normal()
x_train, x_val, y_train, y_val = train_test_split(x_list, y_list, test_size=0.33, random_state=0)

# ----------------------------- GRADIENT DESCENT ----------------------------- #

# Define our model
def model(x):
    global w_hat, b_hat
    return w_hat * x + b_hat

# Loss function
def mse(y, y_hat):
    return np.sum((y - y_hat)**2)**0.5

# One epoch training
def onetrain(lr):
    global w_hat, b_hat
    y_hat = model(x_train)
    grad_w = np.sum((y_train - y_hat) * (-x_train))
    grad_b = np.sum((y_train - y_hat) * (-1))
    w_hat = w_hat - lr * grad_w
    b_hat = b_hat - lr * grad_b
    return y_hat

# Training (learning)
def train(n_epoch, lr):
    global w_hat, b_hat
    mse_list = []
    for epoch in range(n_epoch):
        y_hat = onetrain(lr)
        mse_list.append(mse(y_train, y_hat))

# ----------------------------- ANALYTIC SOLUTION ---------------------------- #

def anaSol():
    X = np.stack([x_list, np.ones(len(x_list))]).T
    return inv(X.T @ X) @ X.T @ y_list

# ---------------------------------- SKLEARN --------------------------------- #

from sklearn.linear_model import LinearRegression
def sklearnLR():
    
    reg = LinearRegression().fit([[x] for x in x_list], y_list)
    return reg.coef_[0], reg.intercept_

if __name__ == '__main__':
    if args.mode == 'gd':
        train(n_epoch=args.epoch, lr=1e-5)
        print(f'w_hat={w_hat:.3f}, b_hat={b_hat:.3f}')
    elif args.mode == 'a':
        w_hat, b_hat = anaSol()
        print(f'w_hat={w_hat:.3f}, b_hat={b_hat:.3f}')
    elif args.mode == 'sk':
        w_hat, b_hat = sklearnLR()
        print(f'w_hat={w_hat:.3f}, b_hat={b_hat:.3f}')
    else:
        print('Warning: Invalid mode')