# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import pandas as pd
from scipy.io import arff
import statistics

# %%
# Utilities

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

def compute_gradient(X, y, w, b):
        """
        Args:
          X (ndarray (m,n))
          y (ndarray (m,))
          w (ndarray (n,))
          b (scalar)
        """
        
        m,n = X.shape
        f = sigmoid(X@w + b)
        diff = f - y
        dj_dw = (X.T @ diff) / m
        dj_db = diff.mean()
        
        return dj_dw, dj_db
                
def cost(X, y, w, b):
        """
        Args:
          X (ndarray(m,n))
          y (ndarray(m,))
          w (ndarray(n,))
          b (scalar)
        """
        
        m = X.shape[0]
        f = sigmoid(X@w + b)
        f = np.clip(f, 1e-9, 1 - 1e-9)
        cost = -y*np.log(f) - (1-y)*np.log(1-f) 
    
        return cost.mean()

def gradient_descent(X, y, w_in, b_in, alpha, iterations):
        """
        Args:
          X (ndarray(m,n))
          y (ndarray(n,))
          w (ndarray(n,))
          b (scalar)
          alpha (scalar)
          iterations (scalar)
        """
        
        J_history = []  
        n = w_in.shape
        w = copy.deepcopy(w_in)
        b = b_in
        for i in range(iterations):
                dj_dw, dj_db = compute_gradient(X, y, w, b)
                w -= alpha*dj_dw
                b -= alpha*dj_db
                J = cost(X, y, w, b)
                J_history.append(J)
                if i % math.ceil(iterations / 10) == 0:
                        print(f"Iteration {i:4d}: Cost {J_history[i]}")
        return w, b, J_history


# %%
train_data, meta = arff.loadarff('KDDTrain+.arff')
df = pd.DataFrame(train_data)
for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
for col in ['protocol_type', 'service', 'flag', 'class']:
    df[col] = df[col].astype('category').cat.codes

# %%
# split training data into X and Y
# Isolating some categories that don't need encoding or cleaning
# n-feature isolated picture to demonstrate the difference in scaled and non-scaled features
# and gain experience plotting
train_X = df.iloc[:, :41].to_numpy(dtype=float)
train_y = df.iloc[:, 41].to_numpy(dtype=float)

print(train_X[:5])
print(train_y[:5])
print(train_X.shape, train_X.dtype)      # expect (~125973, 2) and float64
print(train_y.shape, train_y.dtype)

# %%
# Fit n-feature training data to model!
# init weights and biases...
w_in = np.zeros_like(train_X[0])
b_in = 0
alpha = 0.1
iters = 10000
#scaled_train_X = (train_X - train_X.mean(axis=0)) / train_X.std(axis=0)
scaled_train_X = train_X
stdevs = train_X.std(axis=0)
stdevs = np.clip(stdevs, 1e-7, None)
scaled_train_X = (train_X - train_X.mean(axis=0)) / stdevs
scaled_train_X[4:6] = np.log1p(train_X[4:6])
print(scaled_train_X[0])
# print(np.ptp(scaled_train_X, axis=0))
# print(scaled_train_X[:5])
#plt.plot(np.zeros_like(scaled_train_X), scaled_train_X, 'x')
#plt.show()

# %%
#w,b,J_history = gradient_descent(train_X, train_y, w_in, b_in, alpha, iters)
#print(f"New weights: {w}, new bias: {b}")

# The above code does not work!
# Since there are unscaled pieces of data (which are in the thousands)
# Predictions being fed into sigmoid (X[i]@w + b)
# are so large. Coupled with exp() being... exponential, overflow is inevitable
# So, scaling is not just for better results, it is impossible not to scale!

# %%
w,b,J_history = gradient_descent(scaled_train_X, train_y, w_in, b_in, alpha, iters)
print(f"New weights: {w}, new bias: {b}")

# %%
# plot cost vs. iterations
plt.plot(np.arange(len(J_history)), J_history)
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("training loss vs. iteration")
plt.show()

# %%
preds = (sigmoid(train_X @ w + b) >= 0.5).astype(int)
print((preds == train_y).mean())

# %%
# Displays proportion of dataset that is labeled normal and attack
print(train_y.mean())
print(1 - train_y.mean())

# %%
# Below is an education plot
# Assuming training on two features (any 2, but here src and dst bytes)
# Plots a scatter plot on src vs. dst bytes, points colour coded for attack or normal classification
# with decision boundary 
# x1 = scaled_train_X[:, 0]
# x2 = scaled_train_X[:, 1]

# plt.figure(figsize=(8, 6))

# y=train_y
# # scatter the two classes separately so they get different colours + a legend
# plt.scatter(x1[y == 0], x2[y == 0], s=5, alpha=0.3, label="normal")
# plt.scatter(x1[y == 1], x2[y == 1], s=5, alpha=0.3, label="attack")

# # the decision boundary: the line where w0*x1 + w1*x2 + b = 0
# # rearranged for x2:  x2 = -(w0*x1 + b) / w1
# xs = np.linspace(x1.min(), x1.max(), 200)
# boundary = -(w[0] * xs + b) / w[1]
# plt.plot(xs, boundary, color="black", linewidth=2, label="decision boundary")

# plt.xlabel("log src_bytes (scaled)")
# plt.ylabel("log dst_bytes (scaled)")
# plt.legend()
# plt.ylim(x2.min(), x2.max())   # keep the view on the data if the line shoots off
# plt.show()

# %%
# Get test inputs
test_data, test_meta = arff.loadarff('KDDTest+.arff')
test_df = pd.DataFrame(test_data)

for col in test_df.select_dtypes([object]):
        test_df[col] = test_df[col].str.decode('utf-8')
for col in ['protocol_type', 'service', 'flag', 'class']:
    test_df[col] = test_df[col].astype('category').cat.codes

test_X = test_df.iloc[:, :41].to_numpy(dtype=float)
test_y = test_df.iloc[:, 41].to_numpy(dtype=float)

# %%
preds = (sigmoid(test_X @ w + b) >= 0.5).astype(int)
print((preds == test_y).mean())

# %%
