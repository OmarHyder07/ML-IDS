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
        dj_dw = np.zeros((n,))
        dj_db = 0.
        
        for i in range(m):
                fwb_i = sigmoid(X[i]@w + b)
                diff = fwb_i - y[i]
                for j in range(n):
                        dj_dw[j] += diff*X[i, j] 
                dj_db += diff
        dj_dw /= m
        dj_db /= m      
        return dj_dw, dj_db
                
def cost(X, y, w, b):
        """
        Args:
          X (ndarray(m,n))
          y (ndarray(n,))
          w (ndarray(n,))
          b (scalar)
        """
        
        m = X.shape[0]
        cost = 0
        for i in range(m):
                fwb_i = sigmoid(X[i]@w + b)
                if fwb_i == 0.0:
                        fwb_i = 0.000001
                if fwb_i == 1.0:
                        fwb_i = 0.999999999999999
                cost += -y[i]*math.log(fwb_i) - (1-y[i])*math.log(1-fwb_i)
        cost /= m
        return cost

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
# Isolating just the categories src_bytes and dst_bytes
# 2-feature isolated picture to demonstrate the difference in scaled and non-scaled features
# and gain experience plotting
train_X = df.iloc[:, 4:6].to_numpy(dtype=float)
train_y = df.iloc[:, 41].to_numpy(dtype=float)

print(train_X[:5])
print(train_y[:5])
print(train_X.shape, train_X.dtype)      # expect (~125973, 2) and float64
print(train_y.shape, train_y.dtype)

# %%
# Fit 2-feature (src_bytes, dst_bytes) training data to model!
# init weights and biases...
w_in = np.zeros_like(train_X[0])
b_in = 0
alpha = 0.01
iters = 9

# %%
w,b,J_history = gradient_descent(train_X, train_y, w_in, b_in, alpha, iters)
print(f"New weights: {w}, new bias: {b}")

# The above code does not work!
# Since there are unscaled pieces of data (which are in the thousands)
# Predictions being fed into sigmoid (X[i]@w + b)
# are so large. Coupled with exp() being... exponential, overflow is inevitable
# So, scaling is not just for better results, it is impossible not to scale!

# %%
def z_scale(X):
    # X (ndarray(m,n))

    m,n = X.shape
    scaled_X = copy.deepcopy(X)
    averages = []
    stdevs   = []
    for i in range(n):
        averages.append([col[i] for col in X])
    for i in range(n):
        stdevs.append(statistics.stdev(averages[i]))
        averages[i] = np.mean(averages[i])
    for i in range(n):
        scaled_X -= averages
        scaled_X /= stdevs
    return scaled_X
    


# %%
scaled_train_X = z_scale(train_X)
w,b,J_history = gradient_descent(scaled_train_X, train_y, w_in, b_in, alpha, iters)
print(f"New weights: {w}, new bias: {b}")

# %%
# plot cost vs. iterations
plt.plot(np.arange(len(J_history)), J_history)
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("training loss vs. iteration")
plt.show()
