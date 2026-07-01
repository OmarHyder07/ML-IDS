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

# %%
# Utilities

def sigmoid(x):
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
		cost += -y[i]*math.log(fwb_i) + (1-y[i])*math.log(1-fwb_i)
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

X = np.array([[1.0, 5., 6.], [2.2, 5., -3.6], [4., 2.8, 0.]])
y = np.array([1, 1, 0])

print(X[1])

win = np.zeros_like(X[0])
print(win)

bin = 0.
alpha = 0.1
iters = 100000
w,b,_ = gradient_descent(X, y, win, bin, alpha, iters)

