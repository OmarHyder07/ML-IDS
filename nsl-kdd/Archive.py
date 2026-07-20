# ---
# jupyter:
#   jupytext:
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
# Displays proportion of dataset that is labeled normal and attack
print(train_y.mean())
print(1 - train_y.mean())

# %%
#w,b,J_history = gradient_descent(train_X, train_y, w_in, b_in, alpha, iters)
#print(f"New weights: {w}, new bias: {b}")

# The above code does not work!
# Since there are unscaled pieces of data (which are in the thousands)
# Predictions being fed into sigmoid (X[i]@w + b)
# are so large. Coupled with exp() being... exponential, overflow is inevitable
# So, scaling is not just for better results, it is impossible not to scale!

# %%
print(w_in)
print(b_in)
#print(scaled_train_X[0])
# print(np.ptp(scaled_train_X, axis=0))
# print(scaled_train_X[:5])
#plt.plot(np.zeros_like(scaled_train_X), scaled_train_X, 'x')
#plt.show()

# %%
# Creating a confusion matrix on the test set
from sklearn import metrics
tn, fp, fn, tp = metrics.confusion_matrix(test_y, preds).ravel().tolist()
print("  Postitive    Negative [predicted]")
print(f"P  {tp}         {fn}")
print(f"N  {fp}         {tn}")
# In our context:
# True Positive is a correctly classified n
# False Positive: normal traffic labeled as an attack
# True Negative: 
