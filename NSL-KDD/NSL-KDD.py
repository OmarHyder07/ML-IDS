# ---
# jupyter:
#   jupytext:
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
import pandas as pd
from sklearn import tree, metrics
from scipy.io import arff

train_data, meta = arff.loadarff('KDDTrain+.arff')
train_df = pd.DataFrame(train_data)

# %%
# label-encoding (string categories to numerical values)
train_df["protocol_type"] = train_df["protocol_type"].astype('category').cat.codes
train_df["service"] = train_df["service"].astype('category').cat.codes
train_df["flag"] = train_df["flag"].astype('category').cat.codes
train_df["class"] = train_df["class"].astype('category').cat.codes

# %%
# split training data into X and Y
train_X = train_df.iloc[:, :41]
train_Y = train_df.iloc[:, 41:]

print(train_X.head())
print(train_Y.head())

# %%
# Fit train inputs to train outputs with Decision Tree
model = tree.DecisionTreeClassifier()
model.fit(train_X, train_Y)

# %%
# Get test inputs
test_data, test_meta = arff.loadarff('KDDTest+.arff')
test_df = pd.DataFrame(test_data)
test_df["protocol_type"] = test_df["protocol_type"].astype('category').cat.codes
test_df["service"] = test_df["service"].astype('category').cat.codes
test_df["flag"] = test_df["flag"].astype('category').cat.codes
test_df["class"] = test_df["class"].astype('category').cat.codes
test_X = test_df.iloc[:, :41]
test_Y = test_df.iloc[:, 41:]

# %%
# Predict
predictions = model.predict(test_X)

# %%
print(metrics.accuracy_score(test_Y, predictions))
