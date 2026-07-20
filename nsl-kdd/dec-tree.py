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

# %% [markdown]
# # Evaluating if decision trees are better suited for an IDS than logistic regression

# %%
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, fbeta_score, accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import math

RANDOM_STATE = 5
np.random.seed(RANDOM_STATE)


# %% [markdown]
# ## Scaling
# Per-feature: log1p for the heavy-tailed byte columns, standardise the rest.

# %%
def fit_scaler(X, std_cols):
    mean = X[std_cols].mean()
    std  = np.clip(X[std_cols].std(), 1e-7, None)
    return mean, std

def apply_scaler(X, mean, std, std_cols, log_cols):
    scaled = X.copy()
    scaled[std_cols] = (X[std_cols] - mean) / std
    scaled[log_cols] = np.log1p(X[log_cols])
    return scaled


# %% [markdown]
# ## Evaluation helper

# %%
def evaluate(preds, y, title=""):
    """Print report + F2, show confusion matrix. Returns the predictions."""
    print(f"--- {title} ---")
    print(f"accuracy: {(preds == y).mean():.4f}")
    print(classification_report(y, preds, target_names=["normal", "anomaly"]))
    print(f"F2-score: {fbeta_score(y, preds, beta=2):.4f}\n")
    cm = confusion_matrix(y, preds, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["normal", "anomaly"]).plot()
    plt.title(title)
    return preds


# %% [markdown]
# Load & Encode NSL-KDD data
#
# LabelEncoder fitted on train, applied to test. one-hot encoding applied to 'service' column.

# %%
train_data, _ = arff.loadarff('data/KDDTrain+.arff')
test_data, _ = arff.loadarff('data/KDDTest+.arff')
df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Decode bytes
for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')
        test_df[col] = test_df[col].str.decode('utf-8')

# Encode class column
cats = ['normal', 'anomaly']  # index 0 = normal, index 1 = anomaly
train_class    = pd.Categorical(df['class'], categories=cats).codes
test_class      = pd.Categorical(test_df['class'], categories=cats).codes

# one-hot encode categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
ohe.fit(df[categorical_cols])

train_one_hot = ohe.transform(df[categorical_cols]); test_one_hot = ohe.transform(test_df[categorical_cols])
df = df.drop(categorical_cols + ['class'], axis=1)
test_df = test_df.drop(categorical_cols + ['class'], axis=1)

# Scale numeric columns
log_cols = ['src_bytes', 'dst_bytes']
standard_cols = [c for c in df.columns if c not in log_cols + ['class']]
df[standard_cols] = df[standard_cols].apply(pd.to_numeric, errors='coerce')
test_df[standard_cols] = test_df[standard_cols].apply(pd.to_numeric, errors='coerce')
mean, std = fit_scaler(df, standard_cols)
scaled_num_train_X = apply_scaler(df, mean, std, standard_cols, log_cols)
scaled_num_test_X  = apply_scaler(test_df, mean, std, standard_cols, log_cols)

# training sets below currently hold only numeric columns
train_X = df.to_numpy(dtype=float)
train_y = train_class
test_X = test_df.to_numpy(dtype=float)
test_y = test_class

# Stick one-hot encoded categorical columns back onto the numeric (scaled) columns
scaled_train_X = np.hstack((scaled_num_train_X, train_one_hot))
scaled_test_X  = np.hstack((scaled_num_test_X, test_one_hot))

# Take 20% of training set to form cross validation set
scaled_train_X, cv_X, train_y, cv_y = train_test_split(
    scaled_train_X, train_y, test_size=0.2, stratify=train_y, random_state=0
)

# %% [markdown]
# ## Decision Tree
# First we will train a single decision tree. We will tune two hyper-parameters:
# - minimum samples to allow a further split;
# - maximum tree depth.

# %%
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.

# %%
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(scaled_train_X,train_y) 
    predictions_train = model.predict(scaled_train_X) ## The predicted values for the train dataset
    predictions_val = model.predict(cv_X) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,train_y)
    accuracy_val = accuracy_score(predictions_val,cv_y)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])

# %%
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(scaled_train_X,train_y) 
    predictions_train = model.predict(scaled_train_X) ## The predicted values for the train dataset
    predictions_val = model.predict(cv_X) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,train_y)
    accuracy_val = accuracy_score(predictions_val,cv_y)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])

# %% [markdown]
# For this dataset, we see increasing the min_samples_split leads to lower train and CV accuracies, with the opposite true for max_depth. A very much aligned CV and train accuracy with these hyper-paramters is aligned with the results from logistic regression. In an attempt to keep overfitting low, I will select the following values for these hyper-parameters:
# - `max_depth = 4`
# - `min_samples_split = 200`

# %%
model = DecisionTreeClassifier(max_depth = 4,
                               min_samples_split = 200,
                               random_state = RANDOM_STATE).fit(scaled_train_X, train_y)
evaluate(model.predict(scaled_train_X), train_y, title="DEC TREE TRAIN")
evaluate(model.predict(scaled_test_X), test_y, title="DEC TREE TEST")

# %% [markdown]
# ## Random Forest
# Secondly, we will train a random forest model. We will tune for the same two previous hyper-paramaters, along with the new number of estimators. {FUTURE ME: use GridSearchCV instead!}

# %%
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                             ## If it is a float, then it is the percentage of the dataset
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]

# %%
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(scaled_train_X,train_y) 
    predictions_train = model.predict(scaled_train_X) ## The predicted values for the train dataset
    predictions_val = model.predict(cv_X) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,train_y)
    accuracy_val = accuracy_score(predictions_val,cv_y)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])

# %%
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(scaled_train_X,train_y) 
    predictions_train = model.predict(scaled_train_X) ## The predicted values for the train dataset
    predictions_val = model.predict(cv_X) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,train_y)
    accuracy_val = accuracy_score(predictions_val,cv_y)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])

# %%
accuracy_list_train = []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE).fit(scaled_train_X,train_y) 
    predictions_train = model.predict(scaled_train_X) ## The predicted values for the train dataset
    predictions_val = model.predict(cv_X) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,train_y)
    accuracy_val = accuracy_score(predictions_val,cv_y)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])

# %% [markdown]
# Let's then fit a random forest with the following parameters:
#
#  - max_depth: 16
#  - min_samples_split: 700 (train and CV cost begin to converge)
#  - n_estimators: 100

# %%
model = RandomForestClassifier(max_depth = 16,
                               min_samples_split = 700,
                               n_estimators = 100,
                               random_state = RANDOM_STATE).fit(scaled_train_X, train_y)
evaluate(model.predict(scaled_train_X), train_y, title="RANDOM FOREST TRAIN")
evaluate(model.predict(scaled_test_X), test_y, title="RANDOM FOREST TEST")

# %% [markdown]
# ## XGBoost
# Finally, we will train an XGBoost ensemble.

# %%
xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE,  early_stopping_rounds=10)
xgb_model.fit(scaled_train_X,train_y, eval_set = [(cv_X,cv_y)])

# %%
xgb_model.best_iteration

# %%
evaluate(xgb_model.predict(scaled_train_X), train_y, title="XGB TRAIN")
evaluate(xgb_model.predict(scaled_test_X), test_y, title="XGB TEST")

# %%
