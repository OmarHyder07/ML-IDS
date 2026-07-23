# Phase 1 — NSL-KDD

Supervised intrusion detection on the NSL-KDD benchmark, starting from a from-scratch NumPy implementation and working up to tree ensembles. The purpose of this phase was to build a baseline properly and understand its failure modes before scaling to a larger dataset. Discovered the drawbacks of a supervised-only ML IDS being the inability to classify unseen or zero-day attacks.

**Full write-up: [`writeup.pdf`](./writeup.pdf)** (LaTeX source: [`writeup.tex`](./docs/writeup.tex))

## Results

| Metric (test set) | Logistic regression |
|---|---|
| Accuracy | 0.75 |
| Precision (anomaly) | 0.92 |
| Recall (anomaly) | 0.62 |
| F1 (anomaly) | 0.74 |
| F2 (anomaly) | 0.66 |

I also built Decision tree, random forest, and XGBoost models to see if greater expressiveness could overcome the unseen-attack problem. With minimal tuning (to avoid long compute times):
| Model | F2 (anomaly) |
|---|---|
| Decision tree | 0.742 |
| Random Forest | 0.648 |
| XGBoost | 0.710 |

## The core finding

The model reaches ~96% training accuracy but only ~75% on `KDDTest+`. That 20-point gap looks like textbook overfitting — but an L2 regularisation sweep found the optimum at **λ = 0**, with train and cross-validation loss both low and near-identical. The model generalises fine *within* the training distribution.

The real cause is specific to this benchmark: `KDDTest+` deliberately contains **17 attack types that never appear in training** (`httptunnel`, `mailbomb`, `processtable`, `mscan`, `saint`, …). Splitting test recall by whether the attack family was seen during training:

| | Recall |
|---|---|
| Attack types seen in training | **0.714** |
| Attack types novel to the test set | **0.380** |

A supervised classifier cannot recognise what it has never seen. This is a ceiling that no tuning on the training distribution can lift — and it is the finding that motivates Phase 2.

| (Decision tree) | Recall |
|---|---|
| Attack types seen in training | **0.784** |
| Attack types novel to the test set | **0.502** |

## Method notes

- **From scratch.** Sigmoid, log loss, vectorised gradient, and batch gradient descent implemented in NumPy rather than called from a library — the mechanics were the point, and understanding them made several bugs diagnosable that otherwise would not have been.
- **Threshold selection.** For an IDS a false negative costs far more than a false positive, so the decision threshold was chosen to favour recall via Youden's J on the CV set, and F2 is reported alongside F1.

## Running it

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib
```
