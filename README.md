Notebooks for the write-up: ["An Exploration of Machine Learning for zero-day detection in network Intrusion Detection Systems"](cic-ids2017/zero-days.pdf)
## Abstract

Machine-learning intrusion detection systems routinely report >99% accuracy, but on random splits that expose every attack type during training. This write-up seeks to investigate what happens when models are introduced to zero-days. I conducted two experiments on the CIC-IDS2017 dataset: a random split and an unseen-class split, comparing XGBoost, Isolation Forest and Local Outlier Factor models. With all attack classes seen, XGBoost reaches a false-positive rate ~1-2% and a macro attack recall ~0.85. With none seen, macro attack recall falls to 0.23 and weighted recall to 0.01. Unsupervised anomaly detectors recover 0.67-0.76 macro attack recall at the cost of 37-46% FPR. The relative ranking of iForest and LOF anomaly detectors reverses between validation and test depending on which unseen attacks are present, suggesting even zero-day focused evaluation can mislead. The gap between known-attack and zero-day performance is large, and appears to persist across architectures. Through this write-up I advocate for held-out class splits and greater emphasis on zero-day capabilities to be standard in future machine-learning intrusion detection systems research.

## Results
Table 7:
| Model | Split | Test attack classes seen in training | FPR | Alert precision | Macro attack recall | Weighted attack recall |
|---|---|---|---|---|---|---| 
|XGBoost|Random|14/14|0.0191|0.91|0.86|0.99|
|XGBoost|Temporal|0/7|0.0115|0.19|0.23|0.01|
|iForest|Temporal|0/7|0.3744|0.29|0.67|0.46|
|LOF|Temporal|0/7|0.4602|0.40|0.76|0.91|

## Directories
```cic-ids2017``` - write-up notebooks: Imbalance handling experiment, supervised temporal performance & anomaly detection.

```nsl-kdd``` - Initial exploration of supervised ML-IDS: logistic regression and decision trees.

## How to reproduce results
Download flow data (CSVs/MachineLearningCSV.zip) from [cicresearch](https://cicresearch.ca/CICDataset/CIC-IDS-2017/). Extract in ```cic-ids2017/data/```. Run notebooks in order. 
