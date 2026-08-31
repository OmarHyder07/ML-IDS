Notebooks for the write-up: ["An Exploration of Machine Learning for zero-day detection in network Intrusion Detection Systems"](cic-ids2017/zero-days.pdf)

ML intrusion detectors that score 99%+ on CIC-IDS2017 collapse to ~1% attack recall when the attacks in the test set were never seen in training. This project measures that gap and what anomaly detection recovers.

## Abstract

Machine-learning (ML) intrusion detection systems (IDS) routinely report > 99% accuracy, but on random splits that expose every attack type during training. Held-out splits with unseen attack classes in the test set cause dramatic declines in model performance. With exploit de-ployment now routinely preceding patches and AI compressing attacker timelines[5], zero-day detection capability is the true deployment-relevant metric, not known-attack accuracy. This write-up seeks to investigate IDS performance upon introduction to unseen, zero-day attacks. Two experiments were conducted on the CIC-IDS2017 dataset: a random split and an unseen-class split, comparing XGBoost, Isolation Forest (iForest) and Local Outlier Factor (LOF) models. With all attack classes seen, XGBoost reaches a false-positive rate (FPR) ∼1-2% and a macro attack recall of 0.86 (weighted 0.99). With none seen, macro attack recall falls to 0.23 (weighted 0.01). Unsupervised anomaly detectors recover 0.67-0.76 macro attack recall at the cost of 37-46% FPR. The relative ranking of iForest and LOF anomaly detectors reverses between validation and test depending on which unseen attacks are present; even zero-day focused evaluation can mislead. The gap between known-attack and zero-day performance is large, and appears to persist across architectures. The results argue for held-out class splits and greater emphasis on zero-day capabilities to be standard in future ML IDS research.

## Results
Figure 3: Weighted attack recall vs. FPR for iForest and LOF on test. Unseen attacks: Web attacks, Infiltration, DDoS, PortScan, Botnet. 

![alt text](https://github.com/OmarHyder07/ML-IDS/blob/main/cic-ids2017/docs/Screenshot%20From%202026-08-26%2023-03-08.png?raw=true)

Table 7: Comparison of supervised and unsupervised on Random and Temporal splits of data. Displays high performance on random split and extreme drop-off on time-based / held-out split. Anomaly detectors recover attack-recall at the cost of extreme FPR and low alert precision.
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
Download flow data (CSVs/MachineLearningCSV.zip) from [cicresearch](https://cicresearch.ca/CICDataset/CIC-IDS-2017/). Extract in ```cic-ids2017/data/```. Create virtual environment, python v3.14.4, install dependencies in venv via ```pip install requirements.txt```. Run notebooks in order. (Code ran on Ubuntu 26.04 LTS)
