# Flood Prediction

Predicts flood risk from environmental and geographic data.

## Dataset
Flood dataset sourced from Kaggle (`flood.csv`).

## Approach
- Label encoding of categorical features
- Trained a **Random Forest Classifier** (100 estimators)

## Results
| Metric | Score |
|---|---|
| Accuracy | 90.19% |
| Precision (class 0 / class 1) | 0.87 / 0.94 |
| Recall (class 0 / class 1) | 0.95 / 0.85 |
| F1-score (class 0 / class 1) | 0.91 / 0.89 |

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn
