# House Price Prediction

Predicts house prices based on property features using regression.

## Dataset
Boston Housing dataset sourced from Kaggle (`BostonHousing.csv`).

## Approach
- Exploratory data analysis to find correlations between features and price
- Trained an **XGBoost Regressor**

## Results
| Metric | Training | Testing |
|---|---|---|
| R² Score | ~1.00 | 0.893 |
| Mean Absolute Error | 0.01 | 1.96 |

The gap between training and testing R² suggests some overfitting — a good candidate for regularization tuning in a future iteration.

## Tech Stack
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn
