# Diabetes Prediction

Predicts diabetes risk from patient health indicators, comparing multiple ML models.

## Dataset
Diabetes dataset sourced from Kaggle (`diabetis.csv`).

## Approach
- Exploratory data analysis and preprocessing with `StandardScaler`
- Trained and hyperparameter-tuned (via `GridSearchCV`) seven classifiers: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, and XGBoost
- Compared models on accuracy, precision, recall, F1-score, and ROC-AUC

## Results
*(Notebook outputs weren't saved for the final comparison table — rerun the notebook to populate exact scores, or let me know the numbers and I'll fill this in.)*

All seven trained models are saved in `AdvancedDB.pkl` for reuse.

## Tech Stack
Python, pandas, scikit-learn, XGBoost
