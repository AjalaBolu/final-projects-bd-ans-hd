# Fake News Detection

Classifies news articles as real or fake using NLP and multiple ML models.

## Dataset
Fake and real news datasets sourced from Kaggle (`Fake.csv`, `True.csv`).

## Approach
- Text cleaning and preprocessing
- Feature extraction using **TF-IDF Vectorization**
- Trained and compared four classifiers: Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 98.82% |
| **Decision Tree** | **99.56%** |
| Gradient Boosting | 99.47% |
| Random Forest | ~99.0% |

Decision Tree was the best-performing model and was saved for deployment, alongside the TF-IDF vectorizer.

## Tech Stack
Python, pandas, scikit-learn, TF-IDF (NLP)
