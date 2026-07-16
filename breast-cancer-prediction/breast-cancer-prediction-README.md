# Breast Cancer Prediction

Classifies tumors as benign or malignant using a neural network.

## Dataset
Breast cancer dataset sourced from Kaggle (`breast_cancerkd.csv`).

## Approach
- Feature standardization with `StandardScaler`
- Built and trained a **Neural Network** using TensorFlow/Keras (Dense layers, ReLU + Sigmoid activations)

## Results
| Metric | Score |
|---|---|
| Testing Accuracy | 95.61% |

Model and scaler are saved (`AdvancedbreastDnn.keras`, `adbreast_scaler.pkl`) for reuse in a prediction system.

## Tech Stack
Python, pandas, scikit-learn, TensorFlow/Keras
