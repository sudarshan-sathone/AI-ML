# Day 12: Binary Classification using Logistic Regression

## Objective
The goal of this task was to move from predicting continuous values to predicting categorical outcomes. Specifically, the task focused on binary classification using Logistic Regression.

---

## Dataset Used
- Breast Cancer Wisconsin Dataset (from sklearn)
- Target:
  - 0 → Malignant
  - 1 → Benign

---

## Steps Performed

### 1. Data Loading & Splitting
The dataset was loaded using sklearn and split into training and testing sets using an 80-20 ratio.

### 2. Feature Scaling
StandardScaler was used to normalize the feature values. This step is important for Logistic Regression as it improves model performance and convergence.

### 3. Model Training
A Logistic Regression model was trained using the scaled training data.

### 4. Predictions
Predictions were made on the test dataset using the trained model.

---

## Model Performance

- Achieved accuracy of more than 90% on the test dataset.
- Accuracy alone was not considered sufficient for evaluation.

---

## Confusion Matrix Analysis

A confusion matrix was generated to better understand the model’s performance.

It helped identify:
- Correct predictions
- False Positives
- False Negatives

This gave a clearer picture of how the model behaves in different scenarios.

---

## Class Prediction vs Probability

- `predict()` returns the final class (0 or 1)
- `predict_proba()` returns probability for each class

Example:[0.11, 0.88]

This means:
- 11% probability of malignant
- 88% probability of benign

This helps understand model confidence instead of just final output.

---

## Reflection

In a medical scenario like cancer detection, a False Negative is more dangerous than a False Positive.

A False Negative means the model predicts that a patient is healthy when they actually have cancer. This can delay diagnosis and treatment, which may lead to serious consequences.

A False Positive, although stressful and leading to extra tests, is comparatively safer because the disease is not missed.

---

## Conclusion

This task helped in understanding:
- The working of Logistic Regression
- Importance of feature scaling
- Difference between prediction and probability
- Why confusion matrix is more informative than accuracy alone

Overall, it provided a good introduction to classification problems and evaluation metrics.