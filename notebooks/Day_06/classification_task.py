# =========================
# Day 6: Logistic Regression Task
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create Dataset

data = {
    'Hours_Sleep': [8, 7, 6, 5, 8, 4, 9, 5, 6, 4],
    'Coffee_Cups': [1, 2, 2, 4, 0, 5, 1, 4, 3, 6],
    'Passed': [1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# 2. Define Features & Target

X = df[['Hours_Sleep', 'Coffee_Cups']]
y = df['Passed']

# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Model

clf = LogisticRegression()
clf.fit(X_train, y_train)

# 5. Predictions

y_pred = clf.predict(X_test)

print("\nModel Predictions on Test Data:", y_pred)
print("Actual Values:", list(y_test))

# 6. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 7. Classification Report

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Feature Importance

importance = clf.coef_[0]

print("\nFeature Importance:")
for i, v in enumerate(importance):
    print(f'{X.columns[i]}: {v:.4f}')

# 9. Manual Prediction (Experiment)

custom_input = pd.DataFrame([[3, 7]], columns=['Hours_Sleep', 'Coffee_Cups'])
prediction = clf.predict(custom_input)

print("\nCustom Input Prediction (3 sleep, 7 coffee):", prediction)

