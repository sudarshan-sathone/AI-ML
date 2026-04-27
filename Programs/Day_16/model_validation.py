from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import numpy as np

# 1. Load Data
digits = load_digits()
X, y = digits.data, digits.target

# 2. Define the Model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 3. Apply 5-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=5)

print("===== DAY 16: MODEL VALIDATION & K-FOLD =====")
print("\n[1] 5-Fold Cross-Validation Results")
print(f"Scores for each fold: {scores}")
print(f"Mean Accuracy: {np.mean(scores):.4f}")
print(f"Standard Deviation (Stability): {np.std(scores):.4f}")

# 4. Detect Overfitting / Generalization
model.fit(X, y)
train_score = model.score(X, y)

print("\n[2] Training vs Validation Comparison")
print(f"Training Accuracy: {train_score:.4f}")
print(f"Validation Accuracy: {np.mean(scores):.4f}")

if train_score - np.mean(scores) > 0.05:
    print("Observation: The model shows signs of OVERFITTING.")
elif train_score < 0.70 and np.mean(scores) < 0.70:
    print("Observation: The model shows signs of UNDERFITTING.")
else:
    print("Observation: The model is GENERALIZING reasonably well.")

# 5. Shuffle Test - 10 Fold without shuffling
kf_no_shuffle = KFold(n_splits=10, shuffle=False)
scores_no_shuffle = cross_val_score(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y,
    cv=kf_no_shuffle
)

print("\n[3] 10-Fold Cross-Validation WITHOUT Shuffle")
print(f"Scores: {scores_no_shuffle}")
print(f"Mean Accuracy: {np.mean(scores_no_shuffle):.4f}")
print(f"Standard Deviation: {np.std(scores_no_shuffle):.4f}")

# 6. Shuffle Test - 10 Fold with shuffling
kf_shuffle = KFold(n_splits=10, shuffle=True, random_state=42)
scores_shuffle = cross_val_score(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y,
    cv=kf_shuffle
)

print("\n[4] 10-Fold Cross-Validation WITH Shuffle")
print(f"Scores: {scores_shuffle}")
print(f"Mean Accuracy: {np.mean(scores_shuffle):.4f}")
print(f"Standard Deviation: {np.std(scores_shuffle):.4f}")

print("\n[5] Shuffle Test Observation")
if np.std(scores_shuffle) < np.std(scores_no_shuffle):
    print("Observation: Shuffling improved the stability of cross-validation scores.")
else:
    print("Observation: Shuffling did not improve stability in this run.")