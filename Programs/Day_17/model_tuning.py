from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Baseline model
baseline_model = RandomForestClassifier(random_state=42)
baseline_scores = cross_val_score(baseline_model, X, y, cv=3, scoring='accuracy')
print("Baseline Accuracy:", baseline_scores.mean())

# 3. Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# 4. GridSearchCV
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1
)

# 5. Train
grid_search.fit(X, y)

# 6. Results
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# 7. Save best model
joblib.dump(grid_search.best_estimator_, "best_model.pkl")

# 8. Total fits calculation
total_fits = 3 * (3 * 3 * 3)  # cv * combinations
print("Total fits performed:", total_fits)