# Day 17 – Hyperparameter Tuning Report

## Objective
The goal of this task was to improve model performance by tuning hyperparameters using GridSearchCV instead of relying on default values.

---

## Approach

### 1. Baseline Model
A Random Forest classifier was trained using default parameters to establish a baseline.

- Baseline Accuracy: ~95.78%

---

### 2. Hyperparameter Tuning

GridSearchCV was used to search for the best combination of parameters.

**Parameter Grid:**
- n_estimators: 50, 100, 200
- max_depth: None, 10, 20
- min_samples_split: 2, 5, 10

**Cross-Validation:**
- 3-Fold Cross Validation

---

### 3. Results

- Best Parameters:
  - n_estimators: 50
  - max_depth: None
  - min_samples_split: 2

- Best Cross-Validation Accuracy: ~96.31%

---

### 4. Performance Comparison

| Model        | Accuracy |
|-------------|----------|
| Baseline     | ~95.78%  |
| Tuned Model  | ~96.31%  |

The improvement was small (~0.5%), indicating that the baseline model was already performing well.

---

### 5. Total Computation

- Total combinations: 3 × 3 × 3 = 27  
- With 3-fold CV: 27 × 3 = 81 total fits

---

### 6. Optimization Logic

GridSearchCV evaluates all possible parameter combinations, which guarantees finding the best configuration within the defined search space. However, this comes at the cost of increased computation time.

For larger datasets or wider search spaces, RandomizedSearchCV is more efficient as it samples a subset of combinations instead of evaluating all.

---

### 7. Model Persistence

The best-performing model was saved using joblib:
