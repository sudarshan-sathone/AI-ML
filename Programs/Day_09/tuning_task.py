import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Define parameter grid
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 500.0]}

# 5. Grid Search
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')

# 6. Train Grid Search
grid_search.fit(X_train_scaled, y_train)

# 7. Output results
print("Best Alpha:", grid_search.best_params_)
print("Best R2 Score:", grid_search.best_score_)