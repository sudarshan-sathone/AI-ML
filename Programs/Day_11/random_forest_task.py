from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load Data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 2. Initialize Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# 3. Train Model
rf_model.fit(X_train, y_train)

# 4. Predict
predictions = rf_model.predict(X_test)

# 5. Evaluate
score = r2_score(y_test, predictions)
print(f"Random Forest R2 Score: {score:.4f}")

# 6. Feature Importance Plot
importances = rf_model.feature_importances_
feature_names = data.feature_names

feat_importances = pd.Series(importances, index=feature_names)

feat_importances.nlargest(5).plot(kind='barh')
plt.title("Top 5 Most Important Features")
plt.savefig("feature_importance.png")  # saves image
plt.show()