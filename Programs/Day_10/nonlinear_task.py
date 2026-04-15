import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# 1. Generate synthetic data
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# 2. Polynomial Regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Predictions for smooth curve
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_poly_pred = poly_model.predict(X_new_poly)

# 3. Decision Tree Models
depths = [2, 5, 20]
tree_predictions = {}

for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X, y)
    y_tree_pred = tree.predict(X_new)
    tree_predictions[depth] = y_tree_pred

# 4. R² Scores
poly_r2 = r2_score(y, poly_model.predict(X_poly))

tree_r2_scores = {}
for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X, y)
    tree_r2_scores[depth] = r2_score(y, tree.predict(X))

print("Polynomial R2:", poly_r2)
for depth in depths:
    print(f"Tree depth {depth} R2:", tree_r2_scores[depth])

# 5. Plot EVERYTHING
plt.figure(figsize=(10,6))

# Original data
plt.scatter(X, y, color='blue', alpha=0.5, label='Data')

# Polynomial curve
plt.plot(X_new, y_poly_pred, color='red', linewidth=2, label='Polynomial Curve')

# Tree predictions
colors = ['green', 'orange', 'purple']
for i, depth in enumerate(depths):
    plt.plot(X_new, tree_predictions[depth],
             color=colors[i],
             linestyle='--',
             label=f'Tree depth={depth}')

plt.legend()
plt.title("Polynomial vs Decision Tree Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)

# Save for submission
plt.savefig("model_comparison.png")
plt.show()