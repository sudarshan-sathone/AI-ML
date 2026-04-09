from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Dataset
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Score': [35,40,55,60,68,72,81,88,92,95]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Score']

# Split again (important)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Predictions for Test Set:", predictions)
print("Actual Scores:", y_test.values)


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared Score: {r2:.2f}")

print("Prediction for 11 hours:", model.predict(pd.DataFrame({'Hours': [11]})))
