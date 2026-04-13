# Day 8: California Housing Project

from sklearn.datasets import fetch_california_housing
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Load Dataset

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print("Dataset Preview:\n")
print(df.head())

print("\nBasic Statistics:\n")
print(df.describe())


# 2. Define Features & Target

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']


# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)


# 4. Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Model Training

model = LinearRegression()

print("\nTraining the model...")
model.fit(X_train_scaled, y_train)


# 6. Predictions

print("Making predictions...")
predictions = model.predict(X_test_scaled)


# 7. Evaluation

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMean Absolute Error: ${mae * 100000:.2f}")
print(f"R2 Score: {r2:.2f}")


# 8. Residual Plot

residuals = y_test - predictions

plt.scatter(y_test, residuals)
plt.axhline(y=0, linestyle='--')

plt.xlabel("Actual Prices")
plt.ylabel("Residuals (Error)")
plt.title("Residual Plot")

plt.show()