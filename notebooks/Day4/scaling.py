from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Sample data: [Age, Salary]
data = np.array([
    [22, 50000],
    [45, 120000],
    [30, 80000]
])

print("Original Data:\n", data)

# Initialize scaler
scaler = MinMaxScaler()

# Scale data
scaled_data = scaler.fit_transform(data)

print("\nScaled Data:\n", scaled_data)