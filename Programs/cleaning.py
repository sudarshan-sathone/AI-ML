import pandas as pd

# Load the dataset
df = pd.read_csv("data.csv")

# 1. IDENTIFY missing values
print("Missing Values:\n", df.isnull().sum())

# 2. Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 3. Fill missing Score with 0
df['Score'] = df['Score'].fillna(0)

# Print cleaned data
print("\nCleaned Data:\n", df)