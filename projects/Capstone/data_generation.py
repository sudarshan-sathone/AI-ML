import pandas as pd
import numpy as np

np.random.seed(42)

rows = 700  # increase size

data = []

for _ in range(rows):
    skill_diff = np.random.randint(0, 11)
    common_tech = np.random.randint(0, 6)
    availability_match = np.random.randint(0, 2)
    communication_score = np.random.randint(1, 11)
    experience_gap = np.random.randint(0, 6)

    # Base logic
    score = 0
    
    if skill_diff <= 3:
        score += 1
    if common_tech >= 3:
        score += 1
    if availability_match == 1:
        score += 1
    if communication_score >= 6:
        score += 1
    if experience_gap <= 2:
        score += 1

    # Add noise (VERY IMPORTANT)
    noise = np.random.choice([0, 1], p=[0.9, 0.1])

    if score >= 3:
        compatibility = 1 if noise == 0 else 0
    else:
        compatibility = 0 if noise == 0 else 1

    data.append([
        skill_diff,
        common_tech,
        availability_match,
        communication_score,
        experience_gap,
        compatibility
    ])

df = pd.DataFrame(data, columns=[
    'skill_diff',
    'common_tech',
    'availability_match',
    'communication_score',
    'experience_gap',
    'compatibility'
])

df.to_csv('dataset.csv', index=False)

print("Dataset created with shape:", df.shape)