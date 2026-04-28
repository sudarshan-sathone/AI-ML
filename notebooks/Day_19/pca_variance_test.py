import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data

# Full PCA
pca_full = PCA().fit(X)

# Cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components for 95%
n_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
print(f"Components needed for 95% variance: {n_95}")

# Plot
plt.plot(cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.show()