import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 1. Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2. Apply PCA (reduce to 2D)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 3. Explained variance
variance_retained = sum(pca.explained_variance_ratio_) * 100
print(f"Variance retained by 2 components: {variance_retained:.2f}%")

# 4. Visualization
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=20)
plt.colorbar(label='Digit Class')
plt.title("Digits Dataset (2D PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()