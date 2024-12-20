import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate a synthetic dataset with three clusters
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Visualize the data points
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Synthetic Data with Three Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Apply K-Means clustering with K=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get the cluster assignments for each data point
cluster_labels = kmeans.labels_

# Get the cluster centroids
cluster_centers = kmeans.cluster_centers_

# Visualize the clustered data
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title("K-Means Clustering Results (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()




# Create a simulated dataset with 3 features
np.random.seed(0)
n_samples = 100
feature1 = np.random.rand(n_samples)
feature2 = 2 * feature1 + np.random.rand(n_samples)
feature3 = 0.5 * feature1 - 2 * feature2 + np.random.rand(n_samples)

# Combine the features into a single dataset
X = np.column_stack((feature1, feature2, feature3))

# Apply PCA to reduce the dimensionality to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the original and PCA-reduced data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original Data")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA-Reduced Data")

plt.tight_layout()
plt.show()

# Print the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)



# Generate some example data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1)

# Simple Hypothesis Space (e.g., linear model)
def simple_hypothesis(X, theta0, theta1):
    return theta0 + theta1 * X

# Complex Hypothesis Space (e.g., high-degree polynomial)
def complex_hypothesis(X, theta):
    # Using a 9th-degree polynomial
    return np.sum(theta[i] * X**i for i in range(len(theta)))

# Fit the models
theta_simple = np.polyfit(X.flatten(), y.flatten(), 1)
theta_complex = np.polyfit(X.flatten(), y.flatten(), 9)

# Generate predictions
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_simple = simple_hypothesis(X_test, theta_simple[1], theta_simple[0])
y_complex = complex_hypothesis(X_test, theta_complex)

# Plot the data and models
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_simple, label='Simple Hypothesis Space (Linear)')
plt.plot(X_test, y_complex, label='Complex Hypothesis Space (9th-degree Polynomial)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Simple vs. Complex Hypothesis Space')
plt.show()



# Generate a synthetic dataset for binary classification
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Simple Hypothesis Space (Linear)
model_simple = LogisticRegression()
model_simple.fit(X, y)

# Complex Hypothesis Space (Polynomial)
model_complex = make_pipeline(PolynomialFeatures(10), LogisticRegression())
model_complex.fit(X, y)

# Generate a range of X values for plotting
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Predictions
y_pred_simple = model_simple.predict_proba(X_plot)[:, 1]
y_pred_complex = model_complex.predict_proba(X_plot)[:, 1]

# Plot the data and decision boundaries
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_plot, y_pred_simple, color='green', label='Simple Hypothesis (Linear)')
plt.plot(X_plot, y_pred_complex, color='red', label='Complex Hypothesis (Polynomial)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Simple vs. Complex Hypothesis Space for Classification')
plt.show()

