import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Initialize KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels (which data points belong to which cluster)
labels = kmeans.labels_

# Get the centroids of the clusters
centroids = kmeans.cluster_centers_

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidth=3)
plt.title('K-means Clustering Result')
plt.show()





# Simple K-means from scratch
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

def kmeans_simple(X, k=4, max_iters=100):
    # Randomly initialize centroids
    n_samples = X.shape[0]
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check if converged
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids

# Run clustering
labels, centroids = kmeans_simple(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidth=3)
plt.title('K-means Clustering Result')
plt.show()