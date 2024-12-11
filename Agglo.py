import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def calculate_distance_matrix(data):
    """Calculate distance matrix between all points"""
    n = len(data)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            distances[i][j] = euclidean_distance(data[i], data[j])
            distances[j][i] = distances[i][j]
    
    return distances

def find_closest_clusters(distances, active_clusters):
    """Find two closest clusters"""
    min_distance = float('inf')
    min_i, min_j = -1, -1
    
    for i in active_clusters:
        for j in active_clusters:
            if i < j:  # Avoid redundant calculations
                if distances[i][j] < min_distance:
                    min_distance = distances[i][j]
                    min_i, min_j = i, j
    
    return min_i, min_j

def update_distances(distances, i, j, active_clusters, method='single'):
    """Update distances after merging clusters i and j"""
    for k in active_clusters:
        if k != i and k != j:
            if method == 'single':
                # Single linkage: minimum distance
                distances[i][k] = min(distances[i][k], distances[j][k])
                distances[k][i] = distances[i][k]
            elif method == 'complete':
                # Complete linkage: maximum distance
                distances[i][k] = max(distances[i][k], distances[j][k])
                distances[k][i] = distances[i][k]
            elif method == 'average':
                # Average linkage: average distance
                distances[i][k] = (distances[i][k] + distances[j][k]) / 2
                distances[k][i] = distances[i][k]

def agglomerative_clustering(data, n_clusters, linkage='single'):
    """Perform agglomerative clustering"""
    n_samples = len(data)
    
    # Initialize each point as a cluster
    clusters = [{i} for i in range(n_samples)]
    active_clusters = set(range(n_samples))
    
    # Calculate initial distances
    distances = calculate_distance_matrix(data)
    
    # Store merge history for dendrogram
    merge_history = []
    
    # Merge clusters until we have desired number
    while len(active_clusters) > n_clusters:
        # Find closest clusters
        i, j = find_closest_clusters(distances, active_clusters)
        
        # Merge clusters
        clusters[i] = clusters[i].union(clusters[j])
        active_clusters.remove(j)
        
        # Record merge
        merge_history.append((i, j, distances[i][j]))
        
        # Update distances
        update_distances(distances, i, j, active_clusters, linkage)
    
    # Create final cluster labels
    labels = np.zeros(n_samples)
    for idx, cluster in enumerate(clusters):
        if cluster:  # Some clusters might be empty after merges
            for point_idx in cluster:
                labels[point_idx] = idx
    
    return labels, merge_history

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target  # For visualization only

# Apply clustering
n_clusters = 3
labels, merge_history = agglomerative_clustering(X, n_clusters, linkage='single')

# Visualize results using first two features
plt.figure(figsize=(10, 5))

# Original data
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Original Iris Classes')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Clustered data
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('Agglomerative Clustering Results')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.tight_layout()
plt.show()

# Print cluster information
for i in range(n_clusters):
    cluster_points = np.where(labels == i)[0]
    print(f"\nCluster {i}:")
    print(f"Size: {len(cluster_points)}")
    print(f"Points: {cluster_points[:5]}...")  # Show first 5 points