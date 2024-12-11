from sklearn import datasets
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

# Load the iris dataset from sklearn
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features (Sepal Length and Sepal Width)
# Transpose data (features, samples) as required by scikit-fuzzy
data_transposed = np.transpose(X)
n_clusters = 3
# Perform Fuzzy C-Means clustering
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data_transposed, n_clusters, 2, error=0.005, maxiter=1000, init=None
)
# Assign labels based on the maximum membership value
fcm_labels = np.argmax(u, axis=0)
# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=fcm_labels, cmap='viridis')
plt.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='x', label='Centroids')  # Centroids
plt.title('Fuzzy C-means Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()






import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2] 

# Initialize parameters
n_clusters = 3
max_iter = 1000
m = 2  # Fuzziness parameter
error_threshold = 0.005
n_samples = X.shape[0]

# Randomly initialize cluster centers (just for simplicity)
centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

# Initialize fuzzy membership matrix
u = np.random.rand(n_clusters, n_samples)
u = u / np.sum(u, axis=0)  # Normalize memberships to sum to 1

# Function to update fuzzy memberships
def update_membership(X, centroids, m):
    dist = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    dist = np.fmax(dist, np.finfo(float).eps)  # Prevent division by zero
    u_new = 1 / dist ** (2 / (m - 1))
    u_new = u_new / np.sum(u_new, axis=0)  # Normalize memberships
    return u_new

# Function to update centroids
def update_centroids(X, u, m):
    um = u ** m
    centroids = np.dot(um, X) / np.sum(um, axis=1)[:, np.newaxis]
    return centroids

# Iterate and update until convergence
for i in range(max_iter):
    # Update membership matrix
    u_new = update_membership(X.T, centroids, m)
    
    # Check for convergence
    if np.linalg.norm(u_new - u) < error_threshold:
        print(f'Converged at iteration {i}')
        break
    
    u = u_new
    
    # Update centroids
    centroids = update_centroids(X, u, m)

# Assign labels based on maximum membership
fcm_labels = np.argmax(u, axis=0)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=fcm_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
plt.title('Fuzzy C-means Clustering (Scratch Implementation)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()






