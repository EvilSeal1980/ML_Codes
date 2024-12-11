import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PNN:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def predict(self, X):
        def gaussian(x1, x2):
            return np.exp(-np.sum((x1 - x2) ** 2) / (2 * self.sigma ** 2))
        
        return np.array([
            self.classes[np.argmax([
                np.mean([gaussian(x, sample) for sample in self.X_train[self.y_train == c]])
                for c in self.classes
            ])] for x in X
        ])

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Test different sigma values
for sigma in [0.1, 0.5, 1.0, 2.0]:
    pnn = PNN(sigma=sigma)
    pnn.fit(X_train, y_train)
    accuracy = np.mean(pnn.predict(X_test) == y_test)
    print(f"Sigma: {sigma}, Accuracy: {accuracy:.4f}")
