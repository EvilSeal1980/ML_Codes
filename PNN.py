import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PNN:
    def __init__(self, sigma=1.0):
        """
        Initialize PNN with smoothing parameter sigma
        sigma: width of the Gaussian window (smoothing parameter)
        """
        self.sigma = sigma
        self.X_train = None
        self.y_train = None
        self.classes = None
    
    def gaussian_kernel(self, x1, x2):
        """
        Calculate Gaussian kernel between two vectors
        K(x1,x2) = exp(-||x1-x2||^2 / (2*sigma^2))
        """
        distance = np.sum((x1 - x2) ** 2)
        return np.exp(-distance / (2 * self.sigma ** 2))
    
    def fit(self, X, y):
        """
        Store training data (PNN doesn't need actual training)
        """
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
    
    def predict_one(self, x):
        """
        Predict class for a single input vector
        """
        # Calculate probabilities for each class
        class_probabilities = []
        
        for c in self.classes:
            # Get all training samples of this class
            class_samples = self.X_train[self.y_train == c]
            
            # Calculate kernel values for all samples of this class
            kernel_values = np.array([self.gaussian_kernel(x, sample) 
                                    for sample in class_samples])
            
            # Average kernel values for this class
            class_probability = np.mean(kernel_values)
            class_probabilities.append(class_probability)
        
        # Return class with highest probability
        return self.classes[np.argmax(class_probabilities)]
    
    def predict(self, X):
        """
        Predict classes for multiple input vectors
        """
        return np.array([self.predict_one(x) for x in X])

# Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different sigma values
sigma_values = [0.1, 0.5, 1.0, 2.0]
results = {}

for sigma in sigma_values:
    # Create and train PNN
    pnn = PNN(sigma=sigma)
    pnn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = pnn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    results[sigma] = accuracy
    
    print(f"\nResults for sigma = {sigma}:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Show some example predictions
    print("\nSample Predictions (first 5 test cases):")
    print("True class -> Predicted class")
    for i in range(5):
        print(f"{y_test[i]} -> {y_pred[i]}")

# Print best sigma
best_sigma = max(results, key=results.get)
print(f"\nBest sigma value: {best_sigma}")
print(f"Best accuracy: {results[best_sigma]:.4f}")