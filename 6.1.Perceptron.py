import numpy as np
from sklearn.linear_model import Perceptron

# Create simple AND gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate (0, 0, 0, 1)

# Initialize the Perceptron model
perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

# Train the model
perceptron.fit(X, y)

# Make predictions
predictions = perceptron.predict(X)

print("Perceptron Results:")
print("Weights:", perceptron.coef_)
print("Bias:", perceptron.intercept_)
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {predictions[i]}, Actual: {y[i]}")





import numpy as np

# Simple Perceptron
def perceptron_train(X, y, learning_rate=0.1, epochs=100):
    """
    Train a simple perceptron
    X: input features (n_samples, n_features)
    y: target values (-1 or 1)
    """
    # Initialize weights and bias
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(epochs):
        for i in range(len(X)):
            # Calculate prediction
            prediction = 1 if np.dot(X[i], weights) + bias > 0 else -1
            
            # Update weights and bias if prediction is wrong
            if prediction != y[i]:
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]
    
    return weights, bias

def perceptron_predict(X, weights, bias):
    """Make predictions using trained perceptron"""
    return np.where(np.dot(X, weights) + bias > 0, 1, -1)

# Example usage of Perceptron
if __name__ == "__main__":
    # Create simple AND gate dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([-1, -1, -1, 1])  # AND gate
    
    # Train perceptron
    weights, bias = perceptron_train(X, y)
    
    # Make predictions
    predictions = perceptron_predict(X, weights, bias)
    
    print("Perceptron Results:")
    print("Weights:", weights)
    print("Bias:", bias)
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted: {predictions[i]}, Actual: {y[i]}")