import numpy as np

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    sx = sigmoid(x)
    return sx * (1 - sx)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """Forward pass"""
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        
        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        
        return self.A2
    
    def backward(self, X, y, learning_rate):
        """Backward pass"""
        m = X.shape[0]
        
        # Output layer
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, learning_rate=0.1, epochs=1000):
        """Train the network"""
        for epoch in range(epochs):
            # Forward pass
            self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Print progress
            if epoch % 100 == 0:
                loss = np.mean(np.square(self.A2 - y))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage of MLP
if __name__ == "__main__":
    # Create XOR dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    mlp = SimpleMLP(input_size=2, hidden_size=4, output_size=1)
    mlp.train(X, y)
    
    # Test network
    predictions = mlp.forward(X)
    
    print("\nMLP Results:")
    print("Final Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted: {predictions[i][0]:.4f}, Actual: {y[i][0]}")