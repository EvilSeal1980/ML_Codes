import numpy as np

# Step 1: Create simple training data (XOR problem)
# XOR means if inputs are different -> output=1, if inputs are same -> output=0
X = np.array([[0,0],  # input 1
              [0,1],  # input 2
              [1,0],  # input 3
              [1,1]]) # input 4
y = np.array([[0],    # output for [0,0]
              [1],    # output for [0,1]
              [1],    # output for [1,0]
              [0]])   # output for [1,1]

# Step 2: Define activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Step 3: Initialize random weights and biases
np.random.seed(42)  # for reproducible results
input_size = 2   # we have 2 inputs (X has 2 columns)
hidden_size = 4  # we'll use 4 neurons in hidden layer
output_size = 1  # we want 1 output

# Weights and biases for hidden layer
W1 = np.random.uniform(size=(input_size, hidden_size))  # 2x4 matrix
b1 = np.zeros((1, hidden_size))                        # 1x4 matrix

# Weights and biases for output layer
W2 = np.random.uniform(size=(hidden_size, output_size)) # 4x1 matrix
b2 = np.zeros((1, output_size))                        # 1x1 matrix

# Step 4: Training loop
learning_rate = 0.1
for epoch in range(10000):  # number of training iterations
    
    # Forward propagation
    # Hidden layer
    Z1 = np.dot(X, W1) + b1  # multiply inputs with weights and add bias
    A1 = sigmoid(Z1)         # apply activation function
    
    # Output layer
    Z2 = np.dot(A1, W2) + b2 # multiply hidden layer with weights and add bias
    A2 = sigmoid(Z2)         # apply activation function (this is our prediction)
    
    # Calculate error
    error = A2 - y
    
    # Backward propagation
    # Output layer
    dZ2 = error * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden layer
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    
    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Step 5: Test the network
print("\nTesting the network:")
print("Input  ->  Predicted Output  ->  Expected Output")
print("-" * 45)
predictions = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
for i in range(len(X)):
    print(f"{X[i]}  ->  {predictions[i][0]:.4f}  ->  {y[i][0]}")