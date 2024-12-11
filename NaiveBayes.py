import numpy as np
import pandas as pd

# Load and prepare data
data = pd.read_csv('Social_Network_Ads.csv')
X = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

# Split data into train and test (70-30 split)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

def naive_bayes_train(X_train, y_train):
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    
    # Calculate prior probabilities
    priors = {}
    for c in classes:
        priors[c] = np.mean(y_train == c)
    
    # Calculate mean and variance for each feature in each class
    means = {}
    variances = {}
    
    for c in classes:
        # Get all samples of this class
        X_c = X_train[y_train == c]
        
        # Calculate mean and variance for each feature
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.var(X_c, axis=0)
    
    return priors, means, variances

def gaussian_probability(x, mean, variance):
    """Calculate probability using Gaussian distribution"""
    exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
    return exponent / np.sqrt(2 * np.pi * variance)

def naive_bayes_predict(X, priors, means, variances):
    predictions = []
    
    for x in X:
        # Calculate probability for each class
        probabilities = {}
        
        for c in priors.keys():
            # Start with prior probability
            prob = priors[c]
            
            # Multiply by likelihood of each feature
            for i in range(len(x)):
                prob *= gaussian_probability(x[i], means[c][i], variances[c][i])
            
            probabilities[c] = prob
        
        # Select class with highest probability
        predictions.append(max(probabilities, key=probabilities.get))
    
    return np.array(predictions)

# Train the model
priors, means, variances = naive_bayes_train(X_train, y_train)

# Make predictions
y_pred = naive_bayes_predict(X_test, priors, means, variances)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Print detailed results
print("\nDetailed Results:")
print("Prior Probabilities:", priors)
print("\nMean values for each class:")
for c in priors.keys():
    print(f"Class {c}:", means[c])

# Show some predictions
print("\nSample Predictions (First 10 test cases):")
print("Age  Salary  Predicted  Actual")
print("-" * 35)
for i in range(10):
    print(f"{X_test[i][0]:3.0f}  {X_test[i][1]:7.0f}  {y_pred[i]:9d}  {y_test[i]:6d}")