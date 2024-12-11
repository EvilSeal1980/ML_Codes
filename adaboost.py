from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load and prepare data
iris = load_iris()
X = iris.data[:100]  # Take only first two classes for binary classification
y = iris.target[:100]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AdaBoost with a Decision Stump as the base estimator
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

# Train the model
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display information about weak classifiers
print("\nWeak Classifier Information:")
for i, estimator in enumerate(adaboost.estimators_[:5]):  # Show first 5 stumps
    print(f"\nStump {i+1}:")
    print(f"Feature: {iris.feature_names[adaboost.estimator_features_[i][0]]}")
    print(f"Alpha (weight): {adaboost.estimator_weights_[i]:.4f}")



import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] >= self.threshold] = -1
            
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
    
    def _find_best_threshold(self, X, y, weights):
        """Find best threshold for decision stump"""
        n_samples, n_features = X.shape
        min_error = float('inf')
        best_stump = DecisionStump()
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Try polarity 1
                predictions = np.ones(n_samples)
                predictions[feature_values < threshold] = -1
                error = np.sum(weights * (predictions != y))
                
                if error < min_error:
                    min_error = error
                    best_stump.feature_idx = feature_idx
                    best_stump.threshold = threshold
                    best_stump.polarity = 1
                
                # Try polarity -1
                predictions = np.ones(n_samples)
                predictions[feature_values >= threshold] = -1
                error = np.sum(weights * (predictions != y))
                
                if error < min_error:
                    min_error = error
                    best_stump.feature_idx = feature_idx
                    best_stump.threshold = threshold
                    best_stump.polarity = -1
        
        return best_stump, min_error
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples
        
        # Convert labels to -1, 1
        y = np.where(y == 0, -1, 1)
        
        for _ in range(self.n_estimators):
            # Find best stump for current weights
            stump, error = self._find_best_threshold(X, y, weights)
            
            # Calculate stump weight (alpha)
            eps = 1e-10  # small value to avoid division by zero
            stump.alpha = 0.5 * np.log((1 - error + eps) / (error + eps))
            
            # Update sample weights
            predictions = stump.predict(X)
            weights *= np.exp(-stump.alpha * y * predictions)
            weights /= np.sum(weights)  # normalize
            
            self.stumps.append(stump)
    
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        # Combine weak classifiers
        for stump in self.stumps:
            predictions += stump.alpha * stump.predict(X)
        
        return np.sign(predictions)

# Load and prepare data
iris = load_iris()
X = iris.data[:100]  # Take only first two classes for binary classification
y = iris.target[:100]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost
adaboost = AdaBoost(n_estimators=50)
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)
y_pred = (y_pred + 1) / 2  # Convert back to 0,1

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Print some information about the stumps
print("\nWeak Classifier Information:")
for i, stump in enumerate(adaboost.stumps[:5]):  # Show first 5 stumps
    print(f"\nStump {i+1}:")
    print(f"Feature: {iris.feature_names[stump.feature_idx]}")
    print(f"Threshold: {stump.threshold:.4f}")
    print(f"Alpha (weight): {stump.alpha:.4f}")