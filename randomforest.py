from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display predictions and actual values
print("\nSample Predictions (first 5 test cases):")
print("True class -> Predicted class")
for i in range(5):
    print(f"{y_test[i]} -> {y_pred[i]}")








import numpy as np
from sklearn.datasets import make_classification
from collections import Counter

# Generate sample data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
    
    def split_data(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        return (X[left_mask], y[left_mask], X[~left_mask], y[~left_mask])
    
    def find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            threshold = np.median(X[:, feature])
            X_left, y_left, X_right, y_right = self.split_data(X, y, feature, threshold)
            
            if len(y_left) > 0 and len(y_right) > 0:
                gain = self.calculate_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def calculate_gain(self, y_parent, y_left, y_right):
        # Simple gain calculation
        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)
        
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        return gini(y_parent) - (n_left/n * gini(y_left) + n_right/n * gini(y_right))
    
    def build_tree(self, X, y, depth=0):
        n_samples_per_class = np.bincount(y)
        predicted_class = np.argmax(n_samples_per_class)
        
        node = {'predicted_class': predicted_class}
        
        if depth < self.max_depth:
            feature, threshold = self.find_best_split(X, y)
            if feature is not None:
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    node['feature'] = feature
                    node['threshold'] = threshold
                    node['left'] = self.build_tree(X_left, y_left, depth + 1)
                    node['right'] = self.build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
    
    def predict_one(self, x, node):
        if 'feature' not in node:
            return node['predicted_class']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        return self.predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

def random_forest(X, y, n_trees=5, max_depth=3):
    trees = []
    n_samples = X.shape[0]
    
    for _ in range(n_trees):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        # Train tree
        tree = SimpleDecisionTree(max_depth=max_depth)
        tree.fit(X_bootstrap, y_bootstrap)
        trees.append(tree)
    
    return trees

def predict_forest(trees, X):
    predictions = np.array([tree.predict(X) for tree in trees])
    # Take majority vote
    return np.array([Counter(predictions[:, i]).most_common(1)[0][0] 
                    for i in range(X.shape[0])])

# Train forest
trees = random_forest(X, y)

# Make predictions
y_pred = predict_forest(trees, X)

# Calculate accuracy
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")
