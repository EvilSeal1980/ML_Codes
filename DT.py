import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

def entropy(y):
    """Calculate entropy"""
    classes = np.unique(y)
    entropy = 0
    for cls in classes:
        p = len(y[y == cls]) / len(y)
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def gini(y):
    """Calculate gini index"""
    classes = np.unique(y)
    gini = 1
    for cls in classes:
        p = len(y[y == cls]) / len(y)
        gini -= p ** 2
    return gini

def information_gain(y_parent, y_left, y_right):
    """Calculate information gain"""
    p = len(y_left) / len(y_parent)
    return entropy(y_parent) - (p * entropy(y_left) + (1-p) * entropy(y_right))

def gain_ratio(y_parent, y_left, y_right):
    """Calculate gain ratio"""
    ig = information_gain(y_parent, y_left, y_right)
    p = len(y_left) / len(y_parent)
    split_info = -p * np.log2(p) - (1-p) * np.log2(1-p)
    return 0 if split_info == 0 else ig / split_info

def find_split(X, y, criterion='gini'):
    """Find best split using specified criterion"""
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            y_left = y[left_mask]
            y_right = y[~left_mask]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            if criterion == 'entropy':
                gain = information_gain(y, y_left, y_right)
            elif criterion == 'gini':
                gain = gini(y) - (len(y_left)/len(y) * gini(y_left) + 
                                len(y_right)/len(y) * gini(y_right))
            elif criterion == 'gain_ratio':
                gain = gain_ratio(y, y_left, y_right)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                
    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=3, criterion='gini'):
    """Build decision tree recursively"""
    # Create leaf node if pure or max depth reached
    if depth == max_depth or len(np.unique(y)) == 1:
        return {'type': 'leaf', 'class': np.argmax(np.bincount(y))}
    
    feature, threshold = find_split(X, y, criterion)
    
    if feature is None:  # No valid split found
        return {'type': 'leaf', 'class': np.argmax(np.bincount(y))}
    
    left_mask = X[:, feature] <= threshold
    
    return {
        'type': 'node',
        'feature': feature,
        'threshold': threshold,
        'left': build_tree(X[left_mask], y[left_mask], depth+1, max_depth, criterion),
        'right': build_tree(X[~left_mask], y[~left_mask], depth+1, max_depth, criterion)
    }

def predict_one(x, tree):
    """Make prediction for one sample"""
    if tree['type'] == 'leaf':
        return tree['class']
    
    if x[tree['feature']] <= tree['threshold']:
        return predict_one(x, tree['left'])
    return predict_one(x, tree['right'])

def predict(X, tree):
    """Make predictions for multiple samples"""
    return np.array([predict_one(x, tree) for x in X])

# Train trees with different criteria
criteria = ['entropy', 'gini', 'gain_ratio']
trees = {}
accuracies = {}

for criterion in criteria:
    # Build tree
    trees[criterion] = build_tree(X, y, criterion=criterion)
    
    # Make predictions
    y_pred = predict(X, trees[criterion])
    
    # Calculate accuracy
    accuracies[criterion] = np.mean(y_pred == y)
    print(f"{criterion} accuracy: {accuracies[criterion]:.3f}")

# Simple visualization helper
def print_tree(tree, depth=0):
    """Print tree structure"""
    indent = "  " * depth
    if tree['type'] == 'leaf':
        print(f"{indent}Leaf: class {tree['class']}")
    else:
        print(f"{indent}Node: feature {tree['feature']} <= {tree['threshold']:.2f}")
        print_tree(tree['left'], depth + 1)
        print_tree(tree['right'], depth + 1)

# Print tree structures
for criterion in criteria:
    print(f"\nTree structure using {criterion}:")
    print_tree(trees[criterion])