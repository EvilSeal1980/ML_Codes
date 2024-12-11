import numpy as np
import pandas as pd

def handle_missing_values(X, strategy='mean'):
    """Replace missing values with mean of column"""
    X_copy = X.copy()
    
    for col in range(X.shape[1]):
        # Get non-nan values in column
        valid_values = X_copy[:, col][~np.isnan(X_copy[:, col])]
        
        if strategy == 'mean':
            fill_value = np.mean(valid_values)
        elif strategy == 'median':
            fill_value = np.median(valid_values)
        else:
            fill_value = np.max(valid_values)
        
        # Replace nan values
        X_copy[:, col][np.isnan(X_copy[:, col])] = fill_value
    
    return X_copy

def encode_categorical(X, categorical_columns):
    """One-hot encode categorical variables"""
    X_copy = X.copy()
    encoded_columns = []
    
    for col in categorical_columns:
        # Get unique values in column
        unique_values = np.unique(X_copy[:, col])
        
        # Create one-hot encoded columns
        for value in unique_values:
            new_col = (X_copy[:, col] == value).astype(int)
            encoded_columns.append(new_col)
    
    # Remove original categorical columns and add encoded ones
    non_cat_cols = [i for i in range(X.shape[1]) if i not in categorical_columns]
    X_numeric = X_copy[:, non_cat_cols]
    
    return np.column_stack([np.array(encoded_columns).T, X_numeric])

def encode_labels(y):
    """Encode categorical labels to numbers"""
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    return np.array([label_map[label] for label in y])

def train_test_split(X, y, test_size=0.2, random_seed=None):
    """Split data into training and test sets"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def scale_features(X_train, X_test):
    """Standardize features by removing mean and scaling to unit variance"""
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Calculate mean and std for each feature using training data
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    
    # Scale both training and test data using training statistics
    X_train_scaled = (X_train - means) / stds
    X_test_scaled = (X_test - means) / stds
    
    return X_train_scaled, X_test_scaled

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('Data.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    print("Original Data:")
    print(X)
    print(y)
    
    # Handle missing values
    X = handle_missing_values(X)
    print("\nAfter handling missing values:")
    print(X)
    
    # Encode categorical variables (assuming first column is categorical)
    X = encode_categorical(X, categorical_columns=[0])
    print("\nAfter encoding categorical variables:")
    print(X)
    
    # Encode labels
    y = encode_labels(y)
    print("\nAfter encoding labels:")
    print(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=1)
    print("\nTraining set size:", len(X_train))
    print("Test set size:", len(X_test))
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("\nAfter scaling features (first few samples):")
    print("Training set:")
    print(X_train_scaled[:3])
    print("Test set:")
    print(X_test_scaled[:3])