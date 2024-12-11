import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load and process data
data = pd.read_csv('Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Encode categorical variables (assuming first column is categorical)
categorical_columns = [0]  # Specify indices of categorical columns
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(X[:, categorical_columns]).toarray()
X_numeric = np.delete(X, categorical_columns, axis=1)
X = np.hstack((X_categorical, X_numeric))

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print results
print("Processed Data:")
print("X_train (first 3 samples):", X_train_scaled[:3])
print("X_test (first 3 samples):", X_test_scaled[:3])
print("y_train (first 3 labels):", y_train[:3])
print("y_test (first 3 labels):", y_test[:3])
