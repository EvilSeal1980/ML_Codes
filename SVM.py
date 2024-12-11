import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

# 1. Linear Kernel
# K(x,y) = x^T * y
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
results['Linear'] = accuracy_score(y_test, y_pred_linear)

# 2. Polynomial Kernel
# K(x,y) = (gamma * x^T * y + coef0)^degree
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train_scaled, y_train)
y_pred_poly = svm_poly.predict(X_test_scaled)
results['Polynomial'] = accuracy_score(y_test, y_pred_poly)

# 3. RBF (Radial Basis Function) Kernel
# K(x,y) = exp(-gamma * ||x-y||^2)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
results['RBF'] = accuracy_score(y_test, y_pred_rbf)

# 4. Sigmoid Kernel
# K(x,y) = tanh(gamma * x^T * y + coef0)
svm_sigmoid = SVC(kernel='sigmoid')
svm_sigmoid.fit(X_train_scaled, y_train)
y_pred_sigmoid = svm_sigmoid.predict(X_test_scaled)
results['Sigmoid'] = accuracy_score(y_test, y_pred_sigmoid)

# Print results
print("SVM Results with Different Kernels:")
print("-" * 40)
for kernel, accuracy in results.items():
    print(f"{kernel} Kernel Accuracy: {accuracy:.4f}")
print("\nDetailed Report for Best Kernel:")
best_kernel = max(results, key=results.get)
print(f"\nBest Kernel: {best_kernel}")
if best_kernel == 'Linear':
    print(classification_report(y_test, y_pred_linear))
elif best_kernel == 'Polynomial':
    print(classification_report(y_test, y_pred_poly))
elif best_kernel == 'RBF':
    print(classification_report(y_test, y_pred_rbf))
else:
    print(classification_report(y_test, y_pred_sigmoid))