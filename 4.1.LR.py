import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def rotate_line(m, b, theta_degrees):
    """
    Rotate line by given angle
    Original line: y = mx + b
    Returns new slope and intercept
    """
    # Convert current slope to angle (in radians)
    current_angle = np.arctan(m)
    
    # Add rotation angle (convert degrees to radians)
    new_angle = current_angle + np.radians(theta_degrees)
    
    # Calculate new slope
    new_m = np.tan(new_angle)
    
    # Keep the same intercept for simplicity
    return new_m, b

# Load and prepare data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Years of experience
y = dataset.iloc[:, -1].values   # Salary

# Initialize and fit linear regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Coefficients of the line
m = regressor.coef_[0]
b = regressor.intercept_

print(f"\nOriginal Line Equation: y = {m:.2f}x + {b:.2f}")

# Custom prediction
custom_experience = float(input("\nEnter years of experience to predict salary: "))
predicted_salary = regressor.predict([[custom_experience]])
print(f"Predicted salary for {custom_experience} years: ${predicted_salary[0]:.2f}")

# Rotate line by 45 degrees and show new equation
rotation_angle = 45
new_m, new_b = rotate_line(m, b, rotation_angle)
print(f"\nAfter {rotation_angle}° rotation:")
print(f"New Line Equation: y = {new_m:.2f}x + {new_b:.2f}")

# Visualize both lines
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data Points')

# Plot original line
x_range = np.array([X.min(), X.max()])
plt.plot(x_range, m * x_range + b, color='blue', label='Original Line')

# Plot rotated line
plt.plot(x_range, new_m * x_range + new_b, color='green', label='Rotated Line')

plt.title('Salary vs Experience (Original and Rotated Lines)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_coefficients(X, y):
    """Calculate slope (m) and intercept (b)"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return coefficients[0], coefficients[1]  # intercept (b), slope (m)

def predict_salary(experience, b, m):
    """Predict salary for given years of experience"""
    return m * experience + b

def rotate_line(m, b, theta_degrees):
    """
    Rotate line by given angle
    Original line: y = mx + b
    Returns new slope and intercept
    """
    # Convert current slope to angle (in radians)
    current_angle = np.arctan(m)
    
    # Add rotation angle (convert degrees to radians)
    new_angle = current_angle + np.radians(theta_degrees)
    
    # Calculate new slope
    new_m = np.tan(new_angle)
    
    # Keep the same intercept for simplicity
    # (in reality, intercept would also change based on rotation point)
    return new_m, b

# Load and prepare data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Calculate coefficients
b, m = calculate_coefficients(X, y)

print(f"\nOriginal Line Equation: y = {m:.2f}x + {b:.2f}")

# Custom prediction
custom_experience = float(input("\nEnter years of experience to predict salary: "))
predicted_salary = predict_salary(custom_experience, b, m)
print(f"Predicted salary for {custom_experience} years: ${predicted_salary:.2f}")

# Rotate line by 45 degrees and show new equation
rotation_angle = 45
new_m, new_b = rotate_line(m, b, rotation_angle)
print(f"\nAfter {rotation_angle}° rotation:")
print(f"New Line Equation: y = {new_m:.2f}x + {new_b:.2f}")

# Visualize both lines
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data Points')

# Plot original line
x_range = np.array([X.min(), X.max()])
plt.plot(x_range, m * x_range + b, color='blue', label='Original Line')

# Plot rotated line
plt.plot(x_range, new_m * x_range + new_b, color='green', label='Rotated Line')

plt.title('Salary vs Experience (Original and Rotated Lines)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()