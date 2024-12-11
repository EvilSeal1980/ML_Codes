from sklearn.datasets import load_iris
from scipy.stats import ttest_ind
# Load the Iris dataset
iris = load_iris()
X = iris.data  # Using all features for simplicity
# Separate two groups: e.g., samples belonging to class 0 and class 1
group_1 = X[iris.target == 0, 0]  # First feature of class 0
group_2 = X[iris.target == 1, 0]  # First feature of class 1
# Perform an independent t-test
t_stat, p_value = ttest_ind(group_1, group_2)
print(f"T-Statistic: {t_stat}")
print(f"P-Value: {p_value}")



from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
X = iris.data
# Separate two groups: e.g., samples belonging to class 0 and class 1
group_1 = X[iris.target == 0, 0]  # First feature of class 0
group_2 = X[iris.target == 1, 0]  # First feature of class 1
# Calculate means
mean_1 = np.mean(group_1)
mean_2 = np.mean(group_2)
# Calculate standard deviations
std_1 = np.std(group_1, ddof=1)  # Use ddof=1 for sample standard deviation
std_2 = np.std(group_2, ddof=1)
# Calculate sample sizes
n1 = len(group_1)
n2 = len(group_2)
# Calculate t-statistic
t_stat = (mean_1 - mean_2) / np.sqrt((std_1**2 / n1) + (std_2**2 / n2))
# Calculate degrees of freedom
df = ((std_1**2 / n1) + (std_2**2 / n2))**2 / (
    ((std_1**2 / n1)**2 / (n1 - 1)) + ((std_2**2 / n2)**2 / (n2 - 1))
)
print(f"T-Statistic: {t_stat}")
print(f"Degrees of Freedom: {df}")

