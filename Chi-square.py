from sklearn.datasets import load_iris
from scipy.stats import chi2_contingency
import pandas as pd
# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using only the first two features for simplicity
# Convert features into categories (e.g., above or below mean)
feature_1 = X[:, 0] > X[:, 0].mean()
feature_2 = X[:, 1] > X[:, 1].mean()
# Create a contingency table
contingency_table = pd.crosstab(feature_1, feature_2)
# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")




from sklearn.datasets import load_iris
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using only the first two features for simplicity
# Convert features into categories (e.g., above or below mean)
feature_1 = X[:, 0] > X[:, 0].mean()
feature_2 = X[:, 1] > X[:, 1].mean()
# Create a contingency table
observed = np.zeros((2, 2))
for i in range(len(feature_1)):
    observed[int(feature_1[i]), int(feature_2[i])] += 1
# Calculate row and column sums
row_sums = observed.sum(axis=1)
col_sums = observed.sum(axis=0)
total = observed.sum()
# Calculate expected frequencies
expected = np.outer(row_sums, col_sums) / total
# Compute Chi-Square statistic
chi2 = ((observed - expected) ** 2 / expected).sum()
# Print results
print(f"Observed Frequencies:\n{observed}")
print(f"Expected Frequencies:\n{expected}")
print(f"Chi-Square Statistic: {chi2}")





