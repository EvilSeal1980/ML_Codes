import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Salary_Data.csv')  # Replace with the actual filename

# Inspect data
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Plot histograms for each column
df['YearsExperience'].hist()
plt.title('Distribution of Years of Experience')
plt.xlabel('YearsExperience')
plt.ylabel('Frequency')
plt.show()

df['Salary'].hist()
plt.title('Distribution of Salary')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# Scatter plot to explore the relationship between YearsExperience and Salary
sns.scatterplot(x='YearsExperience', y='Salary', data=df)
plt.title('Years of Experience vs Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
