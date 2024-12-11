import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Salary_Data.csv")
print("Info:")
print(df.info())
print(df.describe())
print(df.head())
print(df.tail())

cor_mat=df.corr()
sns.heatmap(cor_mat)
plt.title("HeatMap")
plt.show()

sns.boxplot(df)
plt.title("BoxPlot")
plt.show()

sns.scatterplot(df)
plt.title('Years of Experience vs Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

df['YearsExperience'].hist()
plt.title('Distribution of Years of Experience')
plt.show()

df['Salary'].hist()
plt.title('Distribution of Salary')
plt.show()