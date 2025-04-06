import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import os
print(os.getcwd())

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#check info of the dataset
train.info()
train.describe()

#check for null values
missing_values = train.isnull().sum()
missing_values[missing_values > 0]


# Separate numerical and categorical columns
numerical_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).columns

# Basic statistics for numerical features
print("\nNumerical Features Summary:")
train[numerical_features].describe()

# Correlation analysis
plt.figure(figsize=(12, 8))
correlation_matrix = train[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Distribution of target variable (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.show()

# Top 5 most correlated features with SalePrice
top_corr = correlation_matrix['SalePrice'].sort_values(ascending=False)[:6]
print("\nTop 5 correlated features with SalePrice:")
print(top_corr)

# Boxplots for categorical features vs SalePrice
for cat in categorical_features[:3]:  # Showing first 3 categorical features
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=cat, y='SalePrice', data=train)
    plt.xticks(rotation=45)
    plt.title(f'SalePrice vs {cat}')
    plt.show()

# Scatter plots for top 4 correlated numerical features
top_corr_features = correlation_matrix['SalePrice'].sort_values(ascending=False)[1:5].index
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feature in enumerate(top_corr_features):
    row = idx // 2
    col = idx % 2
    sns.scatterplot(data=train, x=feature, y='SalePrice', ax=axes[row, col])
    axes[row, col].set_title(f'SalePrice vs {feature}')
plt.tight_layout()
plt.show()

# Missing values percentage
missing_percentage = (train.isnull().sum() / len(train)) * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
print("\nMissing values percentage:")
print(missing_percentage)

