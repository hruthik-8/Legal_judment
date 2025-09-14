import pandas as pd

# Load the dataset
df = pd.read_csv('sample_criminal.csv')

# 1. Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# 2. Display dataset information
print("\nDataset Information:")
print(df.info())

# 3. Display basic statistics for numerical columns
print("\nBasic Statistics:")
print(df.describe())

# 4. Display unique values in categorical columns
print("\nUnique Charges:")
print(df['charges'].unique())

print("\nUnique Law Articles:")
print(df['law_articles'].unique())

# 5. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Display the shape of the dataset
print(f"\nDataset shape: {df.shape}")