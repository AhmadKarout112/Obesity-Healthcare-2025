import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_excel('data/Obesity_Dataset.xlsx')

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nColumn Names:")
print(df.columns.tolist())
