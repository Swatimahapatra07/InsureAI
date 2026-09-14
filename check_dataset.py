import pandas as pd

file_path = "data\processed\irdai_health_products.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print()

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 records:")
print(df.head())