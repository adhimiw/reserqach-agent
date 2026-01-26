"""
Create test dataset for autonomous coding demonstration
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Generate sample dataset
data = {
    'customer_id': range(1, 101),
    'age': np.random.normal(45, 15, 100).astype(int),
    'income': np.random.normal(50000, 15000, 100),
    'spending_score': np.random.randint(1, 100, 100),
    'membership_years': np.random.randint(1, 10, 100),
    'purchase_frequency': np.random.randint(1, 50, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'satisfaction': np.random.normal(7, 2, 100).round(1),
    'churn': np.random.choice([0, 1], 100, p=[0.7, 0.3])
}

df = pd.DataFrame(data)
df.to_csv('test_customers.csv', index=False)

print(f"✓ Test dataset created: test_customers.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nSummary:")
print(df.describe())
