import numpy as np
import pandas as pd

# Adjusting the maximum feature index for demonstration
max_feature_index_demo = 1000

# Initialize a DataFrame with zeros for all features
df_columns = ['label'] + [f'feature_{i}' for i in range(1, max_feature_index_demo + 1)]
df_demo = pd.DataFrame(columns=df_columns)
df_demo = df_demo.fillna(0)  # Fill NaN values with 0

# Read the file and populate the DataFrame
with open("..\data\Day0.svm", 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        label = int(parts[0])
        feature_values = {f'feature_{int(f.split(":")[0])}': float(f.split(":")[1]) for f in parts[1:]}

        # Create a row with all zeros and update with actual feature values
        row = np.zeros(max_feature_index_demo + 1)
        row[0] = label  # Set the label
        for feature, value in feature_values.items():
            if feature in df_demo.columns:
                row[df_columns.index(feature)] = value

        # Append the row to the DataFrame
        df_demo.loc[len(df_demo)] = row

        # Limit the processing for demonstration
        if len(df_demo) >= 100:
            break

print(df_demo.head())
