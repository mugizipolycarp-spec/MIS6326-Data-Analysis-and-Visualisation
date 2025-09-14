# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:33:51 2025

@author: polycarp.mugizi
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------
# 1. Load the dataset
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Define column names (from UCI dataset description)
columns = [
    "id",
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",       # contains missing values
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"              # target variable: 2 = Benign, 4 = Malignant
]

# Load the dataset into a pandas DataFrame
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
# Convert "bare_nuclei" to numeric.
# If a value is "?" (string), it's turned into NaN (missing value).
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop rows that contain missing values (you could also impute instead)
df = df.dropna()

# ----------------------------
# 3. Select features to normalize
# ----------------------------
# Drop 'id' (identifier, not useful) and 'class' (target variable).
# Keep only the tumor attributes (continuous features).
features = df.drop(columns=["id", "class"])

# ----------------------------
# 4a. Min–Max Normalization (scales values to range [0,1])
# ----------------------------
# Create a MinMaxScaler object
minmax_scaler = MinMaxScaler()

# Fit the scaler to the features and transform them
features_minmax = minmax_scaler.fit_transform(features)

# Convert back into a DataFrame with the same column names
df_minmax = pd.DataFrame(features_minmax, columns=features.columns)

print("First 5 rows after Min-Max Normalization:")
print(df_minmax.head(), "\n")

# ----------------------------
# 4b. Z-Score Standardization (mean=0, std=1). transforms features to have mean = 0 and std = 1
# ----------------------------
# Create a StandardScaler object
zscore_scaler = StandardScaler()

# Fit the scaler to the features and transform them
features_zscore = zscore_scaler.fit_transform(features)

# Convert back into a DataFrame with the same column names
df_zscore = pd.DataFrame(features_zscore, columns=features.columns)

print("First 5 rows after Z-Score Standardization:")
print(df_zscore.head())
