# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:29:05 2025

@author: polycarp.mugizi
"""

import pandas as pd

# ----------------------------
# 1. Load the dataset
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Define column names based on UCI documentation
columns = [
    "id",
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",        # <- this column contains missing values
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"               # 2 = Benign, 4 = Malignant
]

# Load the dataset into a DataFrame
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
# The 'bare_nuclei' column has missing entries marked with '?'
# Convert the column to numeric, turning '?' into NaN automatically
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# ----------------------------
# Option A: Drop rows with missing values
# ----------------------------
df_dropped = df.dropna()  # removes any row containing NaN
print("Original shape:", df.shape)   # should be (699, 11)
print("After dropping:", df_dropped.shape)  # should be (683, 11)

# ----------------------------
# Option B: Impute missing values with the median
# ----------------------------
# Calculate the median of 'bare_nuclei' and replace NaN with it
df_median = df.copy()
df_median["bare_nuclei"] = df_median["bare_nuclei"].fillna(df_median["bare_nuclei"].median())

# ----------------------------
# Option C: Impute missing values with the mode
# ----------------------------
# Calculate the most frequent value (mode) and use it for NaN
df_mode = df.copy()
df_mode["bare_nuclei"] = df_mode["bare_nuclei"].fillna(df_mode["bare_nuclei"].mode()[0])

# ----------------------------
# 3. Verify cleaning
# ----------------------------
print("Missing values (drop method):", df_dropped.isna().sum().sum())
print("Missing values (median method):", df_median.isna().sum().sum())
print("Missing values (mode method):", df_mode.isna().sum().sum())
