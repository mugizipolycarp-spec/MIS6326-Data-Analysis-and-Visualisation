# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:42:23 2025

@author: polycarp.mugizi
"""

import pandas as pd

# ----------------------------
# 1. Load the dataset
# ----------------------------
# URL of the Breast Cancer Wisconsin (Original) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Define column names as per UCI dataset documentation
columns = [
    "id",                         # unique sample ID, not used for modeling
    "clump_thickness",            # tumor attribute (1-10)
    "uniformity_cell_size",       # tumor attribute (1-10)
    "uniformity_cell_shape",      # tumor attribute (1-10)
    "marginal_adhesion",          # tumor attribute (1-10)
    "single_epithelial_cell_size",# tumor attribute (1-10)
    "bare_nuclei",                # tumor attribute (1-10), contains missing values
    "bland_chromatin",            # tumor attribute (1-10)
    "normal_nucleoli",            # tumor attribute (1-10)
    "mitoses",                    # tumor attribute (1-10)
    "class"                       # target: 2 = Benign, 4 = Malignant
]

# Load CSV into pandas DataFrame
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
# Convert "bare_nuclei" to numeric; "?" becomes NaN
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop rows with missing values to ensure clean dataset
df = df.dropna()

# ----------------------------
# 3. Encode target variable
# ----------------------------
# Map class values: 2 (Benign) -> 0, 4 (Malignant) -> 1
df["class"] = df["class"].map({2: 0, 4: 1})

# ----------------------------
# 4. Generate feature ratios
# ----------------------------
# Ratios can highlight relationships between attributes that raw values may not capture

# Add small epsilon (1e-6) to avoid division by zero
epsilon = 1e-6

# Ratio 1: Uniformity of cell size / Uniformity of cell shape
df["size_shape_ratio"] = df["uniformity_cell_size"] / (df["uniformity_cell_shape"] + epsilon)

# Ratio 2: Clump thickness / Uniformity of cell size
df["thickness_size_ratio"] = df["clump_thickness"] / (df["uniformity_cell_size"] + epsilon)

# Ratio 3: Bare nuclei / Normal nucleoli
df["bare_nuclei_nucleoli_ratio"] = df["bare_nuclei"] / (df["normal_nucleoli"] + epsilon)

# ----------------------------
# 5. Inspect new features
# ----------------------------
# Display first few rows of the newly created ratio features
print(df[["size_shape_ratio","thickness_size_ratio","bare_nuclei_nucleoli_ratio"]].head())
