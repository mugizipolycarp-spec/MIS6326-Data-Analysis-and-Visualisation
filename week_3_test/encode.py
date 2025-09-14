# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:38:00 2025

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

# Load CSV into a pandas DataFrame
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
# 'bare_nuclei' has missing values represented as "?"
# Convert to numeric; non-numeric values become NaN
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop rows with missing values to clean the dataset
df = df.dropna()

# ----------------------------
# 3. Encode categorical features
# ----------------------------
# 'class' is categorical with numeric codes: 2 = Benign, 4 = Malignant
# Most ML models prefer binary encoding: 0 or 1
# Map values: 2 -> 0 (Benign), 4 -> 1 (Malignant)
df["class"] = df["class"].map({2: 0, 4: 1})

# ----------------------------
# 4. Verify encoding
# ----------------------------
# Check the number of samples in each class after encoding
print(df["class"].value_counts())
