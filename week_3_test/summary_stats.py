# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:09:00 2025

@author: polycarp.mugizi
"""

import pandas as pd

# Load dataset (update path if local)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Column names as per dataset description
columns = [
    "id",
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"
]

# Read dataset from url
df = pd.read_csv(url, names=columns)

# Replace missing values "?" with NaN and convert to numeric
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop ID and Class (not tumor attributes)
tumor_attributes = df.drop(columns=["id", "class"])

# Compute summary statistics
summary_stats = pd.DataFrame({
    "Mean": tumor_attributes.mean(),
    "StdDev": tumor_attributes.std(),
    "IQR": tumor_attributes.quantile(0.75) - tumor_attributes.quantile(0.25)
})

print(summary_stats)
