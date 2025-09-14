# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:15:41 2025

@author: polycarp.mugizi
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

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

df = pd.read_csv(url, names=columns)

# Handle missing values in bare_nuclei
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop ID and class (not attributes)
tumor_attributes = df.drop(columns=["id", "class"])

# Plot histograms for each attribute
tumor_attributes.hist(bins=10, figsize=(15, 10), edgecolor="black", grid=True)
plt.suptitle("Histograms of Tumor Attributes", fontsize=12)
plt.tight_layout()
plt.show()
