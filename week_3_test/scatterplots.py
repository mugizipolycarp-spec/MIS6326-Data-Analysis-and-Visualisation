# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:24:08 2025

@author: polycarp.mugizi
"""

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load the dataset
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Column names as described in the UCI dataset
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

# Read CSV file directly from UCI repository
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Data Cleaning
# ----------------------------
# 'bare_nuclei' column has missing values marked as '?'
# Convert them to NaN and cast column to numeric
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Convert class labels from numeric (2,4) to descriptive text
# 2 = Benign, 4 = Malignant
df["class"] = df["class"].map({2: "Benign", 4: "Malignant"})

# ----------------------------
# 3. Scatter Plots for Top Features
# ----------------------------
plt.figure(figsize=(15, 5))  # make the figure wide enough for 3 plots

# --- (a) Uniformity of cell size vs Uniformity of cell shape ---
plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
for label, group in df.groupby("class"):  # plot each class separately
    plt.scatter(group["uniformity_cell_size"], group["uniformity_cell_shape"], 
                label=label, alpha=0.6)  # alpha for transparency
plt.xlabel("Uniformity of Cell Size")
plt.ylabel("Uniformity of Cell Shape")
plt.title("Cell Size vs Cell Shape")
plt.legend()

# --- (b) Uniformity of cell size vs Bare nuclei ---
plt.subplot(1, 3, 2)  # second subplot
for label, group in df.groupby("class"):
    plt.scatter(group["uniformity_cell_size"], group["bare_nuclei"], 
                label=label, alpha=0.6)
plt.xlabel("Uniformity of Cell Size")
plt.ylabel("Bare Nuclei")
plt.title("Cell Size vs Bare Nuclei")

# --- (c) Uniformity of cell shape vs Bare nuclei ---
plt.subplot(1, 3, 3)  # third subplot
for label, group in df.groupby("class"):
    plt.scatter(group["uniformity_cell_shape"], group["bare_nuclei"], 
                label=label, alpha=0.6)
plt.xlabel("Uniformity of Cell Shape")
plt.ylabel("Bare Nuclei")
plt.title("Cell Shape vs Bare Nuclei")

# Adjust layout so titles/labels don’t overlap
plt.tight_layout()

# Display all 3 scatter plots
plt.show()
