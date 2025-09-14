# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:19:49 2025

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

# Clean "bare_nuclei"
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Replace class numbers with labels
df["class"] = df["class"].map({2: "Benign", 4: "Malignant"})

# Drop ID
attributes = df.drop(columns=["id"])

# Plot boxplots for each attribute grouped by class
plt.figure(figsize=(10, 15))
for i, col in enumerate(attributes.columns[:-1], 1):  # skip class column
    plt.subplot(5, 2, i)
    attributes.boxplot(column=col, by="class", grid=False)
    plt.title(col)
    plt.suptitle("")  # remove automatic title
    plt.xlabel("Tumor Class")
    plt.ylabel("Score (1–10)")

plt.tight_layout() #Adjust layout so titles/labels don’t overlap
plt.show() #Display 
