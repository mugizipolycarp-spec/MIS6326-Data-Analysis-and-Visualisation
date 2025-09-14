# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 08:04:51 2025

@author: polycarp.mugizi
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load the dataset
# ----------------------------
# URL for the Breast Cancer Wisconsin (Original) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Column names as per UCI repository
columns = [
    "id",                         # unique identifier, not used for modeling
    "clump_thickness",            # tumor attribute (ordinal, 1-10)
    "uniformity_cell_size",       # tumor attribute (ordinal, 1-10)
    "uniformity_cell_shape",      # tumor attribute (ordinal, 1-10)
    "marginal_adhesion",          # tumor attribute (ordinal, 1-10)
    "single_epithelial_cell_size",# tumor attribute (ordinal, 1-10)
    "bare_nuclei",                # tumor attribute (ordinal, 1-10), contains missing values
    "bland_chromatin",            # tumor attribute (ordinal, 1-10)
    "normal_nucleoli",            # tumor attribute (ordinal, 1-10)
    "mitoses",                    # tumor attribute (ordinal, 1-10)
    "class"                       # target variable: 2 = Benign, 4 = Malignant
]

# Read CSV into pandas DataFrame
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
# Convert "?" in 'bare_nuclei' column to NaN, then drop rows with NaN
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")
df = df.dropna()

# ----------------------------
# 3. Encode target variable
# ----------------------------
# Convert class from {2,4} to {0,1} for ML models
# 0 = Benign, 1 = Malignant
df["class"] = df["class"].map({2: 0, 4: 1})

# ----------------------------
# 4. Optional: create ratio features
# ----------------------------
# Small epsilon to avoid division by zero
epsilon = 1e-6

# Ratio 1: uniformity_cell_size / uniformity_cell_shape
df["size_shape_ratio"] = df["uniformity_cell_size"] / (df["uniformity_cell_shape"] + epsilon)

# Ratio 2: clump_thickness / uniformity_cell_size
df["thickness_size_ratio"] = df["clump_thickness"] / (df["uniformity_cell_size"] + epsilon)

# Ratio 3: bare_nuclei / normal_nucleoli
df["bare_nuclei_nucleoli_ratio"] = df["bare_nuclei"] / (df["normal_nucleoli"] + epsilon)

# ----------------------------
# 5. Prepare feature matrix X and target vector y
# ----------------------------
# X = all features except 'id' and 'class'
X = df.drop(columns=["id", "class"])

# y = target variable (0 = Benign, 1 = Malignant)
y = df["class"]

# ----------------------------
# 6. Normalize features (optional)
# ----------------------------
# Decision Trees do not require normalization, but we scale to keep consistent feature ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 7. Train Decision Tree Classifier
# ----------------------------
# Create a Decision Tree model
tree = DecisionTreeClassifier(random_state=42)

# Fit the model to the scaled feature matrix and target
tree.fit(X_scaled, y)

# ----------------------------
# 8. Extract feature importance
# ----------------------------
# Decision Tree assigns an importance score to each feature
# Higher score → feature contributes more to classification
importances = tree.feature_importances_

# Create a DataFrame to make plotting easier
feature_importance_df = pd.DataFrame({
    "feature": X.columns,      # feature names
    "importance": importances  # importance scores
}).sort_values(by="importance", ascending=False)  # sort descending

# ----------------------------
# 9. Plot feature importance
# ----------------------------
plt.figure(figsize=(8,5))

# Horizontal bar chart for readability
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"], color='skyblue')

# Invert y-axis to have highest importance on top
plt.gca().invert_yaxis()

# Labels and title
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance from Decision Tree")

# Display the plot
plt.show()
