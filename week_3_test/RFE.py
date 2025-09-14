# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:43:30 2025

@author: polycarp.mugizi
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# ----------------------------
# 1. Load dataset
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
columns = [
    "id","clump_thickness","uniformity_cell_size","uniformity_cell_shape",
    "marginal_adhesion","single_epithelial_cell_size","bare_nuclei",
    "bland_chromatin","normal_nucleoli","mitoses","class"
]
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")
df = df.dropna()

# ----------------------------
# 3. Encode target
# ----------------------------
df["class"] = df["class"].map({2: 0, 4: 1})

# ----------------------------
# 4. Optional: Create additional ratio features
# ----------------------------
epsilon = 1e-6
df["size_shape_ratio"] = df["uniformity_cell_size"] / (df["uniformity_cell_shape"] + epsilon)
df["thickness_size_ratio"] = df["clump_thickness"] / (df["uniformity_cell_size"] + epsilon)
df["bare_nuclei_nucleoli_ratio"] = df["bare_nuclei"] / (df["normal_nucleoli"] + epsilon)

# ----------------------------
# 5. Prepare feature matrix X and target y
# ----------------------------
# Exclude 'id' and 'class'
X = df.drop(columns=["id","class"])
y = df["class"]

# ----------------------------
# 6. Normalize features (important for some models)
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 7. Apply RFE using Logistic Regression
# ----------------------------
# Create logistic regression model
model = LogisticRegression(max_iter=1000)

# RFE: select top 5 features
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_scaled, y)

# ----------------------------
# 8. Output results
# ----------------------------
# Boolean mask of selected features
print("Selected features (True = chosen):")
print(pd.Series(rfe.support_, index=X.columns))

# Feature ranking (1 = most important)
print("\nFeature ranking:")
print(pd.Series(rfe.ranking_, index=X.columns))
