# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:59:02 2025

@author: polycarp.mugizi
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Try importing UMAP, skip if not installed
try:
    import umap
    umap_installed = True
except ModuleNotFoundError:
    print("UMAP not installed, skipping UMAP.")
    umap_installed = False

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
# 4. Optional: ratio features
# ----------------------------
epsilon = 1e-6
df["size_shape_ratio"] = df["uniformity_cell_size"] / (df["uniformity_cell_shape"] + epsilon)
df["thickness_size_ratio"] = df["clump_thickness"] / (df["uniformity_cell_size"] + epsilon)
df["bare_nuclei_nucleoli_ratio"] = df["bare_nuclei"] / (df["normal_nucleoli"] + epsilon)

# ----------------------------
# 5. Prepare features and target
# ----------------------------
X = df.drop(columns=["id","class"])
y = df["class"]

# ----------------------------
# 6. Normalize features
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 7a. PCA
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio for each PCA component
pca_variance = pca.explained_variance_ratio_
print("PCA explained variance ratio:", pca_variance)

# ----------------------------
# 7b. t-SNE
# ----------------------------
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Approximate "variance explained" for t-SNE: variance of each component
tsne_variance = np.var(X_tsne, axis=0) / np.sum(np.var(X_tsne, axis=0))
print("t-SNE pseudo-variance ratio:", tsne_variance)

# ----------------------------
# 7c. UMAP (if installed)
# ----------------------------
if umap_installed:
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_scaled)
    umap_variance = np.var(X_umap, axis=0) / np.sum(np.var(X_umap, axis=0))
    print("UMAP pseudo-variance ratio:", umap_variance)

# ----------------------------
# 8. Plot variance explained
# ----------------------------
plt.figure(figsize=(7,5))
plt.bar([1,2], pca_variance, alpha=0.7, label="PCA")
plt.xticks([1,2], ["Component 1", "Component 2"])
plt.ylabel("Explained Variance Ratio")
plt.title("Variance Explained by PCA Components")
plt.legend()
plt.show()

# t-SNE & UMAP variance visualization (pseudo-variance)
plt.figure(figsize=(7,5))
plt.bar([1,2], tsne_variance, alpha=0.7, label="t-SNE")
if umap_installed:
    plt.bar([1.3,2.3], umap_variance, alpha=0.7, label="UMAP")
plt.xticks([1,2,1.3,2.3] if umap_installed else [1,2], 
           ["tSNE Comp1","tSNE Comp2","UMAP Comp1","UMAP Comp2"] if umap_installed else ["Comp1","Comp2"])
plt.ylabel("Pseudo-Variance Ratio")
plt.title("t-SNE and UMAP Component Variance")
plt.legend()
plt.show()
