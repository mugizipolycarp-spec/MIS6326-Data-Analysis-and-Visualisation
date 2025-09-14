# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 07:51:52 2025

@author: polycarp.mugizi
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load dataset
# ----------------------------
# URL for Breast Cancer Wisconsin (Original) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Column names based on UCI repository description
columns = [
    "id",                         # unique identifier, not used in modeling
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

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
# 'bare_nuclei' column has "?" for missing values
# Convert to numeric; non-numeric entries become NaN
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop rows with missing values to avoid issues in analysis
df = df.dropna()

# ----------------------------
# 3. Encode target variable
# ----------------------------
# Map 'class' from {2,4} to {0,1} for ML modeling
# 0 = Benign, 1 = Malignant
df["class"] = df["class"].map({2: 0, 4: 1})

# ----------------------------
# 4. Optional: Create ratio features
# ----------------------------
# Ratios can highlight relationships between features
epsilon = 1e-6  # tiny value to avoid division by zero

# Ratio 1: Uniformity of cell size / Uniformity of cell shape
df["size_shape_ratio"] = df["uniformity_cell_size"] / (df["uniformity_cell_shape"] + epsilon)

# Ratio 2: Clump thickness / Uniformity of cell size
df["thickness_size_ratio"] = df["clump_thickness"] / (df["uniformity_cell_size"] + epsilon)

# Ratio 3: Bare nuclei / Normal nucleoli
df["bare_nuclei_nucleoli_ratio"] = df["bare_nuclei"] / (df["normal_nucleoli"] + epsilon)

# ----------------------------
# 5. Prepare features (X) and target (y)
# ----------------------------
# X = all features except 'id' and 'class'
X = df.drop(columns=["id", "class"])

# y = target variable
y = df["class"]

# ----------------------------
# 6. Normalize features
# ----------------------------
# StandardScaler standardizes features to mean=0, std=1
# This is important for PCA, t-SNE, and UMAP to work properly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 7a. PCA (Principal Component Analysis)
# ----------------------------
# PCA is a linear dimensionality reduction technique
# We reduce to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Check how much variance each component explains
print("Explained variance ratio by PCA:", pca.explained_variance_ratio_)

# ----------------------------
# 7b. t-SNE (t-distributed Stochastic Neighbor Embedding)
# ----------------------------
# t-SNE is a non-linear technique, preserves local structure
# Perplexity controls balance between local/global structure
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# ----------------------------
# 7c. UMAP (Uniform Manifold Approximation and Projection)
# ----------------------------
# UMAP is a non-linear technique preserving both local and global structure
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

# ----------------------------
# 8. Visualize 2D embeddings
# ----------------------------
def plot_2d(X_embedded, title):
    """
    Plot 2D embedding with color by tumor class
    X_embedded: 2D array of shape (n_samples, 2)
    title: plot title
    """
    plt.figure(figsize=(6,5))
    plt.scatter(
        X_embedded[:,0], X_embedded[:,1], 
        c=y,                     # color by class
        cmap='coolwarm',         # color map
        alpha=0.7                # point transparency
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Class (0=Benign, 1=Malignant)")
    plt.show()

# Plot PCA, t-SNE, and UMAP embeddings
plot_2d(X_pca, "PCA of Tumor Attributes")
plot_2d(X_tsne, "t-SNE of Tumor Attributes")
plot_2d(X_umap, "UMAP of Tumor Attributes")
