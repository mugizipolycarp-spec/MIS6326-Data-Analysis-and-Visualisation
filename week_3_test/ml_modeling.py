# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 08:07:16 2025

@author: polycarp.mugizi
"""

# ----------------------------
# Import necessary libraries
# ----------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------
# 1. Load the dataset
# ----------------------------
# URL for the Breast Cancer Wisconsin (Original) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Column names based on UCI repository
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
# 'bare_nuclei' column has "?" for missing values
# Convert to numeric, non-numeric entries become NaN
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop rows containing NaN to clean the dataset
df = df.dropna()

# ----------------------------
# 3. Encode target variable
# ----------------------------
# Convert 'class' from {2,4} to {0,1} for ML models
# 0 = Benign, 1 = Malignant
df["class"] = df["class"].map({2: 0, 4: 1})

# ----------------------------
# 4. Optional: create ratio features
# ----------------------------
# Small epsilon to avoid division by zero
epsilon = 1e-6

# Feature 1: size_shape_ratio = uniformity_cell_size / uniformity_cell_shape
df["size_shape_ratio"] = df["uniformity_cell_size"] / (df["uniformity_cell_shape"] + epsilon)

# Feature 2: thickness_size_ratio = clump_thickness / uniformity_cell_size
df["thickness_size_ratio"] = df["clump_thickness"] / (df["uniformity_cell_size"] + epsilon)

# Feature 3: bare_nuclei_nucleoli_ratio = bare_nuclei / normal_nucleoli
df["bare_nuclei_nucleoli_ratio"] = df["bare_nuclei"] / (df["normal_nucleoli"] + epsilon)

# ----------------------------
# 5. Prepare features and target
# ----------------------------
# X = all features except 'id' and 'class'
X = df.drop(columns=["id", "class"])

# y = target variable
y = df["class"]

# ----------------------------
# 6. Split dataset into training and testing sets
# ----------------------------
# 80% training, 20% testing, stratify=y ensures class balance is preserved
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 7. Normalize features
# ----------------------------
# Important for k-NN because it is distance-based
# Naïve Bayes can also benefit from normalized features
scaler = StandardScaler()

# Fit scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 8a. Train Gaussian Naïve Bayes
# ----------------------------
# GaussianNB assumes features are normally distributed
nb_model = GaussianNB()

# Fit the Naïve Bayes model to the training data
nb_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_nb = nb_model.predict(X_test_scaled)

# ----------------------------
# 8b. Train k-Nearest Neighbors
# ----------------------------
# k-NN classifies based on the majority class of the k nearest neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)  # k=5

# Fit k-NN model to training data
knn_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_knn = knn_model.predict(X_test_scaled)

# ----------------------------
# 9. Evaluate models
# ----------------------------
def evaluate_model(y_true, y_pred, model_name):
    """
    Prints evaluation metrics for a model
    y_true: true labels
    y_pred: predicted labels
    model_name: string name of the model
    """
    print(f"=== {model_name} ===")
    
    # Accuracy score
    print("Accuracy:", accuracy_score(y_true, y_pred))
    
    # Confusion matrix shows true positives, true negatives, false positives, false negatives
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    
    # Classification report shows precision, recall, f1-score for each class
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("\n")

# Evaluate Gaussian Naïve Bayes
evaluate_model(y_test, y_pred_nb, "Gaussian Naïve Bayes")

# Evaluate k-Nearest Neighbors
evaluate_model(y_test, y_pred_knn, "k-Nearest Neighbors (k=5)")
