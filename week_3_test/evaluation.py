# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 08:10:18 2025

@author: polycarp.mugizi
"""

# ----------------------------
# Import libraries
# ----------------------------
import pandas as pd  # for data manipulation
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.preprocessing import StandardScaler  # for feature normalization
from sklearn.naive_bayes import GaussianNB  # Naïve Bayes classifier
from sklearn.neighbors import KNeighborsClassifier  # k-NN classifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_curve, auc
)
from sklearn.calibration import calibration_curve  # for calibration curves
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for numerical operations

# ----------------------------
# 1. Load dataset
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Define column names according to UCI repository
columns = [
    "id","clump_thickness","uniformity_cell_size","uniformity_cell_shape",
    "marginal_adhesion","single_epithelial_cell_size","bare_nuclei",
    "bland_chromatin","normal_nucleoli","mitoses","class"
]

# Read the CSV data into a pandas DataFrame
df = pd.read_csv(url, names=columns)

# ----------------------------
# 2. Handle missing values
# ----------------------------
# 'bare_nuclei' column contains "?" for missing values
# Convert to numeric; non-numeric entries become NaN
df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")

# Drop rows with missing values
df = df.dropna()

# ----------------------------
# 3. Encode target variable
# ----------------------------
# Original class: 2 = Benign, 4 = Malignant
# Convert to binary: 0 = Benign, 1 = Malignant
df["class"] = df["class"].map({2: 0, 4: 1})

# ----------------------------
# 4. Optional ratio features
# ----------------------------
epsilon = 1e-6  # small constant to avoid division by zero

# Feature 1: ratio of cell size to cell shape
df["size_shape_ratio"] = df["uniformity_cell_size"] / (df["uniformity_cell_shape"] + epsilon)

# Feature 2: ratio of clump thickness to cell size
df["thickness_size_ratio"] = df["clump_thickness"] / (df["uniformity_cell_size"] + epsilon)

# Feature 3: ratio of bare nuclei to nucleoli
df["bare_nuclei_nucleoli_ratio"] = df["bare_nuclei"] / (df["normal_nucleoli"] + epsilon)

# ----------------------------
# 5. Prepare features and target
# ----------------------------
X = df.drop(columns=["id", "class"])  # predictor features
y = df["class"]  # target variable

# ----------------------------
# 6. Split dataset into training and testing sets
# ----------------------------
# 80% training, 20% testing
# stratify=y ensures same class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 7. Normalize features
# ----------------------------
# Important for k-NN because it is distance-based
# Optional for Naïve Bayes but still helps
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit scaler on training set
X_test_scaled = scaler.transform(X_test)  # transform test set using same scaler

# ----------------------------
# 8a. Train Gaussian Naïve Bayes
# ----------------------------
nb_model = GaussianNB()  # instantiate Naïve Bayes
nb_model.fit(X_train_scaled, y_train)  # train on training data

# Predict class labels on test set
y_pred_nb = nb_model.predict(X_test_scaled)

# Predict probabilities for positive class (Malignant)
y_prob_nb = nb_model.predict_proba(X_test_scaled)[:,1]

# ----------------------------
# 8b. Train k-Nearest Neighbors
# ----------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)  # set k=5 neighbors
knn_model.fit(X_train_scaled, y_train)  # train on training data

# Predict class labels on test set
y_pred_knn = knn_model.predict(X_test_scaled)

# Predict probabilities for positive class (Malignant)
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:,1]

# ----------------------------
# 9. Evaluation function
# ----------------------------
def evaluate_model(y_true, y_pred, y_prob, model_name):
    """
    Evaluate classifier performance with:
    - Accuracy
    - Sensitivity (Recall for positive class)
    - Specificity (Recall for negative class)
    - Confusion matrix
    - Classification report
    - ROC-AUC
    - Calibration curve
    """
    print(f"=== {model_name} ===")
    
    # Accuracy: proportion of correctly classified samples
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(acc,4))
    
    # Confusion matrix: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:\n", cm)
    
    # Sensitivity = TP / (TP + FN)
    sensitivity = tp / (tp + fn)
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp)
    
    print("Sensitivity (Recall):", round(sensitivity,4))
    print("Specificity:", round(specificity,4))
    
    # Detailed classification report (precision, recall, f1-score)
    print("Classification Report:\n", classification_report(y_true, y_pred))
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print("ROC-AUC:", round(roc_auc,4))
    
    # Plot ROC curve
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='red', linestyle='--')  # random classifier
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')  # perfect calibration
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend()
    plt.show()

# ----------------------------
# 10. Evaluate models
# ----------------------------
evaluate_model(y_test, y_pred_nb, y_prob_nb, "Gaussian Naïve Bayes")
evaluate_model(y_test, y_pred_knn, y_prob_knn, "k-Nearest Neighbors (k=5)")
