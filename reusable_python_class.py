# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 13:47:17 2025

@author: polycarp.mugizi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        # Make a copy of the DataFrame to avoid modifying the original one
        self.df = df.copy()
        # Keep a log of all steps performed during cleaning
        self.log = []

    def handle_missing(self, strategy="mean", fill_value=None):
        #Handle missing values in the dataset using different strategies:
        #- mean: fill with column mean
        #- median: fill with column median
        #- mode: fill with column mode (most frequent value)
        #- constant: fill with a given constant value
        #- drop: drop rows where column has missing values
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:   # Only process if column has missing values
                if strategy == "mean" and self.df[col].dtype != "object":
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                    self.log.append(f"Filled missing in {col} with mean.")

                elif strategy == "median" and self.df[col].dtype != "object":
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    self.log.append(f"Filled missing in {col} with median.")

                elif strategy == "mode":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                    self.log.append(f"Filled missing in {col} with mode.")

                elif strategy == "constant":
                    self.df[col] = self.df[col].fillna(fill_value)
                    self.log.append(f"Filled missing in {col} with constant value {fill_value}.")

                elif strategy == "drop":
                    before = self.df.shape[0]  # Number of rows before dropping
                    self.df = self.df.dropna(subset=[col])
                    after = self.df.shape[0]   # Number of rows after dropping
                    self.log.append(f"Dropped {before - after} rows due to missing in {col}.")
        return self   # Return self to allow method chaining

    def handle_outliers(self, method="IQR", factor=1.5):
        
        #Detect and handle outliers in numeric columns using the IQR method.
        #Outliers are capped at the lower/upper bounds instead of removing them.
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)  # 25th percentile
            Q3 = self.df[col].quantile(0.75)  # 75th percentile
            IQR = Q3 - Q1                     # Interquartile Range
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            
            # Cap values outside lower/upper limits
            self.df[col] = np.where(self.df[col] < lower, lower,
                                    np.where(self.df[col] > upper, upper, self.df[col]))
            self.log.append(f"Capped {outliers} outliers in {col}.")
        return self

    def normalize(self, columns=None):
        
        #Normalize numeric values to a [0,1] range using MinMaxScaler.
        #If no columns are provided, all numeric columns are normalized.
        
        scaler = MinMaxScaler()
        cols = columns or self.df.select_dtypes(include=[np.number]).columns
        self.df[cols] = scaler.fit_transform(self.df[cols])
        self.log.append(f"Normalized columns: {list(cols)}.")
        return self

    def standardize(self, columns=None):
        
        #Standardize numeric columns (mean=0, standard deviation=1).
        #Useful when columns have very different scales.
        
        scaler = StandardScaler()
        cols = columns or self.df.select_dtypes(include=[np.number]).columns
        self.df[cols] = scaler.fit_transform(self.df[cols])
        self.log.append(f"Standardized columns: {list(cols)}.")
        return self

    def clean_categorical(self):
        
        #Clean categorical (string) columns:
        #- strip extra spaces
        #- convert to lowercase
        #- replace 'nan' strings with 'unknown'
        
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            self.df[col] = self.df[col].astype(str).str.strip().str.lower()
            self.df[col] = self.df[col].replace("nan", "unknown")
            self.log.append(f"Cleaned categorical column {col}.")
        return self

    def get_log(self):
        #Return the list of cleaning steps performed.
        return self.log

    def get_data(self):
        #Return the cleaned DataFrame.
        return self.df

    
    
# Load dataset
df = pd.read_csv(r"D:\Msc\datasets\heart_disease.csv")

# Show sample of original dataset
print("Original Data Sample:")
print(df.head(10), "\n")

# Create cleaner instance
cleaner = DataCleaner(df)

# Apply cleaning steps
cleaned_df = (
    cleaner.handle_missing(strategy="mode")         # Handle missing values with mode
           .handle_outliers()                       # Cap outliers
         .standardize(columns=["chol", "thalach"])    # Standardize selected columns
           .clean_categorical()                     # Clean categorical columns
           .get_data()                              # Get cleaned DataFrame
)

# Show results
print("Cleaned Data Sample:")
print(cleaned_df.head(10))

print("\nCleaning Log:")
for step in cleaner.get_log():
    print("-", step)

# Save the cleaned dataset
cleaned_df.to_csv(r"D:\Msc\datasets\heart_disease_cleaned.csv", index=False)

