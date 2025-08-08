# Student Performance Prediction with Custom Linear Regression

# Author: Joshua E. Brown

# 

# Overview

# This project demonstrates a full, from-scratch machine learning workflow for predicting student performance. It uses only base scientific Python tools (NumPy, Pandas, Matplotlib, Seaborn) and custom modules for all core machine learning algorithms, explicitly avoiding scikit-learn. The goal is to understand, implement, and explain classical regression techniques—including feature scaling, batch gradient descent, and LASSO regularization—on real data.

# 

# Structure

# text

# your\_project/

# │

# ├── main.py           # Orchestrates full ML workflow

# ├── functions.py      # Predict function and regression metrics

# ├── models.py         # Gradient Descent and LASSO Regression

# ├── data/             # Folder containing dataset CSV

# ├── README.md         # This file

# └── (other supporting scripts or notebooks)

# Modules

# main.py

# Runs all steps of the workflow: loading data, preprocessing, model training, evaluation, plotting, and explanation.

# 

# functions.py

# Contains:

# 

# predict: Linear prediction (vectorized NumPy implementation).

# 

# rmse\_mae\_r2: Regression performance metric computation (RMSE, MAE, R²).

# 

# models.py

# Contains:

# 

# g\_d\_func: Custom batch gradient descent for linear regression.

# 

# lasso\_g\_d\_func: Batch gradient descent with L1 regularization (LASSO).

# 

# Workflow and ML Rationale

# The project executes the following, with clear reasons for each step:

# 

# Load and explore data:

# Use explicit dtype mapping for RAM efficiency and reproducibility.

# 

# Data cleaning and mapping:

# Categoricals are mapped to numerics so that ML algorithms can interpret them; string normalization avoids misspellings.

# 

# Feature scaling (Z-score normalization):

# Standardizes inputs ensuring fair gradient updates and comparability between features.

# 

# Train/test split (80/20):

# Enables proper model validation and prevents information leakage, ensuring credible performance estimates.

# 

# Null value check and cleaning:

# Ensures models only train on valid data for reliable evaluation.

# 

# Feature selection and array conversion:

# Prepares appropriate feature matrices for model input, enforcing correct shapes and types.

# 

# Gradient descent on synthetic/demo data:

# Serves as a proof-of-concept before working with real data, validating model code and learning behavior.

# 

# Cost visualization:

# Plots cost (loss) during training to diagnose convergence, tuning, and spot potential issues.

# 

# Performance metrics (RMSE, MAE, R²):

# Standard metrics quantify model accuracy and explanatory power; all computed from scratch for transparency.

# 

# Training on real data with gradient descent:

# Builds and evaluates a predictive model, with all steps manual for deep learning.

# 

# Running and visualizing LASSO (L1 regularization):

# Encourages sparseness in the weights, which is critical for feature selection and reduces overfitting.

# 

# Residual analysis plots:

# Diagnoses model error patterns to validate assumptions and model health.

# 

# Explanatory comments:

# Document the purpose behind visualizations and interpretation of feature impact.

# 

# How to Run

# Prepare the Data:

# Ensure Student\_Performance.csv is in the data/Student\_Performance/ directory.

# 

# Install Requirements:

# Ensure you have Python 3.8+, with numpy, pandas, matplotlib, and seaborn installed.

# 

# Example:

# 

# bash

# pip install numpy pandas matplotlib seaborn

# Run the Script:

# 

# bash

# python main.py

# This executes the full workflow, reporting intermediate data, training progress, evaluation metrics, and plots.

# 

# Import Functions/Models for Reuse:

# You can reuse the functions and model modules in your own experiments:

# 

# python

# from functions import predict, rmse\_mae\_r2

# from models import g\_d\_func, lasso\_g\_d\_func

# Engineering and Portfolio Notes

# No external ML libraries (scikit-learn, etc.) are used.

# All gradients, predictions, metrics, and regularization are implemented directly for transparency and learning.

# 

# Modular Design:

# Each script focuses on a clear purpose. All functions require inputs as arguments (no global variables or hidden dependencies).

# 

# Reason-Driven Steps:

# Each major step is commented with the reasoning, helping reviewers understand why each is included—not just how.

# 

# Best Practices:

# 

# Code is wrapped in if \_\_name\_\_ == "\_\_main\_\_": to ensure safe importing.

# 

# Docstrings document intent and argument types.

# 

# All code follows standard style as per PEP 8.

# 

# Review and Visualization:

# Intermediate outputs and visualizations make it easy to check correctness, convergence, and model health throughout.

# 

# Why This Approach Matters

# Shows your ability to implement ML techniques "from scratch"—crucial for foundational understanding as a junior engineer.

# 

# Demonstrates engineering discipline: code organization, reproducibility, and transparency.

# 

# Exhibits communication of reasoning: every step is annotated and explained.

# 

# Makes you job-ready for teams that value depth and clarity—not just API usage.

# 

# For further details, inspect the docstrings in functions.py and models.py, and follow the commented explanations in main.py.

# If you encounter errors or want to extend the project, modularity ensures you can do so safely, without confusion over variables or dependencies.

# 

# Good luck and happy modeling!

