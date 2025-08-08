"""
functions.py

Provides core machine learning functions for the Student Performance
Regression project without external ML libraries (no scikit-learn).

Contents:
    - predict: Generates linear model predictions with learned weights (w)
      and bias (b) given input features (X), using NumPy vectorization.
    - rmse_mae_r2: Calculates and prints standard regression performance
      metrics (RMSE, MAE, R²) for evaluation and diagnostic purposes.

Design rationale:
    - Enables reproducible, transparent predictions and metrics calculation.
    - All logic is implemented in plain NumPy for educational clarity.
    - Functions are reusable—no reliance on global variables; all inputs
      must be provided directly as arguments.
    - Docstrings clarify purpose, expected argument types, and example usage.

Why this structure?
    - Keeps workflow modular and portable for portfolio demonstration.
    - Separates key model logic from data processing and visualization,
      encouraging professional engineering practices.
    - Ensures functions can be tested and reused across different scripts,
      projects, or datasets.

Usage:
    Import functions as needed, passing prepared NumPy arrays and model
    parameters.

        from functions import predict, rmse_mae_r2

    Example:
        y_hat = predict(X, b, w)
        rmse_mae_r2(X, b, w, y_actual)

Notes:
    - All functions require compatible NumPy array shapes.
    - No machine learning external packages are used, per project
      requirements.

Author: Joshua E. Brown
"""
import numpy as np

# ===============================================================
# Prediction function/formula
# ===============================================================


def predict(X, b, w):
    """
    Purpose:
        Generates predictions from a linear model using learned weights
        (`w`) and bias (`b`) given input features (`X`). This forms the
        basis for evaluating model performance on new or unseen data.

    Parameters:
        X (np.ndarray): Matrix of input features with shape
            (n_samples, n_features).
            Reason: Represents the data points to predict for. Each row is
            a sample, each column a feature.

        w (np.ndarray): Array of trained model weights, shape
            (n_features,).
            Reason: Contains coefficients learned from training. Each
            weight quantifies the contribution of its corresponding
            feature.

        b (float): Scalar bias (intercept) term.
            Reason: Accounts for the base value of prediction when all
            input features are zero. Ensures model predictions aren't
            forced through the origin.

    Returns:
        np.ndarray: Array of predicted values, shape (n_samples,).
            Reason: Provides the output for each sample in `X`. Enables
            further analysis, such as computing metrics or making
            decisions based on the model.

    Actions:
        - Computes dot-product between `X` and `w` to yield a linear
          combination of features and weights.
        - Adds the bias `b` to each prediction.
        - Returns predicted values for further evaluation or use.

    Example usage:
        >>> y_hat = predict(X_test, b, w)
        >>> print(y_hat.shape)
        (n_samples,)

    Notes:
        - Assumes `X` and `w` have compatible shapes.
        - Pure NumPy; no scikit-learn used.
    """
    return X.dot(w) + b


# ===============================================================
# Performance Metric Calculation
# ===============================================================


def rmse_mae_r2(X, b, w, y_actual):
    """
    Purpose:
        Calculates and prints key regression performance metrics:
        RMSE (root mean squared error), MAE (mean absolute error), and
        R² (coefficient of determination) for model evaluation.

    Parameters:
        X (np.ndarray): Matrix of model input features,
            shape (n_samples, n_features).
            Reason: Needed for predictions.

        w (np.ndarray): Model weights. Shape (n_features,) or broadcastable.
            Reason: Defines learned coefficients for prediction.

        b (float): Model bias term (scalar).
            Reason: Represents the intercept.

        y_actual (np.ndarray): True target values, shape (n_samples,).
            Reason: Ground truth for accuracy assessment.

    Returns:
        None: Prints metrics.

    Actions:
        - Enforces shape for weights (`w.ravel()`).
        - Calculates RMSE, MAE, R².
        - Prints them in a concise format.

    Example usage:
        >>> rmse_mae_r2(X_test, b, w, y_test)
        RMSE 2.5235 | MAE 1.9412 | R² 0.8501
    """
    w = w.ravel()
    y_hat = predict(X, b, w)
    rmse = np.sqrt(np.mean((y_actual - y_hat) ** 2))
    mae = np.mean(np.abs(y_actual - y_hat))
    r2 = 1 - np.sum((y_actual - y_hat) ** 2) / np.sum(
        (y_actual - y_actual.mean()) ** 2
    )
    print(f' RMSE {rmse:,.4f} | MAE {mae:,.4f} | R² {r2:.4f}')
