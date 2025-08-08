"""
models.py

Implements core linear regression algorithms for the Student Performance
project, using explicit NumPy (no scikit-learn) for transparency and
learning purposes.

Contents:
    - g_d_func: Batch gradient descent for linear regression;
      fits model weights and bias to minimize mean squared error (MSE).
    - lasso_g_d_func: Batch gradient descent with L1 regularization
      (LASSO); encourages sparse model coefficients, aiding feature selection
      and reducing overfitting.

Design rationale:
    - Models are implemented "from scratch" in NumPy to build foundational
      machine learning intuition and comply with portfolio requirements.
    - No global variables; all required data and hyperparameters must be
      passed in explicitly as function arguments.
    - Reporting (printed weights, bias, cost history) aids transparency
      and helps monitor optimization progress.
    - Functions return all trained parameters and cost histories for
      later analysis or visualization.

Usage:
    Import the model training functions in your main script and call them
    with NumPy arrays for features and targets:

        from models import g_d_func, lasso_g_d_func

        b, w, cost_history = g_d_func(X, y, ...)
        b_lasso, w_lasso, cost_hist_lasso = lasso_g_d_func(X, y, ...)

Reason for structure:
    - Clear separation of model training logic from data preparation and
      evaluation metrics, following sound engineering and project hygiene.
    - Enables reuse and independent testing in multiple scripts or experiments.

Notes:
    - All modeling is performed with consistent dtype hints for speed and
      reproducibility.
    - LASSO implementation uses standard gradient descent with subgradient
      method for L1 penalty.

Author: Joshua E. Brown
"""
import numpy as np

# ===============================================================
# Gradient Descent Algorithm Function
# ===============================================================


def g_d_func(
    X,
    y,
    b_init=0.0,
    learning_rate=0.05,
    iterations=1000,
    w_init=None,
    i_displayed=100
):
    """
    Purpose:
        Performs linear regression using gradient descent optimization.
        Iteratively updates weights (`w`) and bias (`b`) to minimize
        mean squared error (MSE).

    Parameters:
        X (np.ndarray): Features, shape (n_samples, n_features).
        y (np.ndarray): Targets, shape (n_samples,).
        b_init (float): Initial bias.
        learning_rate (float): Step size.
        iterations (int): Update steps.
        w_init (np.ndarray): Initial weights (optional).
        i_displayed (int): Print interval.

    Returns:
        tuple: (b, w, cost_history)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n_features = X.shape[1]

    if w_init is None:
        w = np.zeros(n_features, dtype=np.float32)
    else:
        w = np.asarray(w_init, dtype=np.float32)
        assert w.shape == (n_features,), 'w_init length must equal n_features'

    b = float(b_init)
    cost_history = []

    for i in range(iterations):
        y_hat = X.dot(w) + b
        error = y_hat - y

        grad_w = (2 / len(X)) * X.T.dot(error)
        grad_b = (2 / len(X)) * error.sum()

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        mse = (error ** 2).mean()
        cost_history.append(mse)

        if i % i_displayed == 0:
            print(f'Iter {i:5d}: MSE={mse:.4f}')

    print('Final Weights:', w)
    print('Final Bias:', b)
    print('Final Cost:', mse)

    return b, w, cost_history


# ===============================================================
# LASSO Regression Implementation with Batch Gradient Descent
# ===============================================================


def lasso_g_d_func(
    X,
    y,
    learning_rate=0.01,
    b_init=0.0,
    lambda_=0.1,
    iterations=1000
):
    """
    Purpose:
        Implements LASSO (L1-regularized) linear regression using
        batch gradient descent in NumPy.

    Parameters:
        X (np.ndarray): Features, shape (n_samples, n_features).
        y (np.ndarray): Targets, shape (n_samples,).
        learning_rate (float): GD step size.
        b_init (float): Initial bias.
        lambda_ (float): L1 penalty strength.
        iterations (int): Update steps.

    Returns:
        tuple: (b, w, cost_history)
    """
    n_samples, n_features = X.shape
    w = np.random.randn(n_features) * 0.000005
    cost_history = []
    b = b_init

    for i_1 in range(iterations):
        y_hat = X.dot(w) + b
        residual = y - y_hat

        dw = (-1 / n_samples) * X.T.dot(residual) + lambda_ * np.sign(w)
        db = (-1 / n_samples) * np.sum(residual)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = (1 / (2 * n_samples) * np.sum(residual ** 2) +
                lambda_ * np.sum(np.abs(w)))
        cost_history.append(cost)

        if i_1 % 10 == 0:
            print(f'Iteration: {i_1}: Cost={cost:.4f}')

    print('Final Weights:', w)
    print('Final Bias:', b)
    print('Final Cost:', cost)

    return b, w, cost_history
