"""
Main script for student performance prediction using custom linear regression.

This script performs the end-to-end machine learning workflow on the
Student Performance dataset, including data loading, preprocessing,
feature engineering, model training using gradient descent and LASSO
regularization, evaluation, and visualization of results.

Modules:
    - NumPy and Pandas for data handling.
    - Matplotlib and Seaborn for visualizations.
    - Custom `functions` and `models` modules for core ML algorithms
      and metrics.

Workflow Steps:
    1. Load and explore dataset with explicit dtype hinting for memory
    efficiency.
    2. Clean and map categorical features to numerical.
    3. Scale numeric features using Z-score normalization.
    4. Split data into 80/20 train and test sets.
    5. Perform null value checks and clean data copies.
    6. Select relevant feature columns for modeling.
    7. Convert data to NumPy arrays for efficient numeric operations.
    8. Run batch gradient descent on demo data to verify functionality.
    9. Visualize demo training cost.
    10. Compute and print performance metrics (RMSE, MAE, R²) on demo data.
    11. Train linear regression with gradient descent on actual data.
    12. Visualize training cost for actual data.
    13. Evaluate performance metrics on test data.
    14. Run LASSO gradient descent for feature-regularized training.
    15. Visualize LASSO training progress and test set performance.
    16. Conduct residual analysis with scatter and histogram plots.
    17. Provide explanatory comments on residual diagnostics and feature
    importance.

Notes:
    - The code uses pure NumPy implementations for ML algorithms
      without scikit-learn, emphasizing foundational understanding.
    - Encapsulated within `if __name__ == "__main__":` to allow safe
      imports without execution.
    - Visualizations aid in model diagnostics and quality assessment.
    - The working directory must contain the data CSV at the specified path.

Example usage:
    Run this script directly to execute the full workflow:

    $ python main.py

    Import functions or models from separate modules as needed for
    reuse or testing.

Author: Joshua E. Brown
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functions
import models

if __name__ == "__main__":
    # ===============================================================
    # Preprocessing of Dataset
    # ===============================================================

    # Implementation of explicit dtype hinting
    dtype_map = {
        'hours_studied': 'float32',
        'previous_scores': 'float32',
        'extra_activities': 'category',
        'sleep_hours': 'float32',
        'sample_question': 'float32',
        'performance_index': 'float32'
    }

    column_names = [
        'hours_studied', 'previous_scores', 'extra_activities',
        'sleep_hours', 'sample_question', 'performance_index'
    ]

    sp_df = pd.read_csv(
        '../data/dataset_a_s_e_p/Student_Performance/Student_Performance.csv',
        names=column_names, header=0, dtype=dtype_map
    )

    # Data exploration
    print('Print first 20 rows of each column:\n', sp_df.head(20))

    print('\nData description:\n')
    with pd.option_context('display.max_columns', None, 'display.max_rows',
                           None):
        print(sp_df.describe())

    # ===============================================================
    # Map 'extra_activities'
    # ===============================================================

    # Standardize values
    sp_df['extra_activities'] = (
        sp_df['extra_activities']
        .str.strip()
        .str.lower()
    )

    # Map 'yes' to 1, 'no' to 0
    sp_df['extra_activities'] = sp_df['extra_activities'].map(
        {'yes': 1, 'no': 0}
    ).astype(np.float32)

    print(sp_df.head(20))

    print('\nData description:\n')
    with pd.option_context('display.max_columns', None, 'display.max_rows',
                           None):
        print(sp_df.describe())

    # ===============================================================
    # Feature Scaling: Z-Score Normalization
    # ===============================================================

    numeric_columns = [
        'hours_studied', 'previous_scores', 'sleep_hours',
        'sample_question'
    ]
    # Skips 'extra_activities' (already scaled), 'performance_index' is target

    # Mean/STD calculation
    mean = sp_df[numeric_columns].mean()
    std = sp_df[numeric_columns].std()

    # Standardization of each column
    sp_df[numeric_columns] = (sp_df[numeric_columns] - mean) / std

    # ===============================================================
    # 80/20 Split (Data is pre-shuffled)
    # ===============================================================

    split_index = int(len(sp_df) * 0.8)

    train_sp_df = sp_df.iloc[:split_index]
    test_sp_df = sp_df.iloc[split_index:]

    print(f'Size of Training Set: {len(train_sp_df)}')
    print(f'Size of Test Set: {len(test_sp_df)}')

    # ===============================================================
    # Data exploration of training and test splits
    # ===============================================================

    print('Print first 20 rows of each column:\n', train_sp_df.head(20))

    print('\nData description:\n')
    with pd.option_context('display.max_columns', None, 'display.max_rows',
                           None):
        print(train_sp_df.describe())

    print('\nData description:\n')
    with pd.option_context('display.max_columns', None, 'display.max_rows',
                           None):
        print(test_sp_df.describe())

    # ===============================================================
    # Null value check and cleaning
    # ===============================================================

    print('\n Null values in training set: \n', train_sp_df.isnull().sum())
    train_sp_df_clean = train_sp_df.copy()  # 100% valid

    print('\n Null values in test set: \n', test_sp_df.isnull().sum())
    test_sp_df_clean = test_sp_df.copy()  # 100% valid

    # ===============================================================
    # Feature Selection
    # ===============================================================

    feature_columns = [
        'hours_studied', 'previous_scores', 'sleep_hours',
        'sample_question', 'extra_activities'
    ]

    print(feature_columns)

    # ===============================================================
    # Creation of Feature/Target/Test Numpy Arrays
    # ===============================================================

    train_feature_data = train_sp_df_clean[feature_columns]
    train_target_data = train_sp_df_clean['performance_index']

    train_X = train_feature_data.to_numpy()
    train_y = train_target_data.to_numpy()

    test_feature_data = test_sp_df_clean[feature_columns]
    test_target_data = test_sp_df_clean['performance_index']

    test_X = test_feature_data.to_numpy()
    test_y = test_target_data.to_numpy()

    # ===============================================================
    # Run and Set-up of Batch Gradient Descent with Demo Data to Verify
    # Functionality
    # ===============================================================

    np.random.seed(42)  # Ensure reproducibility
    n_samples = 100
    n_features = 5
    demo_X = np.random.randn(n_samples, n_features)
    weights_true = np.array([1.5, -2.0, 1.0, 1.5, 2])  # Known true weights
    demo_y = demo_X @ weights_true + 0.5

    print(demo_X.shape)

    # ===============================================================
    # Call to gradient descent on demo data
    # ===============================================================

    demo_b, demo_w, demo_cost_history = models.g_d_func(
        X=demo_X, y=demo_y, learning_rate=0.5, iterations=5, i_displayed=1)

    # ===============================================================
    # Demo Cost Graph
    # ===============================================================

    sns.lineplot(demo_cost_history)
    plt.title('Demo: Cost Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    plt.close()

    # ===============================================================
    # Performance Metric for Demo Data
    # ===============================================================

    functions.rmse_mae_r2(demo_X, demo_b, demo_w, demo_y)

    # ===============================================================
    # Training on Actual Training Data
    # ===============================================================

    train_b, train_w, train_cost_history = models.g_d_func(
        X=train_X, y=train_y, b_init=19.2, learning_rate=0.699,
        iterations=60, i_displayed=10)

    # ===============================================================
    # Training Cost Graph
    # ===============================================================

    sns.lineplot(train_cost_history)
    plt.title('G. D. Training Data: Cost Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    plt.close()

    # ===============================================================
    # Performance Metrics on Test Data After Training
    # ===============================================================

    functions.rmse_mae_r2(test_X, train_b, train_w, test_y)

    # ===============================================================
    # Calling LASSO Gradient Descent on Training Data
    # ===============================================================

    lasso_train_b, lasso_train_w, lasso_train_cost_history = \
        models.lasso_g_d_func(
            train_X,
            train_y,
            learning_rate=0.999999999,
            b_init=19.2,
            lambda_=0.00000000095,
            iterations=50
        )

    # ===============================================================
    # LASSO Training Cost Graph
    # ===============================================================

    sns.lineplot(lasso_train_cost_history)
    plt.title('LASSO Training Data: Cost Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    plt.close()

    # ===============================================================
    # Performance Metrics on Test Data Using LASSO Model
    # ===============================================================

    functions.rmse_mae_r2(test_X, lasso_train_b, lasso_train_w, test_y)

    # ===============================================================
    # Residual Analysis Graphs for LASSO Model on Test Data
    # ===============================================================

    y_hat = functions.predict(test_X, lasso_train_b, lasso_train_w)
    residuals = test_y - y_hat
    std_residual = residuals / np.std(residuals)

    plt.figure(figsize=(8, 5))
    plt.scatter(test_y, std_residual, alpha=0.35, color='black')
    plt.axhline(0, color='green')
    plt.axhline(2, color='red', linestyle='--')
    plt.axhline(-2, color='red', linestyle='--')
    plt.xlabel("True 'Y'")
    plt.ylabel('Standardized Residual')
    plt.title('Standardized Residual Plot — Test Set')
    plt.show()
    plt.close()

    plt.hist(residuals, bins=30, edgecolor='red', color='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()
    plt.close()

    # ===============================================================
    # Explanation of Residual Graphs and Feature Importance
    # ===============================================================

    """
    # Explanation of Residual Graphs
    - Standardized Residual Plot (Scatter Plot)
      - Lacks heteroscedasticity (contains no patterns)
        - Suggests that the errors of the model are random and not part of a
            systematic failure
    - Histogram of Residuals
      - Has a clear normal distribution (no skew)
      - Errors are mostly small and evenly distributed
    - Both graphs appear to represent healthy residuals
    """

    """
    # Feature Columns with the Greatest Impact
    - previous_scores
    - sleep_hours
    - hours_studied
    """
