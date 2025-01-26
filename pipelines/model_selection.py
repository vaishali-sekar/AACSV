<<<<<<< HEAD
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class VerboseCallback:
    """Custom callback for printing progress after each fit."""
    def __init__(self):
        self.fit_count = 0

    def on_fit(self):
        self.fit_count += 1
        print(f"Fit {self.fit_count} completed.")


def model_selection(X_train_scaled, y_train, X_test_scaled, y_test, n_iter=10):
    """
    Performs model selection with hyperparameter tuning using pre-scaled data.

    Args:
    - X_train_scaled (np.ndarray): Preprocessed and scaled feature matrix for training.
    - y_train (np.ndarray or pd.Series): Target variable for training.
    - X_test_scaled (np.ndarray): Preprocessed and scaled feature matrix for testing.
    - y_test (np.ndarray or pd.Series): Target variable for testing.
    - n_iter (int): Number of parameter settings sampled in RandomizedSearchCV.

    Returns:
    - best_model: The best model selected after hyperparameter tuning.
    """
    print("Running model selection with pre-scaled data...")

    # Initialize the progress tracker
    verbose_callback = VerboseCallback()

    # Define models
    rf_model = RandomForestRegressor(random_state=42)
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

    # Hyperparameter grids
    rf_param_distributions = {
        'n_estimators': [25, 50, 75],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    xgb_param_distributions = {
        'n_estimators': [25, 50, 75],
        #'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.2],
        #'subsample': [0.8, 1.0]c
    }

    # Models for randomized search
    models = [
        {"name": "RandomForest", "model": rf_model, "param_distributions": rf_param_distributions},
        {"name": "XGBoost", "model": xgb_model, "param_distributions": xgb_param_distributions}
    ]

    best_model = None
    best_mse = float("inf")
    best_name = ""

    for model in models:
        print(f"Starting hyperparameter tuning for {model['name']}...")
        with mlflow.start_run(run_name=model["name"]):
            randomized_search = RandomizedSearchCV(
                model["model"],
                model["param_distributions"],
                n_iter=n_iter,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,  # Enable verbose output for progress
                random_state=42
            )

            # Fit the randomized search
            randomized_search.fit(X_train_scaled, y_train)
            verbose_callback.on_fit()

            # Log parameters and metrics
            mlflow.log_params(randomized_search.best_params_)
            mse = -randomized_search.best_score_
            mlflow.log_metric("MSE", mse)

            if mse < best_mse:
                best_mse = mse
                best_model = randomized_search.best_estimator_
                best_name = model["name"]

    print(f"Best Model: {best_name} with MSE: {best_mse}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))
    print(f"Test MSE: {mse}")
    print(f"Test R-squared: {r2}")

=======
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class VerboseCallback:
    """Custom callback for printing progress after each fit."""
    def __init__(self):
        self.fit_count = 0

    def on_fit(self):
        self.fit_count += 1
        print(f"Fit {self.fit_count} completed.")


def model_selection(X_train_scaled, y_train, X_test_scaled, y_test, n_iter=10):
    """
    Performs model selection with hyperparameter tuning using pre-scaled data.

    Args:
    - X_train_scaled (np.ndarray): Preprocessed and scaled feature matrix for training.
    - y_train (np.ndarray or pd.Series): Target variable for training.
    - X_test_scaled (np.ndarray): Preprocessed and scaled feature matrix for testing.
    - y_test (np.ndarray or pd.Series): Target variable for testing.
    - n_iter (int): Number of parameter settings sampled in RandomizedSearchCV.

    Returns:
    - best_model: The best model selected after hyperparameter tuning.
    """
    print("Running model selection with pre-scaled data...")

    # Initialize the progress tracker
    verbose_callback = VerboseCallback()

    # Define models
    rf_model = RandomForestRegressor(random_state=42)
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

    # Hyperparameter grids
    rf_param_distributions = {
        'n_estimators': [25, 50, 75],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    xgb_param_distributions = {
        'n_estimators': [25, 50, 75],
        #'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.2],
        #'subsample': [0.8, 1.0]c
    }

    # Models for randomized search
    models = [
        {"name": "RandomForest", "model": rf_model, "param_distributions": rf_param_distributions},
        {"name": "XGBoost", "model": xgb_model, "param_distributions": xgb_param_distributions}
    ]

    best_model = None
    best_mse = float("inf")
    best_name = ""

    for model in models:
        print(f"Starting hyperparameter tuning for {model['name']}...")
        with mlflow.start_run(run_name=model["name"]):
            randomized_search = RandomizedSearchCV(
                model["model"],
                model["param_distributions"],
                n_iter=n_iter,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,  # Enable verbose output for progress
                random_state=42
            )

            # Fit the randomized search
            randomized_search.fit(X_train_scaled, y_train)
            verbose_callback.on_fit()

            # Log parameters and metrics
            mlflow.log_params(randomized_search.best_params_)
            mse = -randomized_search.best_score_
            mlflow.log_metric("MSE", mse)

            if mse < best_mse:
                best_mse = mse
                best_model = randomized_search.best_estimator_
                best_name = model["name"]

    print(f"Best Model: {best_name} with MSE: {best_mse}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))
    print(f"Test MSE: {mse}")
    print(f"Test R-squared: {r2}")

>>>>>>> d7e64b796ce19bc24494cf120a3bbe8eb59697e3
    return best_model