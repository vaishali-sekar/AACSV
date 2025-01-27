import mlflow
from mlflow import log_metric, log_param, log_artifact
from pipelines.data_loader import data_loader
from pipelines.data_preprocessor import data_preprocessor
from pipelines.train_test_split import split_data
from pipelines.model_selection import model_selection
from pipelines.data_scaler import data_scaler
from pipelines.drift import check_drift_in_pipeline
import pandas as pd
import pickle
import os

if __name__ == "__main__":
    # Initialize MLflow
    mlflow.set_experiment("House Price Drift Detection and Retraining")

    with mlflow.start_run():
        # Step 1: Load Data
        file_path = "Bengaluru_House_Data.csv"
        production_file_path = "production_dataset.csv"  # Path to the new dataset
        combined_file_path = "combined_data.csv"

        # Log data file paths
        mlflow.log_param("file_path", file_path)
        mlflow.log_param("production_file_path", production_file_path)

        # Check for Drift
        report_output_path = "data_drift_report.html"
        drift_score = check_drift_in_pipeline(file_path, production_file_path, report_output_path)

        # Log the drift report artifact
        if os.path.exists(report_output_path):
            mlflow.log_artifact(report_output_path, artifact_path="drift_reports")

        # Validate drift_score
        if drift_score is None:
            print("Drift detection failed or returned no score. Exiting.")
            mlflow.log_metric("drift_score", 0.0)
            mlflow.end_run(status="FAILED")
        else:
            mlflow.log_metric("drift_score", drift_score)

            # Threshold for retraining
            DRIFT_THRESHOLD = 0.6
            mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)

            if drift_score > DRIFT_THRESHOLD:
                print(f"Drift score ({drift_score}) exceeds threshold ({DRIFT_THRESHOLD}). Retraining model...")
                mlflow.log_metric("retrain_required", 1)

                # Combine old and new datasets
                print("Combining old and new datasets for retraining...")
                old_data = pd.read_csv(file_path)
                new_data = pd.read_csv(production_file_path)
                combined_data = pd.concat([old_data, new_data], ignore_index=True)

                # Save the combined dataset
                combined_data.to_csv(combined_file_path, index=False)
                mlflow.log_artifact(combined_file_path, artifact_path="datasets")

                # Reload combined data
                data = data_loader(combined_file_path)

                # Step 2: Preprocess Data
                X, y, feature_names = data_preprocessor(data)
                mlflow.log_param("num_features", len(feature_names))

                # Step 3: Train-Test Split
                X_train, X_test, y_train, y_test = split_data(X, y)
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))

                # Step 4: One hot encoded data
                X_train_scaled, X_test_scaled, scaler = data_scaler(X_train, X_test)

                # Step 5: Model Selection
                best_model = model_selection(X_train_scaled, y_train, X_test_scaled, y_test)

                # Save the best model as a pickle file
                model_path = "best_model.pkl"
                features_path = "features.pkl"
                scaler_path = "scaler.pkl"

                with open(model_path, "wb") as f:
                    pickle.dump(best_model, f)

                with open(features_path, "wb") as f:
                    pickle.dump(feature_names, f)

                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)

                # Log artifacts
                mlflow.log_artifact(model_path, artifact_path="models")
                mlflow.log_artifact(features_path, artifact_path="models")
                mlflow.log_artifact(scaler_path, artifact_path="models")

                print("Model retraining complete. Files saved.")
                mlflow.end_run(status="FINISHED")
            else:
                print(f"Drift score ({drift_score}) is below threshold ({DRIFT_THRESHOLD}). No retraining required.")
                mlflow.log_metric("retrain_required", 0)
                mlflow.end_run(status="FINISHED")
