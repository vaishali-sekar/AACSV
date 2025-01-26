<<<<<<< HEAD
from pipelines.data_loader import data_loader
from pipelines.data_preprocessor import data_preprocessor
from pipelines.train_test_split import split_data
from pipelines.model_selection import model_selection
from pipelines.data_scaler import data_scaler
from pipelines.drift import check_drift_in_pipeline
import pandas as pd
import pickle

if __name__ == "__main__":
    # Step 1: Load Data
    file_path = "Bengaluru_House_Data.csv"
    production_file_path = "production_dataset.csv"  # Path to the new dataset
    combined_file_path = "combined_data.csv"

    # Check for Drift
    report_output_path = "data_drift_report.html"
    drift_score = check_drift_in_pipeline(file_path, production_file_path, report_output_path)

    # Validate drift_score
    if drift_score is None:
        print("Drift detection failed or returned no score. Exiting.")
    else:
        # Threshold for retraining
        DRIFT_THRESHOLD = 0.6

        if drift_score > DRIFT_THRESHOLD:
            print(f"Drift score ({drift_score}) exceeds threshold ({DRIFT_THRESHOLD}). Retraining model...")

            # Combine old and new datasets
            print("Combining old and new datasets for retraining...")
            old_data = pd.read_csv(file_path)
            new_data = pd.read_csv(production_file_path)
            combined_data = pd.concat([old_data, new_data], ignore_index=True)

            # Save the combined dataset
            combined_data.to_csv(combined_file_path, index=False)

            # Reload combined data
            data = data_loader(combined_file_path)

            # Step 2: Preprocess Data
            X, y, feature_names = data_preprocessor(data)

            # Step 3: Train-Test Split
            X_train, X_test, y_train, y_test = split_data(X, y)

            # Step 4: One hot encoded data
            X_train_scaled, X_test_scaled, scaler = data_scaler(X_train, X_test)

            # Step 5: Model Selection
            best_model = model_selection(X_train_scaled, y_train, X_test_scaled, y_test)

            # Save the best model as a pickle file
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)

            with open("features.pkl", "wb") as f:
                pickle.dump(feature_names, f)

            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

            print("Model retraining complete. Files saved.")
        else:
            print(f"Drift score ({drift_score}) is below threshold ({DRIFT_THRESHOLD}). No retraining required.")
=======
from pipelines.data_loader import data_loader
from pipelines.data_preprocessor import data_preprocessor
from pipelines.train_test_split import split_data
from pipelines.model_selection import model_selection
from pipelines.data_scaler import data_scaler
from pipelines.drift import check_drift_in_pipeline
import pandas as pd
import pickle

if __name__ == "__main__":
    # Step 1: Load Data
    file_path = "Bengaluru_House_Data.csv"
    production_file_path = "production_dataset.csv"  # Path to the new dataset
    combined_file_path = "combined_data.csv"

    # Check for Drift
    report_output_path = "data_drift_report.html"
    drift_score = check_drift_in_pipeline(file_path, production_file_path, report_output_path)

    # Validate drift_score
    if drift_score is None:
        print("Drift detection failed or returned no score. Exiting.")
    else:
        # Threshold for retraining
        DRIFT_THRESHOLD = 0.6

        if drift_score > DRIFT_THRESHOLD:
            print(f"Drift score ({drift_score}) exceeds threshold ({DRIFT_THRESHOLD}). Retraining model...")

            # Combine old and new datasets
            print("Combining old and new datasets for retraining...")
            old_data = pd.read_csv(file_path)
            new_data = pd.read_csv(production_file_path)
            combined_data = pd.concat([old_data, new_data], ignore_index=True)

            # Save the combined dataset
            combined_data.to_csv(combined_file_path, index=False)

            # Reload combined data
            data = data_loader(combined_file_path)

            # Step 2: Preprocess Data
            X, y, feature_names = data_preprocessor(data)

            # Step 3: Train-Test Split
            X_train, X_test, y_train, y_test = split_data(X, y)

            # Step 4: One hot encoded data
            X_train_scaled, X_test_scaled, scaler = data_scaler(X_train, X_test)

            # Step 5: Model Selection
            best_model = model_selection(X_train_scaled, y_train, X_test_scaled, y_test)

            # Save the best model as a pickle file
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)

            with open("features.pkl", "wb") as f:
                pickle.dump(feature_names, f)

            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

            print("Model retraining complete. Files saved.")
        else:
            print(f"Drift score ({drift_score}) is below threshold ({DRIFT_THRESHOLD}). No retraining required.")
>>>>>>> d7e64b796ce19bc24494cf120a3bbe8eb59697e3
