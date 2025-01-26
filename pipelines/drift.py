<<<<<<< HEAD
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# Load the training and production datasets
def load_datasets(train_file_path, production_file_path):
    train_data = pd.read_csv(train_file_path)
    production_data = pd.read_csv(production_file_path)
    return train_data, production_data

# Generate a drift report
def generate_drift_report(train_data, production_data, output_path):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_data, current_data=production_data)
    report.save_html(output_path)

    # Extract drift score
    drift_metric = report.as_dict()
    drift_score = drift_metric["metrics"][0]["result"]["dataset_drift"]
    return drift_score

# Integration in the pipeline
def check_drift_in_pipeline(train_file_path, production_file_path, report_output_path):
    print("Loading datasets for drift detection...")
    train_data, production_data = load_datasets(train_file_path, production_file_path)

    print("Generating drift report...")
    try:
        drift_score = generate_drift_report(train_data, production_data, report_output_path)
        print(f"Drift report saved to {report_output_path}")
        return drift_score
    except Exception as e:
        print(f"Error during drift detection: {e}")
        return None
=======
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# Load the training and production datasets
def load_datasets(train_file_path, production_file_path):
    train_data = pd.read_csv(train_file_path)
    production_data = pd.read_csv(production_file_path)
    return train_data, production_data

# Generate a drift report
def generate_drift_report(train_data, production_data, output_path):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_data, current_data=production_data)
    report.save_html(output_path)

    # Extract drift score
    drift_metric = report.as_dict()
    drift_score = drift_metric["metrics"][0]["result"]["dataset_drift"]
    return drift_score

# Integration in the pipeline
def check_drift_in_pipeline(train_file_path, production_file_path, report_output_path):
    print("Loading datasets for drift detection...")
    train_data, production_data = load_datasets(train_file_path, production_file_path)

    print("Generating drift report...")
    try:
        drift_score = generate_drift_report(train_data, production_data, report_output_path)
        print(f"Drift report saved to {report_output_path}")
        return drift_score
    except Exception as e:
        print(f"Error during drift detection: {e}")
        return None
>>>>>>> d7e64b796ce19bc24494cf120a3bbe8eb59697e3
