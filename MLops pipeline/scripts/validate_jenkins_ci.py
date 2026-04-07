import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd


# -----------------------------
# 1. Resolve validation paths
# -----------------------------
# This validator checks the project as Jenkins would see it inside the pipeline folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
LOCAL_TRAINING_DIR = Path(os.getenv("LOCAL_TRAINING_DIR", REPO_ROOT / "local_training"))

REQUIRED_FILES = [
    "requirements.txt",
    "requirements-airflow.txt",
    "requirements-mlflow.txt",
    "Dockerfile",
    "Dockerfile.streamlit",
    "Dockerfile.mlops",
    "Dockerfile.mlflow",
    "docker-compose.yml",
    "Jenkinsfile",
    "dataset/travel_capstone/flights.csv",
    "joblib files/flight_price_model.joblib",
    "joblib files/flight_price_model_metadata.json",
    "joblib files/hotel_recommender_latest.joblib",
    "joblib files/hotel_recommender_latest_metadata.json",
    "joblib files/gender_classifier_model_latest.joblib",
    "joblib files/gender_classifier_model_latest_metadata.json",
    "flask_apps/flight_price_flask_app/app.py",
    "flask_apps/hotel_recommendation_flask_app/app.py",
    "flask_apps/gender_classification_flask_app/app.py",
    "streamlit/flight_price_app.py",
    "streamlit/hotel_recommendation_app.py",
    "streamlit/gender_classification_app.py",
    "airflow/dags/flight_price_mlflow_training_pipeline.py",
    "local_training/train_flight_price.py",
    "local_training/requirements.txt",
    "local_training/outputs/models/flight_price_model_latest.joblib",
    "local_training/outputs/metrics/flight_price_model_latest_metadata.json",
    "scripts/validate_flight_regression_workflow.py",
    "scripts/check_flight_dataset_drift.py",
]

PYTHON_FILES = [
    "flask_apps/common.py",
    "flask_apps/flight_price_flask_app/app.py",
    "flask_apps/hotel_recommendation_flask_app/app.py",
    "flask_apps/gender_classification_flask_app/app.py",
    "streamlit/flight_price_app.py",
    "streamlit/hotel_recommendation_app.py",
    "streamlit/gender_classification_app.py",
    "airflow/dags/flight_price_mlflow_training_pipeline.py",
    "local_training/train_flight_price.py",
    "scripts/validate_flight_regression_workflow.py",
    "scripts/check_flight_dataset_drift.py",
]

MLFLOW_OUTPUTS = [
    "local_training/outputs/models/flight_price_model_latest.joblib",
    "local_training/outputs/metrics/flight_price_model_latest_metadata.json",
    "jenkins_artifacts/data_drift/flight_dataset_drift_summary.json",
]

FLIGHT_FEATURE_COLUMNS = [
    "time",
    "year",
    "month",
    "day",
    "from",
    "to",
    "flightType",
    "agency",
]


def parse_args():
    # -----------------------------
    # 2. Read validation options
    # -----------------------------
    # Output directory is configurable so Jenkins and manual runs can keep separate reports.
    parser = argparse.ArgumentParser(description="Run Jenkins-style CI checks for the Voyage Analytics project.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "jenkins_artifacts" / "ci_validation",
        help="Folder where the CI summary JSON should be written.",
    )
    return parser.parse_args()


def project_path(relative_path):
    # -----------------------------
    # 3. Resolve project-relative file path
    # -----------------------------
    # Route file lookups to the correct root because some assets live outside the pipeline folder.
    normalized_path = str(relative_path).replace("\\", "/")

    if normalized_path.startswith("local_training/"):
        return LOCAL_TRAINING_DIR / normalized_path.removeprefix("local_training/")

    if normalized_path.startswith("dataset/"):
        return REPO_ROOT / normalized_path

    return PROJECT_ROOT / relative_path


def display_path(path):
    # -----------------------------
    # 4. Print readable paths
    # -----------------------------
    # Prefer relative paths in summaries so reports stay readable across machines.
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def require_file(relative_path):
    # -----------------------------
    # 5. Validate required file
    # -----------------------------
    # Missing or empty files should fail the pipeline immediately.
    path = project_path(relative_path)
    if not path.exists():
        # Stop execution with a clear error when the input is invalid.
        raise FileNotFoundError(f"Required file is missing: {relative_path}")
    if path.is_file() and path.stat().st_size == 0:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError(f"Required file is empty: {relative_path}")
    return path


def check_required_files():
    # -----------------------------
    # 6. Check expected project files
    # -----------------------------
    # Record file sizes too, so a zero-byte artifact is easier to spot later.
    checked = []
    # Loop through each item that needs the same handling.
    for relative_path in REQUIRED_FILES:
        path = require_file(relative_path)
        checked.append({"path": relative_path, "size_bytes": path.stat().st_size})
    return checked


def check_python_syntax():
    # -----------------------------
    # 7. Compile runtime Python files
    # -----------------------------
    # Compile source code without executing it to catch obvious syntax mistakes fast.
    compiled = []
    # Loop through each item that needs the same handling.
    for relative_path in PYTHON_FILES:
        path = require_file(relative_path)
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        compiled.append(relative_path)
    return compiled


def validate_flight_model():
    # -----------------------------
    # 8. Smoke test flight model
    # -----------------------------
    # Load the serving model and test one realistic prediction from the dataset.
    model = joblib.load(require_file("joblib files/flight_price_model.joblib"))
    metadata = json.loads(require_file("joblib files/flight_price_model_metadata.json").read_text(encoding="utf-8"))
    flights = pd.read_csv(require_file("dataset/travel_capstone/flights.csv"))

    row = flights.iloc[0]
    travel_date = pd.to_datetime(row["date"])
    sample = pd.DataFrame(
        [
            {
                "time": float(row["time"]),
                "year": int(travel_date.year),
                "month": int(travel_date.month),
                "day": int(travel_date.day),
                "from": row["from"],
                "to": row["to"],
                "flightType": row["flightType"],
                "agency": row["agency"],
            }
        ],
        columns=FLIGHT_FEATURE_COLUMNS,
    )

    prediction = float(model.predict(sample)[0])
    if prediction <= 0:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError(f"Flight model returned an invalid prediction: {prediction}")

    expected_columns = metadata.get("model_columns") or metadata.get("raw_feature_columns")
    if expected_columns != FLIGHT_FEATURE_COLUMNS:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError("Flight model metadata feature columns do not match the app input format.")

    metrics = metadata.get("metrics", {})

    return {
        "model_type": type(model).__name__,
        "selected_model": metadata.get("selected_model") or metadata.get("model_name"),
        "sample_prediction": round(prediction, 2),
        "version_id": metadata.get("version_id"),
        "strict_validation": metadata.get("strict_validation"),
        "group_rmse": metrics.get("group_rmse"),
        "time_rmse": metrics.get("time_rmse") or metrics.get("rmse"),
    }


def validate_hotel_model():
    # -----------------------------
    # 9. Smoke test hotel recommender
    # -----------------------------
    # The hotel recommender is a saved object bundle, so verify its core pieces exist.
    recommender = joblib.load(require_file("joblib files/hotel_recommender_latest.joblib"))
    metadata = json.loads(require_file("joblib files/hotel_recommender_latest_metadata.json").read_text(encoding="utf-8"))

    required_keys = ["popular_hotels", "hotel_content", "user_item_matrix", "best_model_name"]
    missing = [key for key in required_keys if key not in recommender]
    if missing:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError(f"Hotel recommender is missing keys: {missing}")

    popular_hotels = recommender["popular_hotels"]
    if len(popular_hotels) == 0:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError("Hotel recommender has no popular hotel rows.")

    return {
        "best_model_name": recommender["best_model_name"],
        "popular_hotel_count": int(len(popular_hotels)),
        "metadata_model_name": metadata.get("model_name") or metadata.get("selected_model"),
    }


def validate_gender_model():
    # -----------------------------
    # 10. Smoke test gender classifier
    # -----------------------------
    # Use two common names to confirm the classifier still returns labels.
    model = joblib.load(require_file("joblib files/gender_classifier_model_latest.joblib"))
    metadata = json.loads(require_file("joblib files/gender_classifier_model_latest_metadata.json").read_text(encoding="utf-8"))

    predictions = list(model.predict(["Amit", "Priya"]))
    if len(predictions) != 2:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError("Gender classifier did not return two predictions.")

    return {
        "model_type": type(model).__name__,
        "sample_predictions": {"Amit": predictions[0], "Priya": predictions[1]},
        "metadata_model_name": metadata.get("model_name") or metadata.get("selected_model"),
    }


def validate_mlflow_airflow_outputs():
    # -----------------------------
    # 11. Validate training and Airflow outputs
    # -----------------------------
    # These outputs prove the local training, drift check, and reporting flow already ran.
    outputs = []
    # Loop through each item that needs the same handling.
    for relative_path in MLFLOW_OUTPUTS:
        path = require_file(relative_path)
        outputs.append({"path": relative_path, "size_bytes": path.stat().st_size})

    drift_summary = json.loads(require_file("jenkins_artifacts/data_drift/flight_dataset_drift_summary.json").read_text())
    local_metadata = json.loads(
        require_file("local_training/outputs/metrics/flight_price_model_latest_metadata.json").read_text(encoding="utf-8")
    )

    metrics_dir = project_path("local_training/outputs/metrics")
    feature_importance_files = sorted(
        metrics_dir.glob("flight_price_feature_importance_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    worst_prediction_files = sorted(
        metrics_dir.glob("flight_price_worst_predictions_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )

    if not feature_importance_files:
        # Stop execution with a clear error when the input is invalid.
        raise FileNotFoundError("No local training feature importance report was found.")

    if not worst_prediction_files:
        # Stop execution with a clear error when the input is invalid.
        raise FileNotFoundError("No local training worst prediction report was found.")

    return {
        "files": outputs,
        "drift_detected": drift_summary.get("drift_detected"),
        "selected_mlflow_model": local_metadata.get("selected_model"),
        "selected_group_rmse": local_metadata.get("metrics", {}).get("group_rmse"),
        "latest_feature_importance_report": display_path(feature_importance_files[-1]),
        "latest_worst_prediction_report": display_path(worst_prediction_files[-1]),
    }


def main():
    # -----------------------------
    # 12. Run full CI validation
    # -----------------------------
    # Build one summary JSON so Jenkins can archive a single easy-to-read report.
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "required_files": check_required_files(),
        "compiled_python_files": check_python_syntax(),
        "models": {
            "flight_price": validate_flight_model(),
            "hotel_recommendation": validate_hotel_model(),
            "gender_classification": validate_gender_model(),
        },
        "mlflow_airflow_outputs": validate_mlflow_airflow_outputs(),
    }

    summary_path = output_dir / "jenkins_ci_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("Jenkins CI validation completed successfully.")
    print(f"Summary file: {display_path(summary_path)}")
    print(f"Flight model: {summary['models']['flight_price']['selected_model']}")
    print(f"Hotel model: {summary['models']['hotel_recommendation']['best_model_name']}")
    print(f"Gender samples: {summary['models']['gender_classification']['sample_predictions']}")
    print(f"MLflow model: {summary['mlflow_airflow_outputs']['selected_mlflow_model']}")
    print(f"Group RMSE: {summary['mlflow_airflow_outputs']['selected_group_rmse']}")


if __name__ == "__main__":
    main()
