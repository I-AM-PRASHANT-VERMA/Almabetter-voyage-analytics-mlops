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
    "scripts/validate_flight_regression_workflow.py",
    "scripts/check_flight_dataset_drift.py",
    "scripts/assess_flight_retraining.py",
    "scripts/trigger_jenkins_cd.py",
    "jenkins/init-voyage.groovy",
    "jenkins/azure-aks-cd-pipeline.groovy",
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
    "scripts/assess_flight_retraining.py",
    "scripts/trigger_jenkins_cd.py",
]

OPTIONAL_TRAINING_OUTPUTS = [
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


def inspect_optional_training_outputs(serving_model_summary):
    # -----------------------------
    # 11. Inspect optional training and Airflow outputs
    # -----------------------------
    # These generated files are useful evidence when present, but a fresh clone should not require them.
    available_files = []
    missing_files = []
    errors = []

    for relative_path in OPTIONAL_TRAINING_OUTPUTS:
        path = project_path(relative_path)
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            available_files.append({"path": relative_path, "size_bytes": path.stat().st_size})
        else:
            missing_files.append(relative_path)

    drift_summary = {}
    drift_path = project_path("jenkins_artifacts/data_drift/flight_dataset_drift_summary.json")
    if drift_path.exists() and drift_path.stat().st_size > 0:
        try:
            drift_summary = json.loads(drift_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid drift summary: {exc}")

    local_metadata = {}
    local_metadata_path = project_path("local_training/outputs/metrics/flight_price_model_latest_metadata.json")
    if local_metadata_path.exists() and local_metadata_path.stat().st_size > 0:
        try:
            local_metadata = json.loads(local_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid local training metadata: {exc}")

    metrics_dir = project_path("local_training/outputs/metrics")
    feature_importance_files = (
        sorted(metrics_dir.glob("flight_price_feature_importance_*.csv"), key=lambda path: path.stat().st_mtime)
        if metrics_dir.exists()
        else []
    )
    worst_prediction_files = (
        sorted(metrics_dir.glob("flight_price_worst_predictions_*.csv"), key=lambda path: path.stat().st_mtime)
        if metrics_dir.exists()
        else []
    )

    status = "available" if not missing_files else "partial" if available_files else "not_available"
    if errors:
        status = "invalid"
    matches_serving_version = (
        local_metadata.get("version_id") == serving_model_summary.get("version_id") if local_metadata else None
    )
    if not errors and matches_serving_version is False:
        status = "stale"

    return {
        "status": status,
        "required_for_ci": False,
        "available_files": available_files,
        "missing_files": missing_files,
        "errors": errors,
        "drift_detected": drift_summary.get("drift_detected"),
        "local_selected_model": local_metadata.get("selected_model"),
        "local_version_id": local_metadata.get("version_id"),
        "serving_version_id": serving_model_summary.get("version_id"),
        "matches_serving_version": matches_serving_version,
        "selected_group_rmse": local_metadata.get("metrics", {}).get("group_rmse"),
        "latest_feature_importance_report": display_path(feature_importance_files[-1]) if feature_importance_files else None,
        "latest_worst_prediction_report": display_path(worst_prediction_files[-1]) if worst_prediction_files else None,
    }


def validate_automation_config():
    compose_text = require_file("docker-compose.yml").read_text(encoding="utf-8")
    bootstrap_text = require_file("jenkins/init-voyage.groovy").read_text(encoding="utf-8")
    cd_text = require_file("jenkins/azure-aks-cd-pipeline.groovy").read_text(encoding="utf-8")

    required_markers = {
        "azure_switch_defaults_false": "VOYAGE_AZURE_DEPLOYMENT_ENABLED:-false" in compose_text,
        "ci_calls_gated_cd_helper": "trigger_jenkins_cd.py" in bootstrap_text,
        "cd_reads_pipeline_from_scm": "CpsScmFlowDefinition" in bootstrap_text,
        "cd_checks_subscription_state": "ACCOUNT_STATE" in cd_text,
        "cd_can_start_existing_aks": "az aks start" in cd_text,
        "validation_cleanup_is_gated": "if (params.DEPLOY_TO_AKS)" in cd_text,
    }
    missing = [name for name, present in required_markers.items() if not present]
    if missing:
        raise ValueError(f"Automation safety markers are missing: {missing}")
    return required_markers


def main():
    # -----------------------------
    # 12. Run full CI validation
    # -----------------------------
    # Build one summary JSON so Jenkins can archive a single easy-to-read report.
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flight_model_summary = validate_flight_model()

    summary = {
        "required_files": check_required_files(),
        "compiled_python_files": check_python_syntax(),
        "models": {
            "flight_price": flight_model_summary,
            "hotel_recommendation": validate_hotel_model(),
            "gender_classification": validate_gender_model(),
        },
        "automation": validate_automation_config(),
        "optional_training_evidence": inspect_optional_training_outputs(flight_model_summary),
    }

    summary_path = output_dir / "jenkins_ci_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("Jenkins CI validation completed successfully.")
    print(f"Summary file: {display_path(summary_path)}")
    print(f"Flight model: {summary['models']['flight_price']['selected_model']}")
    print(f"Hotel model: {summary['models']['hotel_recommendation']['best_model_name']}")
    print(f"Gender samples: {summary['models']['gender_classification']['sample_predictions']}")
    print(f"Optional training evidence: {summary['optional_training_evidence']['status']}")


if __name__ == "__main__":
    main()
