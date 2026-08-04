import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd


# -----------------------------
# 1. Resolve validation paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv"
    if (PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv").exists()
    else REPO_ROOT / "dataset" / "travel_capstone" / "flights.csv"
)
DATASET_PATH = Path(os.getenv("FLIGHT_DATASET_PATH", DEFAULT_DATASET_PATH))
MODEL_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model.joblib"
METADATA_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model_metadata.json"
LOCAL_TRAINING_DIR = Path(os.getenv("LOCAL_TRAINING_DIR", REPO_ROOT / "local_training"))
LOCAL_LATEST_MODEL_PATH = LOCAL_TRAINING_DIR / "outputs" / "models" / "flight_price_model_latest.joblib"
LOCAL_LATEST_METADATA_PATH = LOCAL_TRAINING_DIR / "outputs" / "metrics" / "flight_price_model_latest_metadata.json"

PYTHON_FILES_TO_COMPILE = [
    PROJECT_ROOT / "airflow" / "dags" / "flight_price_mlflow_training_pipeline.py",
    PROJECT_ROOT / "flask_apps" / "common.py",
    PROJECT_ROOT / "flask_apps" / "flight_price_flask_app" / "app.py",
    LOCAL_TRAINING_DIR / "train_flight_price.py",
    PROJECT_ROOT / "streamlit" / "flight_price_app.py",
]

REQUIRED_METADATA_KEYS = [
    "project_name",
    "model_name",
    "selected_model",
    "version_id",
    "target_column",
    "model_columns",
    "split_strategy",
    "strict_validation",
    "tuning_method",
    "dataset_info",
    "group_holdout_info",
    "best_params",
    "metrics",
]

REQUIRED_FEATURE_COLUMNS = [
    "time",
    "year",
    "month",
    "day",
    "from",
    "to",
    "flightType",
    "agency",
]


def display_path(path):
    # -----------------------------
    # 2. Print readable paths
    # -----------------------------
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def parse_args():
    # -----------------------------
    # 3. Read validation options
    # -----------------------------
    parser = argparse.ArgumentParser(  # keep Jenkins and local checks configurable.
        description="Validate the promoted local MLflow flight price model without retraining it."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "jenkins_artifacts" / "flight_validation",
        help="Folder used for validation summary output.",
    )
    return parser.parse_args()


def ensure_file_exists(path, label):
    # -----------------------------
    # 4. Validate required files
    # -----------------------------
    if not path.exists():  # fail before loading when a required artifact is missing.
        # Stop execution with a clear error when the input is invalid.
        raise FileNotFoundError(f"{label} was not found: {path}")

    if path.is_file() and path.stat().st_size == 0:  # empty files should never pass validation.
        # Stop execution with a clear error when the input is invalid.
        raise ValueError(f"{label} is empty: {path}")


def compile_python_sources():
    # -----------------------------
    # 5. Compile runtime Python files
    # -----------------------------
    compiled_files = []  # store compiled file names for the final summary.

    for file_path in PYTHON_FILES_TO_COMPILE:  # compile only runtime-critical files.
        ensure_file_exists(file_path, "Python source file")
        source_text = file_path.read_text(encoding="utf-8")  # read source exactly as Jenkins sees it.
        compile(source_text, str(file_path), "exec")  # catch syntax errors without running the app.
        compiled_files.append(display_path(file_path))  # keep summary paths short.

    return compiled_files


def load_metadata():
    # -----------------------------
    # 6. Validate model metadata
    # -----------------------------
    ensure_file_exists(METADATA_PATH, "Model metadata file")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))  # parse the promoted local training metadata.
    missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]  # check expected handoff fields.

    if missing_keys:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError(f"Model metadata is missing keys: {missing_keys}")

    metrics = metadata["metrics"]  # metric block proves which run produced this model.
    required_metric_keys = [
        "time_mae",
        "time_rmse",
        "time_r2_score",
        "group_mae",
        "group_rmse",
        "group_r2_score",
        "selection_metric",
    ]  # minimum metrics needed for review.
    missing_metric_keys = [key for key in required_metric_keys if key not in metrics]

    if missing_metric_keys:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError(f"Model metrics are missing keys: {missing_metric_keys}")

    if metadata["model_columns"] != REQUIRED_FEATURE_COLUMNS:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError("Metadata feature columns do not match the local prediction input format.")

    return metadata


def inspect_optional_local_training_outputs(serving_metadata):
    # Local training outputs are generated evidence and are not included in a fresh clone.
    paths = [LOCAL_LATEST_MODEL_PATH, LOCAL_LATEST_METADATA_PATH]
    available_files = [display_path(path) for path in paths if path.exists() and path.stat().st_size > 0]
    missing_files = [display_path(path) for path in paths if not path.exists() or path.stat().st_size == 0]

    result = {
        "status": "available" if not missing_files else "partial" if available_files else "not_available",
        "available_files": available_files,
        "missing_files": missing_files,
        "required_for_ci": False,
    }

    if LOCAL_LATEST_METADATA_PATH.exists() and LOCAL_LATEST_METADATA_PATH.stat().st_size > 0:
        try:
            local_metadata = json.loads(LOCAL_LATEST_METADATA_PATH.read_text(encoding="utf-8"))
            matches_serving_version = local_metadata.get("version_id") == serving_metadata.get("version_id")
            result.update(
                {
                    "local_selected_model": local_metadata.get("selected_model"),
                    "local_version_id": local_metadata.get("version_id"),
                    "matches_serving_version": matches_serving_version,
                }
            )
            if not matches_serving_version:
                result["status"] = "stale"
        except (OSError, json.JSONDecodeError) as exc:
            result["status"] = "invalid"
            result["metadata_error"] = str(exc)

    return result


def build_sample_input(flights_df):
    # -----------------------------
    # 7. Build one realistic prediction row
    # -----------------------------
    sample_row = flights_df.iloc[0]  # use one real row so categories match the dataset.
    travel_date = pd.to_datetime(sample_row["date"])  # extract date parts in the same way as the app.

    return pd.DataFrame(
        [
            {
                "time": float(sample_row["time"]),  # model expects flight duration as numeric input.
                "year": int(travel_date.year),  # year comes from the selected travel date.
                "month": int(travel_date.month),  # month comes from the selected travel date.
                "day": int(travel_date.day),  # day comes from the selected travel date.
                "from": sample_row["from"],  # raw origin city is encoded inside the saved pipeline.
                "to": sample_row["to"],  # raw destination city is encoded inside the saved pipeline.
                "flightType": sample_row["flightType"],  # raw cabin type is encoded inside the saved pipeline.
                "agency": sample_row["agency"],  # raw agency name is encoded inside the saved pipeline.
            }
        ],
        columns=REQUIRED_FEATURE_COLUMNS,
    )


def validate_promoted_model():
    # -----------------------------
    # 8. Load and test promoted model
    # -----------------------------
    ensure_file_exists(DATASET_PATH, "Flight dataset")
    ensure_file_exists(MODEL_PATH, "Promoted model file")

    metadata = load_metadata()
    model = joblib.load(MODEL_PATH)  # load the exact promoted joblib used by API and Streamlit.
    flights_df = pd.read_csv(DATASET_PATH)  # read source data only for a realistic smoke-test row.
    sample_input = build_sample_input(flights_df)  # create one raw input row for the saved pipeline.
    prediction = float(model.predict(sample_input)[0])  # confirm the model can produce a numeric prediction.

    if prediction <= 0:
        # Stop execution with a clear error when the input is invalid.
        raise ValueError(f"Prediction should be positive, got {prediction}")

    return {
        "model_file": display_path(MODEL_PATH),
        "metadata_file": display_path(METADATA_PATH),
        "model_type": type(model).__name__,
        "selected_model": metadata["selected_model"],
        "version_id": metadata["version_id"],
        "strict_validation": metadata["strict_validation"],
        "tuning_method": metadata["tuning_method"],
        "input_columns": list(sample_input.columns),
        "sample_prediction": round(prediction, 2),
        "actual_sample_price": float(flights_df.iloc[0]["price"]),
        "metrics": metadata["metrics"],
    }


def main():
    # -----------------------------
    # 9. Run full validation workflow
    # -----------------------------
    args = parse_args()
    output_dir = args.output_dir.resolve()  # normalize the output path for local and Jenkins runs.
    output_dir.mkdir(parents=True, exist_ok=True)  # make sure the summary folder exists.

    compiled_files = compile_python_sources()
    model_summary = validate_promoted_model()
    optional_training_outputs = inspect_optional_local_training_outputs(model_summary)

    summary = {
        "dataset_file": display_path(DATASET_PATH),
        "compiled_files": compiled_files,
        "promoted_model_summary": model_summary,
        "optional_local_training_outputs": optional_training_outputs,
    }

    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")  # write evidence for CI artifacts.

    print("Flight regression local MLflow model validation completed successfully.")
    print(f"Summary file: {display_path(summary_path)}")
    print(f"Validated model: {model_summary['selected_model']}")
    print(f"Strict validation: {model_summary['strict_validation']}")
    print(f"Sample prediction: {model_summary['sample_prediction']}")
    print(f"Optional local training evidence: {optional_training_outputs['status']}")


if __name__ == "__main__":
    # -----------------------------
    # 10. Script entry point
    # -----------------------------
    main()
