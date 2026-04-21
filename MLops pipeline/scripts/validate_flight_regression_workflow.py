import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "training" / "train_flight_price_regression_mlflow.py"
DATASET_PATH = PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv"

PYTHON_FILES_TO_COMPILE = [
    PROJECT_ROOT / "training" / "train_flight_price_regression_mlflow.py",
    PROJECT_ROOT / "airflow" / "dags" / "flight_price_regression_pipeline.py",
    PROJECT_ROOT / "flask_apps" / "common.py",
    PROJECT_ROOT / "flask_apps" / "flight_price_flask_app" / "app.py",
    PROJECT_ROOT / "streamlit" / "flight_price_app.py",
]


# Small CLI wrapper so Jenkins or a local check can choose a custom output folder.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the flight regression workflow in a Jenkins-friendly temp folder."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "jenkins_artifacts" / "flight_validation",
        help="Folder used for Jenkins validation outputs.",
    )
    return parser.parse_args()


# Fail early when a required file is missing or empty.
def ensure_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")

    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"Required file is empty: {path}")


# Compile a short list of critical Python files before launching the heavier workflow check.
def compile_python_sources():
    compiled_files = []

    for file_path in PYTHON_FILES_TO_COMPILE:
        ensure_file_exists(file_path)
        source_text = file_path.read_text(encoding="utf-8")
        compile(source_text, str(file_path), "exec")
        compiled_files.append(str(file_path.relative_to(PROJECT_ROOT)))

    return compiled_files


# Run the real training script in an isolated folder and validate the outputs it produces.
def run_training_validation(output_dir):
    # CI writes into its own runtime folder so this check can retrain the model
    # without touching the repo's normal joblib or MLflow folders.
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (output_dir / "runs" / run_label).resolve()
    validation_root = (run_root / "runtime").resolve()

    model_output = (validation_root / "joblib" / "flight_price_model_ci.joblib").resolve()
    metadata_output = (validation_root / "joblib" / "flight_price_model_ci_metadata.json").resolve()
    tracking_dir = (validation_root / "mlruns").resolve()
    logs_dir = (run_root / "logs").resolve()

    logs_dir.mkdir(parents=True, exist_ok=True)
    validation_root.mkdir(parents=True, exist_ok=True)

    # Reuse the same training script the repo depends on in normal runs.
    command = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--experiment-name",
        "voyage_analytics_flight_price_regression_jenkins",
        "--run-name",
        "jenkins_validation_run",
        "--tracking-dir",
        str(tracking_dir),
        "--model-output",
        str(model_output),
        "--metadata-output",
        str(metadata_output),
    ]

    # Capture stdout and stderr so Jenkins can archive them when something goes wrong.
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    (logs_dir / "training_stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs_dir / "training_stderr.log").write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Training validation failed. Check {logs_dir} for details."
        )

    # Confirm the training run created the same key outputs the repo expects.
    ensure_file_exists(model_output)
    ensure_file_exists(metadata_output)

    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    required_metadata_keys = [
        "run_id",
        "experiment_name",
        "metrics",
        "model_file",
        "tracking_uri",
    ]
    missing_keys = [key for key in required_metadata_keys if key not in metadata]

    if missing_keys:
        raise ValueError(f"Validation metadata is missing keys: {missing_keys}")

    metrics = metadata["metrics"]
    required_metric_keys = ["mae", "mse", "rmse", "r2"]
    missing_metric_keys = [key for key in required_metric_keys if key not in metrics]

    if missing_metric_keys:
        raise ValueError(f"Validation metrics are missing keys: {missing_metric_keys}")

    run_id = str(metadata["run_id"])
    run_directories = [
        meta_file.parent
        for meta_file in tracking_dir.rglob("meta.yaml")
        if meta_file.parent.name == run_id
    ]

    if not run_directories:
        raise FileNotFoundError(f"MLflow run directory was not found for run_id={run_id}")

    run_directory = run_directories[0]
    model_artifacts = list(run_directory.rglob("flight_price_model_ci.joblib"))
    metadata_artifacts = list(run_directory.rglob("flight_price_model_ci_metadata.json"))

    # Check that MLflow captured both the saved model and the metadata artifact.
    if not model_artifacts:
        raise FileNotFoundError("MLflow validation run is missing the saved joblib artifact.")

    if not metadata_artifacts:
        raise FileNotFoundError("MLflow validation run is missing the saved metadata artifact.")

    return {
        "run_folder": str(run_root.relative_to(PROJECT_ROOT)),
        "validation_root": str(validation_root.relative_to(PROJECT_ROOT)),
        "model_output": str(model_output.relative_to(PROJECT_ROOT)),
        "metadata_output": str(metadata_output.relative_to(PROJECT_ROOT)),
        "tracking_dir": str(tracking_dir.relative_to(PROJECT_ROOT)),
        "run_id": run_id,
        "experiment_name": str(metadata["experiment_name"]),
        "metrics": metrics,
        "mlflow_run_directory": str(run_directory.relative_to(PROJECT_ROOT)),
    }


# Entry point for local checks and Jenkins runs.
def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_file_exists(DATASET_PATH)

    compiled_files = compile_python_sources()
    training_summary = run_training_validation(output_dir)

    summary = {
        "dataset_file": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "compiled_files": compiled_files,
        "training_summary": training_summary,
    }

    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("Flight regression validation completed successfully.")
    print(f"Summary file: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Validated run ID: {training_summary['run_id']}")


if __name__ == "__main__":
    main()
