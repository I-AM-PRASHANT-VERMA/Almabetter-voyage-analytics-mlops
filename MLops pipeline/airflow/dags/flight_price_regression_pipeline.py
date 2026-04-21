from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv"
TRAINING_SCRIPT_PATH = (
    PROJECT_ROOT / "training" / "train_flight_price_regression_mlflow.py"
)
MODEL_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model.joblib"
METADATA_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model_metadata.json"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"


# Run a project command from the repo root and surface stdout and stderr in the task log.
def run_command(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


@dag(
    dag_id="flight_price_regression_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "flight-price", "regression"],
    description="Train and validate the flight price regression model with MLflow tracking.",
)
def flight_price_regression_pipeline():
    # Airflow stays focused on orchestration here. The actual model logic still
    # lives in the training script so local runs and DAG runs behave the same way.
    @task(task_id="check_data_availability")
    def check_data_availability() -> str:
        # Stop the DAG before any work starts if the source dataset is missing.
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

        if DATASET_PATH.stat().st_size == 0:
            raise ValueError(f"Dataset is empty: {DATASET_PATH}")

        dataset_message = (
            f"Dataset is ready at {DATASET_PATH.relative_to(PROJECT_ROOT)} "
            f"({DATASET_PATH.stat().st_size} bytes)."
        )
        print(dataset_message)
        return dataset_message

    @task(task_id="prepare_output_folders")
    def prepare_output_folders() -> dict[str, str]:
        # Pre-create the local output folders so downstream tasks can write predictably.
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

        folder_summary = {
            "model_folder": str(MODEL_PATH.parent.relative_to(PROJECT_ROOT)),
            "metadata_folder": str(METADATA_PATH.parent.relative_to(PROJECT_ROOT)),
            "mlflow_folder": str(MLFLOW_TRACKING_DIR.relative_to(PROJECT_ROOT)),
        }
        print(folder_summary)
        return folder_summary

    @task(task_id="run_regression_training")
    def run_regression_training() -> str:
        # Give Airflow-triggered runs a distinct name so they are easy to spot in MLflow.
        run_name = f"flight_price_airflow_{pendulum.now('UTC').format('YYYYMMDD_HHmmss')}"
        command = [
            sys.executable,
            str(TRAINING_SCRIPT_PATH),
            "--run-name",
            run_name,
        ]
        run_command(command)
        return run_name

    @task(task_id="verify_model_artifact")
    def verify_model_artifact() -> str:
        # The DAG is not complete unless the refreshed model file is actually present.
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file was not created: {MODEL_PATH}")

        if MODEL_PATH.stat().st_size == 0:
            raise ValueError(f"Model file is empty: {MODEL_PATH}")

        model_message = (
            f"Model file is ready at {MODEL_PATH.relative_to(PROJECT_ROOT)} "
            f"({MODEL_PATH.stat().st_size} bytes)."
        )
        print(model_message)
        return model_message

    @task(task_id="verify_metadata_file")
    def verify_metadata_file() -> dict[str, str]:
        # The metadata file is the quick link between the saved model and the MLflow run.
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata file was not created: {METADATA_PATH}")

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        required_keys = ["run_id", "experiment_name", "metrics", "model_file"]
        missing_keys = [key for key in required_keys if key not in metadata]

        if missing_keys:
            raise ValueError(f"Metadata file is missing keys: {missing_keys}")

        metadata_summary = {
            "run_id": str(metadata["run_id"]),
            "experiment_name": str(metadata["experiment_name"]),
            "metadata_file": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        }
        print(metadata_summary)
        return metadata_summary

    @task(task_id="verify_mlflow_artifacts")
    def verify_mlflow_artifacts(metadata_summary: dict[str, str]) -> str:
        # Read the run id from the metadata file, then confirm the matching MLflow
        # run folder contains the artifacts this workflow is supposed to log.
        run_id = metadata_summary["run_id"]
        run_directories = [
            meta_file.parent
            for meta_file in MLFLOW_TRACKING_DIR.rglob("meta.yaml")
            if meta_file.parent.name == run_id
        ]

        if not run_directories:
            raise FileNotFoundError(f"MLflow run folder was not found for run_id={run_id}")

        run_directory = run_directories[0]
        model_artifacts = list(run_directory.rglob("flight_price_model.joblib"))
        metadata_artifacts = list(run_directory.rglob("flight_price_model_metadata.json"))

        if not model_artifacts:
            raise FileNotFoundError(
                f"MLflow joblib artifact was not found inside run {run_id}"
            )

        if not metadata_artifacts:
            raise FileNotFoundError(
                f"MLflow metadata artifact was not found inside run {run_id}"
            )

        mlflow_message = (
            f"MLflow artifacts are ready for run {run_id} in "
            f"{run_directory.relative_to(PROJECT_ROOT)}."
        )
        print(mlflow_message)
        return mlflow_message

    # Keep the dependency graph explicit so the Airflow UI reads cleanly during demos.
    dataset_ready = check_data_availability()
    folders_ready = prepare_output_folders()
    training_run_name = run_regression_training()
    model_ready = verify_model_artifact()
    metadata_ready = verify_metadata_file()
    mlflow_ready = verify_mlflow_artifacts(metadata_ready)

    [dataset_ready, folders_ready] >> training_run_name
    training_run_name >> [model_ready, metadata_ready]
    [model_ready, metadata_ready] >> mlflow_ready


flight_price_regression_pipeline()
