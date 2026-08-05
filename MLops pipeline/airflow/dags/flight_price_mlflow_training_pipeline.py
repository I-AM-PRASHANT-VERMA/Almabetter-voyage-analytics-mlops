from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task


# -----------------------------
# 1. Resolve project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to the MLops pipeline folder
REPO_ROOT = PROJECT_ROOT.parent  # used when shared folders are kept one level above the pipeline folder

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv"  # first try the pipeline dataset path
    if (PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv").exists()
    else REPO_ROOT / "dataset" / "travel_capstone" / "flights.csv"  # fallback to the repo-level dataset
)
DATASET_PATH = Path(os.getenv("FLIGHT_DATASET_PATH", DEFAULT_DATASET_PATH))  # allow Docker/Airflow to override the dataset path

DRIFT_SCRIPT = PROJECT_ROOT / "scripts" / "check_flight_dataset_drift.py"  # checks the dataset before training
RETRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "assess_flight_retraining.py"
TRAINING_RUNNER = PROJECT_ROOT / "scripts" / "run_flight_price_mlflow_experiments.py"  # shared runner for MLflow training and promotion
CD_TRIGGER_SCRIPT = PROJECT_ROOT / "scripts" / "trigger_jenkins_cd.py"
DRIFT_REPORT = PROJECT_ROOT / "jenkins_artifacts" / "data_drift" / "flight_dataset_drift_summary.json"  # drift output expected by this DAG
RETRAINING_DECISION = PROJECT_ROOT / "jenkins_artifacts" / "automation" / "flight_retraining_decision.json"
EXPERIMENT_SUMMARY_PATH = PROJECT_ROOT / "jenkins_artifacts" / "mlflow_experiments" / "mlflow_experiment_summary.json"  # runner summary file

SERVING_MODEL_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model.joblib"  # final model file used by the apps
SERVING_METADATA_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model_metadata.json"  # metadata paired with the serving model
AIRFLOW_SCHEDULE = os.getenv("AIRFLOW_FLIGHT_TRAINING_SCHEDULE", "0 2 * * *").strip() or None


# -----------------------------
# 2. Format readable paths
# -----------------------------
def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))  # prefer a short path inside the pipeline folder
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))  # fallback for repo-level files like dataset/local_training
        except ValueError:
            return str(path)  # keep full path if it sits outside the known project roots


# -----------------------------
# 3. Run shell commands safely
# -----------------------------
def run_command(command: list[str]) -> None:
    env = os.environ.copy()  # copy the current Airflow environment before adding project defaults
    env.setdefault("PYTHONIOENCODING", "utf-8")  # keeps subprocess logs readable on Windows and containers
    env.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # lets local MLflow file storage work when no server is passed
    env.setdefault("MLFLOW_EXPERIMENT_NAME", "flight-price-local-training-airflow")  # separates Airflow runs in MLflow
    env.setdefault("TUNING_ITERATIONS", "5")  # keeps this local training pipeline practical to rerun

    result = subprocess.run(
        command,  # command is passed as a list to avoid shell quoting issues
        cwd=PROJECT_ROOT,  # run from the pipeline folder so relative paths stay stable
        env=env,  # pass the prepared environment into the child script
        capture_output=True,  # capture logs so Airflow can print them in the task output
        text=True,  # return stdout and stderr as strings
        check=False,  # handle failures manually after printing useful logs
    )

    if result.stdout:
        print(result.stdout)  # keep successful subprocess logs visible in Airflow

    if result.stderr:
        print(result.stderr, file=sys.stderr)  # keep warnings and errors in the stderr stream

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")  # fail the Airflow task cleanly


# -----------------------------
# 4. Define Airflow DAG
# -----------------------------
@dag(
    dag_id="flight_price_mlflow_training_pipeline",  # name shown in the Airflow UI
    schedule=AIRFLOW_SCHEDULE,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),  # required by Airflow even for manual DAGs
    catchup=False,  # do not create old missed runs
    tags=["mlops", "flight-price", "mlflow", "local-training"],  # makes the DAG easy to filter in Airflow
    description="Check dataset drift, run local MLflow training, and promote the best flight price model.",
)
def flight_price_mlflow_training_pipeline():
    # -----------------------------
    # 5. Check required files
    # -----------------------------
    @task(task_id="check_training_files")
    def check_training_files() -> None:
        for path in [DATASET_PATH, DRIFT_SCRIPT, RETRAINING_SCRIPT, TRAINING_RUNNER, CD_TRIGGER_SCRIPT]:
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Required training file is missing or empty: {path}")  # stop before drift/training starts

    # -----------------------------
    # 6. Check dataset drift
    # -----------------------------
    @task(task_id="check_dataset_drift")
    def check_dataset_drift() -> str:
        run_command([sys.executable, str(DRIFT_SCRIPT)])  # create the latest drift summary report
        return str(DRIFT_REPORT.relative_to(PROJECT_ROOT))  # pass a short report path to the next task

    # -----------------------------
    # 7. Decide whether retraining is needed
    # -----------------------------
    @task(task_id="assess_retraining")
    def assess_retraining(drift_report: str) -> str:
        run_command(
            [
                sys.executable,
                str(RETRAINING_SCRIPT),
                "--drift-report",
                str(PROJECT_ROOT / drift_report),
                "--output",
                str(RETRAINING_DECISION),
            ]
        )
        return str(RETRAINING_DECISION)

    # -----------------------------
    # 8. Run MLflow experiments when required
    # -----------------------------
    @task(task_id="run_mlflow_experiments")
    def run_mlflow_experiments(decision_file: str) -> str:
        decision = json.loads(Path(decision_file).read_text(encoding="utf-8"))
        print(f"Retraining required: {decision['retrain_required']}")
        print(f"Retraining reasons: {decision['reasons']}")

        if not decision["retrain_required"]:
            return "existing-serving-artifacts"

        command = [sys.executable, str(TRAINING_RUNNER), "--profile", "standard"]  # delegate training and promotion to the shared runner
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()  # use MLflow server when Airflow provides one
        if tracking_uri:
            command.extend(["--tracking-uri", tracking_uri])  # enables registry flow in the MLflow runner

        run_command(command)  # run training, registry, promotion, and summary creation
        return str(EXPERIMENT_SUMMARY_PATH)  # pass the runner summary to the verification task

    # -----------------------------
    # 9. Verify promoted outputs
    # -----------------------------
    @task(task_id="verify_mlflow_outputs")
    def verify_mlflow_outputs(summary_file: str) -> None:
        if summary_file == "existing-serving-artifacts":
            if not SERVING_MODEL_PATH.exists() or not SERVING_METADATA_PATH.exists():
                raise FileNotFoundError("Existing serving artifacts are missing.")
            print("No retraining was required; existing serving artifacts remain active.")
            return

        summary_path = Path(summary_file)  # summary path returned by the MLflow runner task
        if not summary_path.exists() or not SERVING_MODEL_PATH.exists() or not SERVING_METADATA_PATH.exists():
            raise FileNotFoundError("MLflow summary or promoted serving files were not created.")  # fail if promotion did not produce the expected files

        summary = json.loads(summary_path.read_text(encoding="utf-8"))  # load the stable experiment result
        metadata = summary["selected_model"]  # selected model metadata written by the runner
        metrics = metadata["metrics"]  # evaluation metrics saved with the selected model
        print(
            {
                "selected_model": metadata["selected_model"],
                "version_id": metadata["version_id"],
                "time_rmse": metrics["time_rmse"],
                "group_rmse": metrics["group_rmse"],
                "registry_status": summary["registry"]["registry_status"],
                "promoted_model": display_path(SERVING_MODEL_PATH),
            }
        )

    # -----------------------------
    # 10. Trigger gated CD after a new model is promoted
    # -----------------------------
    @task(task_id="trigger_gated_azure_cd")
    def trigger_gated_azure_cd(decision_file: str) -> None:
        run_command(
            [
                sys.executable,
                str(CD_TRIGGER_SCRIPT),
                "--decision-file",
                decision_file,
                "--only-if-retraining-required",
            ]
        )

    # -----------------------------
    # 11. Wire Airflow task order
    # -----------------------------
    files_ready = check_training_files()  # first make sure required inputs exist
    drift_report = check_dataset_drift()  # then create the dataset drift report
    decision_file = assess_retraining(drift_report)
    experiment_summary = run_mlflow_experiments(decision_file)
    outputs_verified = verify_mlflow_outputs(experiment_summary)
    cd_triggered = trigger_gated_azure_cd(decision_file)

    files_ready >> drift_report  # explicit dependency: drift check waits for file validation
    outputs_verified >> cd_triggered


# -----------------------------
# 12. Register DAG object
# -----------------------------
flight_price_mlflow_training_dag = flight_price_mlflow_training_pipeline()  # expose this DAG to Airflow
