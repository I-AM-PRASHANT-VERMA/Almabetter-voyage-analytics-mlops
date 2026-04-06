
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlflow
from mlflow import MlflowClient


# -----------------------------
# 1. Resolve training and artifact paths
# -----------------------------
# The script can run from the pipeline folder while still finding the shared root training folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_LOCAL_TRAINING_DIR = (
    PROJECT_ROOT / "local_training"
    if (PROJECT_ROOT / "local_training" / "train_flight_price.py").exists()
    else REPO_ROOT / "local_training"
)
LOCAL_TRAINING_DIR = Path(os.getenv("LOCAL_TRAINING_DIR", DEFAULT_LOCAL_TRAINING_DIR))
# Use the single official local training script so MLflow, Airflow, and Jenkins stay aligned.
TRAINING_SCRIPT = LOCAL_TRAINING_DIR / "train_flight_price.py"
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "dataset" / "travel_capstone"
    if (PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv").exists()
    else REPO_ROOT / "dataset" / "travel_capstone"
)
DATA_DIR = Path(os.getenv("FLIGHT_DATA_DIR", DEFAULT_DATA_DIR))
LOCAL_MODEL_PATH = LOCAL_TRAINING_DIR / "outputs" / "models" / "flight_price_model_latest.joblib"
LOCAL_METADATA_PATH = LOCAL_TRAINING_DIR / "outputs" / "metrics" / "flight_price_model_latest_metadata.json"
BEST_MODEL_PATH = PROJECT_ROOT / "joblib files" / "mlflow_training" / "flight_price_model_mlflow.joblib"
BEST_METADATA_PATH = PROJECT_ROOT / "joblib files" / "mlflow_training" / "flight_price_model_mlflow_metadata.json"
SERVING_MODEL_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model.joblib"
SERVING_METADATA_PATH = PROJECT_ROOT / "joblib files" / "flight_price_model_metadata.json"
SUMMARY_PATH = PROJECT_ROOT / "jenkins_artifacts" / "mlflow_experiments" / "mlflow_experiment_summary.json"
REGISTERED_MODEL_NAME = "voyage_flight_price_regression"
REGISTERED_MODEL_ALIAS = "champion"
FINAL_RUN_PREFIX = "register_best_model_"
QUICK_TRAIN_SAMPLE_ROWS = "30000"
QUICK_TUNING_ITERATIONS = "2"


def display_path(path):
    # -----------------------------
    # 2. Print readable paths
    # -----------------------------
    # Show short paths in logs whenever possible.
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def run_training(profile, tracking_uri=None):
    # -----------------------------
    # 3. Run local training script
    # -----------------------------
    # Validate the two most important inputs before starting a heavier training run.
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAINING_SCRIPT}")

    if not (DATA_DIR / "flights.csv").exists():
        raise FileNotFoundError(f"Flight dataset not found: {DATA_DIR / 'flights.csv'}")

    environment = os.environ.copy()
    # Pass dataset and MLflow settings to the child training process.
    environment["FLIGHT_DATA_DIR"] = str(DATA_DIR)
    # MLflow prints Unicode status symbols that need UTF-8 in Windows terminals.
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    environment.setdefault("MLFLOW_EXPERIMENT_NAME", f"flight-price-local-{profile}")
    if tracking_uri:
        environment["MLFLOW_TRACKING_URI"] = tracking_uri

    # Quick mode reduces both data volume and tuning attempts for smoke tests.
    if profile == "quick":
        environment.setdefault("TRAIN_SAMPLE_ROWS", QUICK_TRAIN_SAMPLE_ROWS)
        environment.setdefault("TUNING_ITERATIONS", QUICK_TUNING_ITERATIONS)

    training_started_at_ms = int(time.time() * 1000)
    subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT)],
        cwd=LOCAL_TRAINING_DIR,
        env=environment,
        check=True,
    )
    return {
        "experiment_name": environment["MLFLOW_EXPERIMENT_NAME"],
        "started_at_ms": training_started_at_ms,
    }


def register_latest_model(profile, tracking_uri, training_context):
    # -----------------------------
    # 4. Register the final MLflow model
    # -----------------------------
    # Direct local runs can still promote joblib files without claiming a registry version.
    if not tracking_uri:
        return {
            "registry_status": "not_requested",
            "registry_note": "No tracking server URI was provided, so only local model promotion was completed.",
        }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    experiment = client.get_experiment_by_name(training_context["experiment_name"])
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not found: {training_context['experiment_name']}")

    # Find the final-model run created by this training invocation, not an older run.
    recent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    final_runs = [
        run
        for run in recent_runs
        if run.info.start_time >= training_context["started_at_ms"]
        and run.info.status == "FINISHED"
        and run.data.tags.get("mlflow.runName", "").startswith(FINAL_RUN_PREFIX)
    ]
    if not final_runs:
        raise RuntimeError("The final MLflow model run from this training invocation was not found.")

    final_run = final_runs[0]
    model_uri = f"runs:/{final_run.info.run_id}/sklearn_model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        tags={"profile": profile, "selection_metric": "group_rmse"},
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=REGISTERED_MODEL_ALIAS,
        version=str(model_version.version),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=str(model_version.version),
        key="source_run_name",
        value=final_run.data.tags.get("mlflow.runName", ""),
    )

    return {
        "registry_status": "registered",
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": str(model_version.version),
        "registered_model_alias": REGISTERED_MODEL_ALIAS,
        "model_uri": model_uri,
        "source_run_id": final_run.info.run_id,
        "source_run_name": final_run.data.tags.get("mlflow.runName", ""),
    }


def promote_local_training_output(profile, registry_details):
    # -----------------------------
    # 5. Promote latest training output
    # -----------------------------
    # After training, the latest root output becomes the promoted MLflow copy.
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Latest local model not found: {LOCAL_MODEL_PATH}")

    if not LOCAL_METADATA_PATH.exists():
        raise FileNotFoundError(f"Latest local metadata not found: {LOCAL_METADATA_PATH}")

    latest_metadata = json.loads(LOCAL_METADATA_PATH.read_text(encoding="utf-8"))

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SERVING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Keep one MLflow-tagged copy and one serving copy used by the apps.
    shutil.copy2(LOCAL_MODEL_PATH, BEST_MODEL_PATH)
    shutil.copy2(LOCAL_METADATA_PATH, BEST_METADATA_PATH)
    shutil.copy2(LOCAL_MODEL_PATH, SERVING_MODEL_PATH)
    shutil.copy2(LOCAL_METADATA_PATH, SERVING_METADATA_PATH)

    # Extend the metadata so later checks know this model came from local MLflow flow.
    promoted_metadata = {
        **latest_metadata,
        "profile": profile,
        **registry_details,
        "promotion_note": "Root local_training output promoted for MLflow, Airflow, Jenkins, Docker, and Kubernetes checks.",
        "source_model_file": str(LOCAL_MODEL_PATH),
        "source_metadata_file": str(LOCAL_METADATA_PATH),
        "promoted_model_file": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "serving_model_file": str(SERVING_MODEL_PATH.relative_to(PROJECT_ROOT)),
    }
    BEST_METADATA_PATH.write_text(json.dumps(promoted_metadata, indent=4), encoding="utf-8")
    SERVING_METADATA_PATH.write_text(json.dumps(promoted_metadata, indent=4), encoding="utf-8")
    return promoted_metadata


def run_experiments(profile, tracking_uri=None):
    # -----------------------------
    # 6. Run experiment and save summary
    # -----------------------------
    # Registration happens before serving files are updated, so a failed registry step cannot promote a stale release.
    training_context = run_training(profile, tracking_uri)
    registry_details = register_latest_model(profile, tracking_uri, training_context)
    promoted_metadata = promote_local_training_output(profile, registry_details)

    # Save one summary file for Jenkins and manual inspection.
    summary = {
        "profile": profile,
        "local_training_dir": str(LOCAL_TRAINING_DIR),
        "training_script": str(TRAINING_SCRIPT),
        "data_dir": str(DATA_DIR),
        "tracking_uri": tracking_uri or "local_training file store",
        "registry": registry_details,
        "selected_model": promoted_metadata,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("Local MLflow experiments completed successfully.")
    print(f"Selected model: {promoted_metadata['selected_model']}")
    metrics = promoted_metadata.get("metrics", {})
    print(f"Selection metric: {metrics.get('selection_metric') or metrics.get('group_rmse') or metrics.get('time_rmse')}")
    if registry_details["registry_status"] == "registered":
        print(
            f"Registered model: {registry_details['registered_model_name']} "
            f"version {registry_details['registered_model_version']} "
            f"with alias {registry_details['registered_model_alias']}"
        )
    else:
        print(f"Registry status: {registry_details['registry_status']}")
    print(f"Summary file: {display_path(SUMMARY_PATH)}")
    return summary


def parse_args():
    # -----------------------------
    # 7. Read command line options
    # -----------------------------
    # Keep the CLI small: one profile choice and one optional tracking override.
    parser = argparse.ArgumentParser(
        description="Run local MLflow training, register the final model when a server is provided, and promote serving files."
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "standard"],
        default="standard",
        help="Quick uses a smaller sample and fewer tuning attempts; standard uses the full configured training workflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Docker can use the MLflow UI service; local runs can omit it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # -----------------------------
    # 8. Script entry point
    # -----------------------------
    # Standard Python entry point for local runs and Airflow/Jenkins calls.
    parsed_args = parse_args()
    run_experiments(parsed_args.profile, parsed_args.tracking_uri)
