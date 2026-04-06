#!/usr/bin/env bash

# -----------------------------
# 1. Start Airflow from WSL
# -----------------------------
set -euo pipefail

# -----------------------------
# 2. Resolve project and virtual environment
# -----------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_VENV="${AIRFLOW_VENV:-$HOME/.venvs/voyage-airflow}"

# -----------------------------
# 3. Configure Airflow and MLflow environment
# -----------------------------
# Move into the project folder before running commands.
cd "$PROJECT_ROOT"
source "$AIRFLOW_VENV/bin/activate"

unset MLFLOW_TRACKING_URI || true

export AIRFLOW_HOME="$PROJECT_ROOT/airflow_runtime"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-flight-price-local-training-airflow}"
export TUNING_ITERATIONS="${TUNING_ITERATIONS:-5}"

# -----------------------------
# 4. Start Airflow standalone server
# -----------------------------
mkdir -p "$AIRFLOW_HOME" "$PROJECT_ROOT/runtime_logs"

airflow db migrate
airflow standalone
