#!/usr/bin/env bash

# -----------------------------
# 1. Start detached Airflow from WSL
# -----------------------------
set -euo pipefail

# -----------------------------
# 2. Resolve project and virtual environment
# -----------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_VENV="${AIRFLOW_VENV:-$HOME/.venvs/voyage-airflow}"

# -----------------------------
# 3. Skip startup when Airflow is already running
# -----------------------------
# Move into the project folder before running commands.
cd "$PROJECT_ROOT"
mkdir -p runtime_logs airflow_runtime

if pgrep -f "$AIRFLOW_VENV/bin/airflow" >/dev/null; then
    echo "Airflow is already running."
    pgrep -af "$AIRFLOW_VENV/bin/airflow"
    exit 0
fi

# -----------------------------
# 4. Launch Airflow in background
# -----------------------------
nohup bash -lc "
    # Move into the project folder before running commands.
    cd '$PROJECT_ROOT'
    source '$AIRFLOW_VENV/bin/activate'
    unset MLFLOW_TRACKING_URI || true
    export AIRFLOW_HOME='$PROJECT_ROOT/airflow_runtime'
    export AIRFLOW__CORE__DAGS_FOLDER='$PROJECT_ROOT/airflow/dags'
    export AIRFLOW__CORE__LOAD_EXAMPLES=False
    export MLFLOW_ALLOW_FILE_STORE=true
    export MLFLOW_EXPERIMENT_NAME=\"\${MLFLOW_EXPERIMENT_NAME:-flight-price-local-training-airflow}\"
    export TUNING_ITERATIONS=\"\${TUNING_ITERATIONS:-5}\"
    airflow db migrate
    exec airflow standalone
" > runtime_logs/airflow_standalone.log 2>&1 < /dev/null &

echo $! > runtime_logs/airflow_standalone.pid
echo "Started Airflow with PID $(cat runtime_logs/airflow_standalone.pid)"
