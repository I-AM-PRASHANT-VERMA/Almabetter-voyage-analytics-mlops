#!/usr/bin/env bash

# -----------------------------
# 1. Trigger Airflow MLflow DAG from WSL
# -----------------------------
set -euo pipefail

# -----------------------------
# 2. Resolve project, virtualenv, and run id
# -----------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_VENV="${AIRFLOW_VENV:-$HOME/.venvs/voyage-airflow}"
DAG_ID="flight_price_mlflow_training_pipeline"
RUN_ID="manual_airflow_mlflow_$(date +%Y%m%d_%H%M%S)"
export RUN_ID

# -----------------------------
# 3. Configure Airflow runtime
# -----------------------------
# Move into the project folder before running commands.
cd "$PROJECT_ROOT"
source "$AIRFLOW_VENV/bin/activate"

export AIRFLOW_HOME="$PROJECT_ROOT/airflow_runtime"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-flight-price-local-training-airflow}"
export TUNING_ITERATIONS="${TUNING_ITERATIONS:-5}"

# -----------------------------
# 4. Preserve optional MLflow tracking URI
# -----------------------------
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    export MLFLOW_TRACKING_URI
else
    unset MLFLOW_TRACKING_URI || true
fi

# -----------------------------
# 5. Trigger DAG run
# -----------------------------
mkdir -p "$AIRFLOW_HOME" "$PROJECT_ROOT/runtime_logs"

airflow dags unpause "$DAG_ID" >/dev/null
airflow dags trigger "$DAG_ID" -r "$RUN_ID"
echo "$RUN_ID" > "$PROJECT_ROOT/runtime_logs/latest_airflow_mlflow_run_id.txt"
echo "Triggered $DAG_ID with run id: $RUN_ID"

# -----------------------------
# 6. Poll DAG status until finish
# -----------------------------
for _ in $(seq 1 90); do
    state="$(
        python3 - "$DAG_ID" "$RUN_ID" <<'PY'
import sqlite3
import sys

dag_id = sys.argv[1]
run_id = sys.argv[2]

con = sqlite3.connect("airflow_runtime/airflow.db")
row = con.execute(
    "select state from dag_run where dag_id = ? and run_id = ?",
    (dag_id, run_id),
).fetchone()

print(row[0] if row else "waiting")
PY
    )"

    echo "DAG state: $state"

    if [[ "$state" == "success" ]]; then
        airflow tasks states-for-dag-run "$DAG_ID" "$RUN_ID"
        exit 0
    fi

    if [[ "$state" == "failed" ]]; then
        airflow tasks states-for-dag-run "$DAG_ID" "$RUN_ID"
        exit 2
    fi

    sleep 10
done

airflow tasks states-for-dag-run "$DAG_ID" "$RUN_ID"
exit 3
