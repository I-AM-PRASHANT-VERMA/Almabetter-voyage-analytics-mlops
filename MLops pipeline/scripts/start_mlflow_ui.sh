#!/bin/sh

set -eu

# Keep the runtime defaults explicit so the MLflow UI behaves predictably in Azure.
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-/app/mlruns}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-*}"
MLFLOW_CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-*}"

exec mlflow ui \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
  --cors-allowed-origins "${MLFLOW_CORS_ALLOWED_ORIGINS}"
