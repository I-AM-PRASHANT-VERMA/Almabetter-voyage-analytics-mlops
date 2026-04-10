# Airflow Training Pipeline

Airflow is used to show the training workflow as a pipeline. The DAG is focused on the flight price regression workflow.

## Main File

```text
MLops pipeline/airflow/dags/flight_price_mlflow_training_pipeline.py
```

## DAG Name

```text
flight_price_mlflow_training_pipeline
```

## DAG Purpose

The DAG checks that required files exist, checks flight dataset drift, runs MLflow experiments, and verifies that expected MLflow outputs are produced.

## Pipeline Tasks

| Task | Purpose |
| --- | --- |
| `check_training_files` | Confirms required scripts and folders are available |
| `check_dataset_drift` | Runs the flight dataset drift check |
| `run_mlflow_experiments` | Starts the MLflow experiment script |
| `verify_mlflow_outputs` | Checks that expected reports/artifacts were created |

## Run Airflow Locally

Helper scripts are included:

```text
MLops pipeline/scripts/start_airflow_wsl.sh
MLops pipeline/scripts/start_airflow_wsl_detached.sh
MLops pipeline/scripts/run_airflow_mlflow_dag_wsl.sh
MLops pipeline/start_airflow_standalone.bat
```

These scripts are intended for local Airflow setup and demo runs.

## Expected Output

After the DAG runs, the workflow should produce training or validation artifacts and show the task status in the Airflow UI.

## Why Airflow Is Used

Airflow makes the training process visible as a pipeline instead of a single manual script. This supports the assignment requirement for scheduling, orchestration, and workflow automation.

## Notes

The DAG is configured for manual triggering. This is safer for a project demo because it avoids retraining unexpectedly whenever Airflow starts.
