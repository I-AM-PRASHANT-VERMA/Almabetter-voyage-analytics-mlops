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

The DAG checks that required files exist, checks flight dataset drift, compares the current dataset fingerprint with the promoted model metadata, retrains only when required, and verifies the promoted outputs.

## Pipeline Tasks

| Task | Purpose |
| --- | --- |
| `check_training_files` | Confirms required scripts and folders are available |
| `check_dataset_drift` | Runs the flight dataset drift check |
| `assess_retraining` | Combines drift and dataset fingerprint evidence |
| `run_mlflow_experiments` | Starts MLflow training only when required |
| `verify_mlflow_outputs` | Checks that expected reports/artifacts were created |
| `trigger_gated_azure_cd` | Calls Jenkins CD only after a new model is promoted and the Azure switch is enabled |

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

## Schedule and Safety

The default schedule is daily at 02:00 UTC. It can be changed through `AIRFLOW_FLIGHT_TRAINING_SCHEDULE`, or set to an empty value for manual-only runs.

The Azure trigger remains a no-op while `VOYAGE_AZURE_DEPLOYMENT_ENABLED=false`. Airflow can still detect drift, retrain, promote, and validate the local model without contacting Azure.
