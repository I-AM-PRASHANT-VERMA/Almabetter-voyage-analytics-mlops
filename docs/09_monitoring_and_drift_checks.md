# Monitoring and Drift Checks

Monitoring is included to show how the project can be checked after training and deployment. The project covers dataset drift checks, service health checks, workflow validation, and Azure telemetry support.

## Important Files

| File | Purpose |
| --- | --- |
| `MLops pipeline/scripts/check_flight_dataset_drift.py` | Compares current flight data with a baseline |
| `MLops pipeline/scripts/assess_flight_retraining.py` | Combines drift and dataset fingerprint evidence |
| `dataset/baselines/flight_dataset_baseline.json` | Stored baseline for flight dataset checks |
| `MLops pipeline/scripts/validate_flight_regression_workflow.py` | Validates promoted flight workflow outputs |
| `MLops pipeline/scripts/monitor_voyage_workflow.py` | Builds a workflow monitoring report |
| `MLops pipeline/monitoring.py` | Application Insights setup helper |

## Dataset Drift Check

The drift check looks for meaningful changes in the flight dataset. This is useful because a model trained on one data distribution can become less reliable if the input data changes heavily.

Run from the `MLops pipeline` folder:

```bash
python scripts/check_flight_dataset_drift.py
```

Refresh the baseline only when the current dataset is intentionally accepted:

```bash
python scripts/check_flight_dataset_drift.py --refresh-baseline
```

The promoted flight model metadata stores the dataset SHA-256 fingerprint used for training. Airflow and Jenkins compare that fingerprint with the current dataset. A changed fingerprint or a drift alert requests retraining; unchanged data keeps the current model.

## Health Checks

Docker Compose and Kubernetes manifests include health checks for APIs, Streamlit apps, and the gateway. These checks help confirm whether services are responding after startup or deployment.

Examples:

```text
http://localhost:5002/health
http://localhost:5001/health
http://localhost:5003/health
http://localhost:8090/health
```

## Workflow Monitoring

The monitoring script checks project workflow status and writes a report that can be reviewed after local or CI runs.

```bash
python scripts/monitor_voyage_workflow.py
```

## Azure Telemetry

The project supports Azure Application Insights through an environment variable:

```text
APPLICATIONINSIGHTS_CONNECTION_STRING
```

In Kubernetes, this value is provided through a secret. This avoids storing real telemetry secrets in the repository.

## Expected Output

Monitoring and validation can produce:

- drift summary JSON
- Jenkins validation artifacts
- workflow monitoring reports
- service health check results
- Azure telemetry events when Application Insights is configured

## Notes

The monitoring setup is meant to support project review and troubleshooting. Dataset drift is connected to the scheduled Airflow retraining decision, while Azure deployment remains protected by the separate environment switch.
