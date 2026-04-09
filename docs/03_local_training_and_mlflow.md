# Local Training and MLflow

This part covers the repeatable flight price model training workflow. The goal is to move from development workflow experimentation to a script-based workflow that can be validated, tracked, and reused.

## Important Files

| File | Purpose |
| --- | --- |
| `local_training/train_flight_price.py` | Local flight price training script |
| `MLops pipeline/scripts/check_flight_dataset_drift.py` | Checks current flight dataset against a saved baseline |
| `MLops pipeline/scripts/run_flight_price_mlflow_experiments.py` | Runs MLflow-tracked flight model experiments |
| `MLops pipeline/scripts/validate_flight_regression_workflow.py` | Validates the promoted flight model workflow |
| `dataset/baselines/flight_dataset_baseline.json` | Baseline used for dataset drift comparison |

## Local Training Role

The local training script prepares data, trains the flight price regression model, evaluates metrics, and saves model outputs. It is useful when the model needs to be trained outside interactive in a repeatable way.

## MLflow Role

MLflow is used to track experiment runs, parameters, metrics, and artifacts. This helps compare model runs and keep a record of which model version was selected.

## Run MLflow Workflow

From the `MLops pipeline` folder:

```bash
docker compose --profile mlflow up --build
```

This starts the MLflow UI and the training job profile.

The MLflow UI is available at:

```text
http://localhost:5004
```

## Dataset Drift Check

The drift check compares the current flight dataset with a stored baseline. This helps catch major dataset changes before training or validation.

Run it directly from the `MLops pipeline` folder:

```bash
python scripts/check_flight_dataset_drift.py
```

To refresh the baseline intentionally:

```bash
python scripts/check_flight_dataset_drift.py --refresh-baseline
```

## Validation Check

The validation script checks that the promoted flight regression workflow is usable and that required files are in place.

```bash
docker compose --profile mlops run --rm flight-validation
```

## Expected Outputs

- MLflow runs and metrics
- saved model artifacts
- validation reports under local artifact folders
- drift report under `jenkins_artifacts/data_drift`

## Serving Artifacts

The small top-level joblib artifacts required by the Flask and Streamlit apps are included in the repository. This allows a fresh clone to run the local demo. The larger MLflow run folders and archived training outputs remain ignored because they are generated artifacts.

The MLflow workflow can regenerate and promote the flight price serving artifact when needed.

## Notes

The model may show very high performance because the dataset has a structured pricing pattern. The project includes duplicate cleanup and feature-overlap checks to reduce leakage risk, but results should still be understood within the limits of the provided dataset.
