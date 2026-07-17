# User Guide and Troubleshooting

This guide gives practical commands for running and checking the project locally.

## Requirements

- Git
- Python
- Docker Desktop with Docker Compose
- Optional: WSL for Airflow helper scripts
- Optional: Azure CLI, kubectl, and Jenkins credentials for cloud deployment

## Clone the Repository

```bash
git clone https://github.com/I-AM-PRASHANT-VERMA/Almabetter-voyage-analytics-mlops.git
cd Almabetter-voyage-analytics-mlops
```

## Run Main Local Apps

```bash
cd "MLops pipeline"
docker compose up --build
```

The required small serving model files are included in the repository. If a model file is regenerated later, the local training or MLflow workflow can update it.

Open:

```text
http://localhost:8090
```

## Stop Local Apps

```bash
docker compose down
```

## Run MLflow Workflow

```bash
docker compose --profile mlflow up --build
```

Open MLflow:

```text
http://localhost:5004
```

## Run Validation

```bash
docker compose --profile mlops run --rm flight-validation
```

## Start Jenkins

```bash
docker compose --profile jenkins up --build
```

Open:

```text
http://localhost:8081
```

## Common Issues

| Issue | What to check |
| --- | --- |
| Docker command fails | Make sure Docker Desktop is running |
| Port already in use | Stop the app using that port or change the port mapping |
| Streamlit app does not load | Wait for the related API health check to pass |
| API cannot find model file | Check that `MLops pipeline/joblib files` contains the serving model files |
| Dataset error | Confirm `dataset/travel_capstone` contains flights, hotels, and users CSV files |
| MLflow UI not opening | Use the `mlflow` profile and open `http://localhost:5004` |
| Jenkins cannot run Docker steps | Docker must be available to the Jenkins container or host setup |
| Azure deployment fails | Check Azure login, ACR, AKS, Key Vault, and Jenkins credentials |

## Useful Checks

Check Docker Compose syntax:

```bash
docker compose config
```

Check Git status:

```bash
git status
```

Check the current branch:

```bash
git branch
```

## Notes for Reviewers

The notebooks explain the analysis and modeling work. The local MLOps code shows how the project can be run as services and prepared for deployment. The regression model is the main focus of the deeper MLOps workflow.
