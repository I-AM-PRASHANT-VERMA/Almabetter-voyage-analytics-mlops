# Docker Compose Local Setup

Docker Compose is used to run the project as a group of local services. It brings together APIs, Streamlit apps, gateway routing, MLflow, Jenkins, and validation jobs.

## Main File

```text
MLops pipeline/docker-compose.yml
```

## Default Services

Running Docker Compose without a profile starts the main serving layer:

```bash
cd "MLops pipeline"
docker compose up --build
```

This starts:

- flight API
- hotel API
- gender API
- flight Streamlit app
- hotel Streamlit app
- gender Streamlit app
- local Nginx gateway

Open the gateway:

```text
http://localhost:8090
```

The required small serving artifacts are tracked in the repository, so these services can load their model files after a fresh clone.

## Optional Profiles

Some services are profile-based because they are not needed every time.

| Profile | Command | Purpose |
| --- | --- | --- |
| `mlflow` | `docker compose --profile mlflow up --build` | Starts MLflow UI and MLflow training job |
| `mlops` | `docker compose --profile mlops run --rm flight-validation` | Runs flight workflow validation |
| `jenkins` | `docker compose --profile jenkins up --build` | Starts local Jenkins CI/CD server |

## Stop Services

```bash
docker compose down
```

## Main Ports

| Service | Port |
| --- | --- |
| Gateway | `8090` |
| Flight API | `5002` |
| Hotel API | `5001` |
| Gender API | `5003` |
| Flight Streamlit | `8501` |
| Hotel Streamlit | `8502` |
| Gender Streamlit | `8503` |
| Jenkins | `8081` |
| MLflow UI | `5004` |

## Validation Command

Before running a full demo, the Compose file can be checked with:

```bash
docker compose config
```

This checks whether Docker Compose can parse the service configuration.

## Notes

The project uses read-only mounts for dataset and model files where possible. Serving containers should read the model and data, not rewrite them during normal use.
