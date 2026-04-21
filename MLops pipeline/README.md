# Voyage Analytics MLOps Capstone

Travel analytics project with three app surfaces and one end-to-end MLOps workflow:

- Flight price regression
- Hotel recommendation
- Gender classification

The main MLOps pipeline in this repo is built around the flight price model. It uses MLflow for experiment tracking, Airflow for orchestration, Docker for packaging, Jenkins for validation, and Kubernetes manifests for a simple local-cluster deployment.

## Project Layout

- `training/train_flight_price_regression_mlflow.py`: flight model training with MLflow logging
- `airflow/dags/flight_price_regression_pipeline.py`: Airflow DAG for the flight workflow
- `scripts/validate_flight_regression_workflow.py`: local/Jenkins validation helper
- `streamlit/`: Streamlit apps for flight, hotel, and gender demos
- `flask_apps/`: Flask apps and browser pages for the same models
- `ngrok_apps/`: small launchers that expose each Flask app through ngrok
- `k8s/`: Kubernetes manifests for the flight workflow
- `joblib files/`: local model output folder used for local runs and mounted runtime storage
- `dataset/travel_capstone/`: source CSV files used by the apps and training flow

## Quick Setup

Install the main dependencies from the project root:

```powershell
pip install -r requirements.txt
```

For Airflow, install its separate requirement file inside a WSL or Linux environment:

```bash
pip install -r requirements-airflow.txt --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.2.0/constraints-3.12.txt"
```

## Run The Apps Locally

### Streamlit

```powershell
streamlit run streamlit/flight_price_app.py
streamlit run streamlit/hotel_recommendation_app.py
streamlit run streamlit/gender_classification_app.py
```

Default URLs:

- Flight: `http://127.0.0.1:8501`
- Hotel: Streamlit will choose the next open port
- Gender: Streamlit will choose the next open port

### Flask

```powershell
python flask_apps/flight_price_flask_app/app.py
python flask_apps/hotel_recommendation_flask_app/app.py
python flask_apps/gender_classification_flask_app/app.py
```

Default URLs:

- Flight API: `http://127.0.0.1:5002`
- Hotel API: `http://127.0.0.1:5001`
- Gender API: `http://127.0.0.1:5003`

Useful health routes:

- `http://127.0.0.1:5001/health`
- `http://127.0.0.1:5002/health`
- `http://127.0.0.1:5003/health`

### ngrok

Set an ngrok token first if you want public links:

```powershell
$env:NGROK_AUTHTOKEN="your_ngrok_token_here"
```

Then run any launcher you need:

```powershell
python ngrok_apps/run_hotel_recommendation_ngrok.py
python ngrok_apps/run_flight_price_ngrok.py
python ngrok_apps/run_gender_classification_ngrok.py
```

## Flight MLOps Workflow

### Train The Flight Model With MLflow

```powershell
python training/train_flight_price_regression_mlflow.py
```

Optional custom run name:

```powershell
python training/train_flight_price_regression_mlflow.py --run-name flight_price_main_model
```

This updates your local runtime folders:

- `joblib files/flight_price_model.joblib`
- `joblib files/flight_price_model_metadata.json`
- `mlruns/`

### Open MLflow UI

```powershell
mlflow ui --backend-store-uri "./mlruns" --host 127.0.0.1 --port 5000
```

Or use the included shortcut:

```powershell
start_mlflow_ui.bat
```

MLflow UI:

- `http://127.0.0.1:5000`

### Run The Validation Script

This is the same workflow check Jenkins uses, but it writes into a separate output folder instead of touching the main tracked artifacts.

```powershell
python scripts/validate_flight_regression_workflow.py --output-dir jenkins_artifacts/local_check
```

## Docker

The Docker setup only covers the flight workflow.

### Build single images

```powershell
docker build -t voyage-flight-api .
docker build -f Dockerfile.streamlit -t voyage-flight-streamlit .
docker build -f Dockerfile.mlops -t voyage-flight-mlops .
```

### Run training in Docker

```powershell
docker compose --profile mlops run --rm flight-training
```

The compose setup mounts `joblib files/` and `mlruns/` from the local machine. The cloud deployment uses Azure-mounted storage instead of baking those folders into the image.

### Run MLflow UI in Docker

```powershell
docker compose --profile mlops up --build mlflow-ui
```

### Run the flight API and Streamlit app together

```powershell
docker compose up --build flight-api flight-streamlit
```

Compose URLs:

- Flight API: `http://127.0.0.1:5002`
- Flight Streamlit: `http://127.0.0.1:8501`
- MLflow UI: `http://127.0.0.1:5000`

If you want the Streamlit container to call the API outside Compose, copy `.env.example` to `.env` and adjust `FLIGHT_PRICE_API_URL`.

## Airflow

Airflow is meant to run from WSL2 or Linux. Native Windows PowerShell is not supported for `airflow standalone` in this repo.

Start Airflow in WSL:

```bash
cd "/mnt/e/E almabetter projects/specilization track/voyage-analytics-almabetter-mlops-prashant"
source .venv-airflow/bin/activate
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
airflow standalone
```

Trigger the DAG from a second WSL terminal:

```bash
cd "/mnt/e/E almabetter projects/specilization track/voyage-analytics-almabetter-mlops-prashant"
source .venv-airflow/bin/activate
export AIRFLOW_HOME="$(pwd)/airflow"
airflow dags trigger flight_price_regression_pipeline
```

Airflow UI:

- `http://127.0.0.1:8080`

## Jenkins

The Jenkins pipeline is defined in `Jenkinsfile` and is designed for a Linux agent with Python 3. Docker is optional on the agent; if Docker is not available, the Docker validation stage is skipped.

Related files:

- `Jenkinsfile`
- `scripts/validate_flight_regression_workflow.py`

## Kubernetes

The Kubernetes manifests are also limited to the flight workflow:

- training job
- Flask API
- Streamlit app
- MLflow UI

Build the images first:

```powershell
docker build -t voyage-flight-api:latest .
docker build -f Dockerfile.streamlit -t voyage-flight-streamlit:latest .
docker build -f Dockerfile.mlops -t voyage-flight-mlops:latest .
```

Apply the manifests:

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/persistent-storage.yaml -n voyage-mlops
kubectl apply -f k8s/flight-training-job.yaml
kubectl apply -k k8s
```

Port-forward the services during a local demo:

```powershell
kubectl -n voyage-mlops port-forward svc/flight-api 5002:5002
kubectl -n voyage-mlops port-forward svc/flight-streamlit 8501:8501
kubectl -n voyage-mlops port-forward svc/mlflow-ui 5000:5000
```

## Notes

- Local training writes model files into `joblib files/` and MLflow history into `mlruns/`.
- The Azure deployment reads those runtime files from mounted Azure storage, not from tracked folders in the public repo.
- Runtime logs are written as structured JSON to standard output.
- Azure collects those container logs centrally in Log Analytics, and app telemetry can also flow into Application Insights when the connection string is present in the runtime.
- Alert rules are meant to watch the centralized Azure logs instead of local files inside the repo.
- The hotel and gender apps are available locally, but the Docker, Jenkins, Airflow, and Kubernetes workflow in this repo is centered on the flight model.
- A few local-only folders and personal notes are intentionally ignored in `.gitignore`.
