# Project Overview

Voyage Analytics is an end-to-end machine learning and MLOps project for travel data. The project uses users, flights, and hotels datasets to build analysis development workflows, machine learning models, APIs, dashboards, and deployment workflows.

The work is divided into two parts:

1. interactive development workflows for analysis, model development, and explanation.
2. Local MLOps code for repeatable training, serving, orchestration, CI/CD, deployment, and monitoring.

This separation is intentional. development workflows are better for exploration and presentation. The local MLOps pipeline is better for repeatable execution and deployment.

## Main Use Cases

| Use case | Purpose |
| --- | --- |
| Flight price prediction | Predict flight prices from travel and booking features |
| Hotel recommendation | Recommend hotels using travel and hotel data |
| Gender classification | Classify user gender from available user-related information |

## End-to-End Flow

```text
Dataset
  -> EDA and modeling development workflows
  -> local training scripts
  -> MLflow experiment tracking
  -> saved model artifacts
  -> Flask APIs
  -> Streamlit dashboards
  -> Docker Compose local setup
  -> Airflow training workflow
  -> Jenkins CI/CD checks
  -> Kubernetes and Azure deployment files
  -> monitoring and drift checks
```

## Important Folders

| Folder | Role |
| --- | --- |
| `EDA_development workflow_` | Exploratory data analysis |
| `Flight Price Prediction Regression ML Model` | Flight regression development workflow |
| `hotel_recommendation_ML_model` | Hotel recommendation development workflow |
| `Gender_classification_ML_model` | Gender classification development workflow |
| `local_training` | Local flight model training script |
| `MLops pipeline` | Production-style MLOps workflow |
| `dataset` | Source datasets and baseline files |

## What This Project Demonstrates

- Data analysis and model development
- Model training and evaluation
- MLflow experiment tracking
- REST API serving with Flask
- Streamlit dashboards
- Docker-based local orchestration
- Airflow workflow orchestration
- Jenkins CI/CD pipeline
- Kubernetes and Azure deployment structure
- Dataset drift and workflow monitoring
