# Notebooks and Modeling Workflow

The notebooks show the research and experimentation part of the project. They are useful for explaining the business problem, exploring the datasets, comparing model results, and documenting reasoning.

The local MLOps code is not a replacement for the notebooks. It is the next step after notebook experimentation. It converts selected project work into scripts, services, containers, and deployment files.

## Notebook Files

| Notebook | Purpose |
| --- | --- |
| `EDA_Notebook_/flight_and_hotel_EDA.ipynb` | Explores users, flights, and hotels data |
| `Flight Price Prediction Regression ML Model/flight_price_regression_model.ipynb` | Builds and explains the flight price regression model |
| `hotel_recommendation_ML_model/hotel_recommendation.ipynb` | Builds and explains hotel recommendation logic |
| `Gender_classification_ML_model/Gender_classification_model.ipynb` | Builds and explains gender classification |

## Why Notebooks Are Kept

Notebooks are kept because they show how the work was developed. They include the analysis, modeling choices, evaluation results, and explanation paragraphs. This is useful for project review and video presentation.

## Why Local MLOps Code Is Also Needed

Colab notebooks are good for experimentation, but they are not the best place to run a complete MLOps setup with Docker, Airflow, Jenkins, APIs, Kubernetes, and Azure deployment files.

The local MLOps pipeline exists to show how the selected work can be made repeatable and deployable.

## Relationship Between Notebook and Local Pipeline

| Notebook side | Local MLOps side |
| --- | --- |
| Explores data | Uses fixed dataset paths |
| Tries models | Runs repeatable training scripts |
| Shows reasoning | Saves reproducible metrics and artifacts |
| Good for explanation | Good for automation and deployment |
| Runs in Colab | Runs locally through scripts and containers |

## Regression Model Focus

The MLOps workflow mainly focuses on the flight price regression model because the assignment asks the video presentation to explain the regression model through modeling, deployment, scheduling, pipelining, and monitoring.

The hotel and gender components are also included as working model applications, but the deepest MLOps workflow is built around the flight price regression flow.
