# Jenkins CI/CD Workflow

Jenkins is used to demonstrate CI/CD for the MLOps project. It validates the project, checks Docker configuration, can build images, and includes an Azure AKS deployment pipeline.

## Main Files

| File | Purpose |
| --- | --- |
| `MLops pipeline/Jenkinsfile` | Local CI/CD pipeline |
| `MLops pipeline/jenkins/local-mounted-pipeline.groovy` | Local Jenkins job definition |
| `MLops pipeline/jenkins/azure-aks-cd-pipeline.groovy` | Azure AKS deployment pipeline |
| `MLops pipeline/jenkins/Dockerfile` | Jenkins image setup |
| `MLops pipeline/jenkins/init-voyage.groovy` | Jenkins initialization |

## Start Jenkins Locally

From the `MLops pipeline` folder:

```bash
docker compose --profile jenkins up --build
```

Open Jenkins:

```text
http://localhost:8081
```

Default local credentials are configured through environment variables in Docker Compose. The placeholder password is only for the local demo.

## Local CI Pipeline Stages

The local `Jenkinsfile` includes stages such as:

- agent and workspace checks
- Python environment setup
- Jenkins-style validation
- Docker Compose config check
- optional Docker image build
- optional local Docker deployment
- deployment gate

## Azure AKS CD Pipeline Stages

The Azure pipeline includes:

- project validation
- Azure login check
- Docker image build and push to ACR
- Kubernetes deployment to AKS
- gateway smoke tests

## Credentials

Azure deployment expects Jenkins credentials for:

- Azure service principal id
- Azure service principal secret
- Azure tenant id
- Azure subscription id

The actual secret values are not stored in the repository.

## Expected Output

Jenkins should create validation artifacts under local artifact folders and show pipeline stage results in the Jenkins UI.

## Notes

The local Jenkins pipeline is useful for showing CI/CD structure even when the full cloud deployment is not run every time. The Azure pipeline is included to show how the project can be deployed to a Kubernetes environment.
