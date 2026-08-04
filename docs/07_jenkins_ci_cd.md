# Jenkins CI/CD Workflow

Jenkins validates the project automatically against the GitHub `main` branch. It checks the tracked models and Docker configuration. Azure deployment remains disabled while the cloud environment is stopped.

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

The Jenkins container uses `restart: unless-stopped`, so it comes back when Docker Desktop starts. Docker Desktop and the computer still need to be running for local Jenkins automation.

## Automatic GitHub CI

The local CI job checks out a clean copy of GitHub `main` before every build. It supports two triggers:

- GitHub push webhook at `https://your-safe-public-endpoint/github-webhook/`
- SCM polling every five minutes as a local fallback

Do not point GitHub at `localhost`; GitHub cannot reach a private computer address. Only add the webhook after a stable HTTPS endpoint is available. The polling fallback needs no public tunnel and detects new commits while Jenkins is running.

Automatic builds keep `BUILD_DOCKER_IMAGES` and `ENABLE_DEPLOYMENT` set to `false`, so push events run validation only.

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

The Azure pipeline job is currently disabled. Its saved stages include:

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
