# Jenkins CI/CD Workflow

Jenkins validates the project automatically against the GitHub `main` branch. It checks tracked models, drift, retraining inputs, and Docker configuration. Azure deployment remains disabled while the cloud environment is stopped.

## Main Files

| File | Purpose |
| --- | --- |
| `MLops pipeline/Jenkinsfile` | Local CI/CD pipeline |
| `MLops pipeline/jenkins/local-mounted-pipeline.groovy` | Optional mounted-workspace fallback for local Docker CD |
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

Every successful CI run calls the gated CD helper. The helper exits without contacting Azure while `VOYAGE_AZURE_DEPLOYMENT_ENABLED=false`.

If the flight dataset, training code, or drift state requires a new model, CI runs the MLflow training and promotion workflow before completing validation.

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

The bootstrap keeps the Azure job disabled while the environment switch is false. When the same Azure environment is ready again, set the following local environment value and restart the Jenkins container:

```text
VOYAGE_AZURE_DEPLOYMENT_ENABLED=true
```

After that, successful GitHub CI builds can call the CD job automatically. Its stages include:

- project validation
- dataset drift and conditional model retraining
- Azure login check
- Docker image build and push to ACR
- start the existing AKS cluster when it is stopped
- Kubernetes deployment to AKS
- gateway smoke tests

The CD job reads its pipeline and application files from a clean GitHub `main` checkout. It does not deploy from a stale mounted folder.

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

Turning the Azure switch on may start the existing AKS cluster and resume Azure costs. Keep it false until the subscription and cluster are ready. A failed validation, login, image push, rollout, or smoke test stops the remaining stages.
