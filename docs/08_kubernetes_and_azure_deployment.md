# Kubernetes and Azure Deployment

This project includes Kubernetes manifests for local-style deployment and Azure AKS deployment. The Azure files are used with Azure Container Registry, AKS, and Application Insights connection settings.

## Main Folders

| Folder | Purpose |
| --- | --- |
| `MLops pipeline/k8s` | Kubernetes manifests for local or generic cluster deployment |
| `MLops pipeline/k8s/azure` | Azure AKS deployment manifests |
| `MLops pipeline/scripts/deploy_flight_to_aks.ps1` | PowerShell helper for flight deployment |
| `MLops pipeline/jenkins/azure-aks-cd-pipeline.groovy` | Jenkins pipeline for AKS deployment |

## Azure AKS Components

The Azure Kubernetes setup includes:

- namespace
- flight API deployment and service
- flight Streamlit deployment and service
- hotel API deployment and service
- hotel Streamlit deployment and service
- gender API deployment and service
- gender Streamlit deployment and service
- Nginx gateway deployment and service
- kustomization file for applying all Azure manifests together

## Deployment Flow

```text
Validate current GitHub main
  -> retrain and promote the flight model when required
  -> build Docker images
  -> push images to Azure Container Registry
  -> start the existing AKS cluster when stopped
  -> apply Kubernetes manifests
  -> update deployment images
  -> wait for rollout status
  -> run gateway smoke tests
```

## Jenkins-Based Azure Deployment

The Azure Jenkins pipeline performs the deployment through these main actions:

1. Validate the project from a clean GitHub `main` checkout.
2. Check drift and retrain the flight model when required.
3. Log in to the existing Azure subscription using Jenkins credentials.
4. Build and push API and Streamlit images to the existing ACR.
5. Start the existing AKS cluster when it is stopped.
6. Apply Kubernetes manifests and update deployment images.
7. Add the Application Insights connection string from Key Vault.
8. Run smoke tests through the gateway.

## Monitoring Connection

The Azure manifests read the Application Insights connection string from a Kubernetes secret named:

```text
appinsights-connection
```

The secret key used by the apps is:

```text
connection-string
```

## Expected Output

After deployment, the AKS namespace should contain running pods and services for the three APIs, three Streamlit apps, and the gateway.

## Notes

Azure credentials and real connection strings are not stored in this repository. They must be supplied through Jenkins credentials, Azure CLI, Key Vault, or Kubernetes secrets.

This workflow reuses the existing resource group, ACR, AKS cluster, Key Vault, and Application Insights instance. It does not create Azure infrastructure. Keep `VOYAGE_AZURE_DEPLOYMENT_ENABLED=false` while the subscription is disabled.
