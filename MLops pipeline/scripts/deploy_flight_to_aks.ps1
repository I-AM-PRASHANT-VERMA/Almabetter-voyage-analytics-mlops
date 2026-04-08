
# -----------------------------
# 1. Read deployment inputs
# -----------------------------
param(
    [string]$ResourceGroup = "rg-voyage-mlops-v2",
    [string]$AcrName = "acrvoyagemlopsv2",
    [string]$AksName = "aks-voyage-mlops-v2",
    [string]$KeyVaultName = "kv-voyage-mlops-v2",
    [string]$ImageTag = "",
    [string]$Namespace = "voyage-mlops"
)

# -----------------------------
# 2. Resolve local project paths
# -----------------------------
$ErrorActionPreference = "Stop"  # stop the script as soon as one deployment command fails.

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$mlopsRoot = Resolve-Path (Join-Path $projectRoot "MLops pipeline")

# -----------------------------
# 3. Build image names
# -----------------------------
if ([string]::IsNullOrWhiteSpace($ImageTag)) {
    $ImageTag = (git -C $projectRoot rev-parse --short=12 HEAD).Trim()
}

$loginServer = (az acr show --name $AcrName --query loginServer -o tsv).Trim()
$flightApiImage = "$loginServer/voyage-v2/flight-api:$ImageTag"
$flightStreamlitImage = "$loginServer/voyage-v2/flight-streamlit:$ImageTag"

# -----------------------------
# 4. Build and push Docker images
# -----------------------------
Write-Host "Building flight images with tag $ImageTag"
docker build -t $flightApiImage -f (Join-Path $mlopsRoot "Dockerfile") $projectRoot
docker build -t $flightStreamlitImage -f (Join-Path $mlopsRoot "Dockerfile.streamlit") $projectRoot

Write-Host "Pushing images to ACR"
az acr login --name $AcrName
docker push $flightApiImage
docker push $flightStreamlitImage

# -----------------------------
# 5. Configure AKS access
# -----------------------------
Write-Host "Configuring AKS context"
az aks get-credentials --resource-group $ResourceGroup --name $AksName --overwrite-existing

$connectionString = az keyvault secret show `
    --vault-name $KeyVaultName `
    --name applicationinsights-connection-string `
    --query value `
    -o tsv

# -----------------------------
# 6. Apply Kubernetes manifests
# -----------------------------
Write-Host "Applying Kubernetes manifests"
kubectl apply -f (Join-Path $mlopsRoot "k8s\azure\namespace.yaml")
kubectl -n $Namespace create secret generic appinsights-connection `
    --from-literal=connection-string=$connectionString `
    --dry-run=client `
    -o yaml | kubectl apply -f -

kubectl apply -k (Join-Path $mlopsRoot "k8s\azure")
kubectl -n $Namespace set image deployment/flight-api "flight-api=$flightApiImage"
kubectl -n $Namespace set image deployment/flight-streamlit "flight-streamlit=$flightStreamlitImage"

# -----------------------------
# 7. Verify deployment rollout
# -----------------------------
kubectl -n $Namespace rollout status deployment/flight-api --timeout=300s
kubectl -n $Namespace rollout status deployment/flight-streamlit --timeout=300s
kubectl -n $Namespace get pods,svc -o wide
