
pipeline {
    // -----------------------------
    // 1. Jenkins run behavior
    // -----------------------------
    // Let Jenkins run this pipeline on the built-in executor for the current local setup.
    agent any

    // -----------------------------
    // 2. Pipeline safety options
    // -----------------------------
    options {
        // Add timestamps so every deploy log is easier to read later.
        timestamps()
        // Prevent two Azure deploys from clashing with each other.
        disableConcurrentBuilds()
    }

    // -----------------------------
    // 3. Deployment switches
    // -----------------------------
    parameters {
        // Keep AKS deploy optional so the same pipeline can still be used for validation-only runs.
        booleanParam(name: 'DEPLOY_TO_AKS', defaultValue: false, description: 'Deploy the Voyage apps to Azure AKS after validation.')
        booleanParam(name: 'START_AKS_IF_STOPPED', defaultValue: true, description: 'Start the existing AKS cluster when deployment is enabled and the cluster is stopped.')
        booleanParam(name: 'FORCE_RETRAIN', defaultValue: false, description: 'Retrain the flight model even when its dataset fingerprint is unchanged.')
        // Allow a manual image tag override when we want a predictable release tag.
        string(name: 'IMAGE_TAG', defaultValue: '', description: 'Optional Docker image tag. Empty uses the current git commit.')
    }

    // -----------------------------
    // 4. Azure and project settings
    // -----------------------------
    environment {
        // The Pipeline-from-SCM job checks out the current GitHub main branch here.
        REPO_ROOT = "${WORKSPACE}"
        PROJECT_ROOT = "${WORKSPACE}/MLops pipeline"
        // Create one temporary virtual environment for Python validation steps.
        VENV_DIR = '/tmp/voyage-jenkins-cd-venv'
        // Reuse the Azure Container Registry already created for this project.
        ACR_NAME = 'acrvoyagemlopsv2'
        // Reuse the Azure Kubernetes Service cluster already created for this project.
        AKS_NAME = 'aks-voyage-mlops-v2'
        // Reuse the resource group that owns the AKS cluster and related resources.
        RESOURCE_GROUP = 'rg-voyage-mlops-v2'
        // Read monitoring secrets from the existing Key Vault.
        KEY_VAULT_NAME = 'kv-voyage-mlops-v2'
        // Force unbuffered Python logs so Jenkins console output stays live.
        PYTHONUNBUFFERED = '1'
        PIP_CACHE_DIR = '/var/jenkins_home/pip-cache'
    }

    stages {
        // -----------------------------
        // 5. Validate local project first
        // -----------------------------
        stage('Validate Project') {
            steps {
                sh '''
                    # Stop immediately if any validation step fails.
                    set -e
                    # Remove any old temporary virtual environment from a previous run.
                    rm -rf "$VENV_DIR"
                    # Create a fresh temporary Python virtual environment for this Jenkins run.
                    python3 -m venv "$VENV_DIR"
                    # Activate the temporary environment so pip and python use the same interpreter.
                    . "$VENV_DIR/bin/activate"
                    # Move into the actual project folder before installing requirements.
                    cd "$PROJECT_ROOT"
                    # Upgrade pip so dependency installation is less fragile.
                    python -m pip install --upgrade pip
                    # Install the project requirements used by the validators.
                    pip install -r requirements.txt
                    # Run the broad CI validator for models, files, and local workflow pieces.
                    python scripts/validate_jenkins_ci.py --output-dir jenkins_artifacts/jenkins_cd_validation
                    # Run the dedicated flight regression validator because flight is still the main training path.
                    python scripts/validate_flight_regression_workflow.py --output-dir jenkins_artifacts/jenkins_cd_flight_validation
                    # Save local health evidence without making optional Airflow or stopped Azure services block CD validation.
                    python scripts/monitor_voyage_workflow.py --output-dir monitoring_reports/jenkins_cd_local_monitor
                '''
            }
        }

        stage('Assess And Retrain Flight Model') {
            when {
                expression { params.DEPLOY_TO_AKS }
            }
            steps {
                sh '''
                    set -e
                    . "$VENV_DIR/bin/activate"
                    cd "$PROJECT_ROOT"
                    python scripts/check_flight_dataset_drift.py
                    FORCE_ARG=""
                    if [ "$FORCE_RETRAIN" = "true" ]; then
                        FORCE_ARG="--force"
                    fi
                    python scripts/assess_flight_retraining.py \
                        --properties-output jenkins_artifacts/cd_retraining.properties \
                        $FORCE_ARG
                    . jenkins_artifacts/cd_retraining.properties

                    if [ "$RETRAIN_REQUIRED" = "true" ]; then
                        export MLFLOW_ALLOW_FILE_STORE=true
                        python scripts/run_flight_price_mlflow_experiments.py --profile standard
                        python scripts/validate_flight_regression_workflow.py --output-dir jenkins_artifacts/jenkins_cd_post_training_validation
                    else
                        echo "Tracked serving model already matches the flight dataset."
                    fi
                '''
            }
        }

        // -----------------------------
        // 6. Check Azure login
        // -----------------------------
        stage('Azure Login Check') {
            when {
                // Skip Azure auth when the run is meant to be validation only.
                expression { params.DEPLOY_TO_AKS }
            }
            steps {
                withCredentials([
                    // Read the service principal id and secret from Jenkins credentials.
                    usernamePassword(credentialsId: 'voyage-azure-sp', usernameVariable: 'AZURE_CLIENT_ID', passwordVariable: 'AZURE_CLIENT_SECRET'),
                    // Read the tenant id from Jenkins credentials.
                    string(credentialsId: 'voyage-azure-tenant-id', variable: 'AZURE_TENANT_ID'),
                    // Read the subscription id from Jenkins credentials.
                    string(credentialsId: 'voyage-azure-subscription-id', variable: 'AZURE_SUBSCRIPTION_ID')
                ]) {
                    sh '''
                        # Stop immediately if Azure login fails.
                        set -e
                        # Sign in to Azure using the service principal created for Jenkins.
                        az login --service-principal \
                            --username "$AZURE_CLIENT_ID" \
                            --password "$AZURE_CLIENT_SECRET" \
                            --tenant "$AZURE_TENANT_ID" >/dev/null
                        # Select the subscription that owns the Voyage project resources.
                        az account set --subscription "$AZURE_SUBSCRIPTION_ID"
                        ACCOUNT_STATE="$(az account show --query state -o tsv)"
                        if [ "$ACCOUNT_STATE" != "Enabled" ]; then
                            echo "Azure deployment stopped: subscription state is $ACCOUNT_STATE."
                            exit 3
                        fi
                        az account show --query '{name:name,state:state}' -o table
                    '''
                }
            }
        }

        // -----------------------------
        // 7. Build and push app images
        // -----------------------------
        stage('Build And Push App Images') {
            when {
                // Skip image work when the run is meant to be validation only.
                expression { params.DEPLOY_TO_AKS }
            }
            steps {
                sh '''
                    # Stop immediately if any build or push command fails.
                    set -e
                    # Move to the repository root because docker build uses root-level COPY instructions.
                    cd "$REPO_ROOT"
                    # Start from the optional user-provided image tag.
                    IMAGE_TAG_VALUE="$IMAGE_TAG"
                    # Use the current git commit when no explicit tag was passed.
                    if [ -z "$IMAGE_TAG_VALUE" ]; then
                        IMAGE_TAG_VALUE="$(git rev-parse --short=12 HEAD)"
                    fi
                    # Read the Azure Container Registry login server once for all image pushes.
                    LOGIN_SERVER="$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)"
                    # Sign Docker into ACR before building tags that will be pushed.
                    az acr login -n "$ACR_NAME"
                    # Build the shared API image for the flight API service tag.
                    docker build -t "$LOGIN_SERVER/voyage-v2/flight-api:$IMAGE_TAG_VALUE" -f "MLops pipeline/Dockerfile" .
                    # Build the shared API image for the hotel API service tag.
                    docker build -t "$LOGIN_SERVER/voyage-v2/hotel-api:$IMAGE_TAG_VALUE" -f "MLops pipeline/Dockerfile" .
                    # Build the shared API image for the gender API service tag.
                    docker build -t "$LOGIN_SERVER/voyage-v2/gender-api:$IMAGE_TAG_VALUE" -f "MLops pipeline/Dockerfile" .
                    # Build the shared Streamlit image for the flight UI service tag.
                    docker build -t "$LOGIN_SERVER/voyage-v2/flight-streamlit:$IMAGE_TAG_VALUE" -f "MLops pipeline/Dockerfile.streamlit" .
                    # Build the shared Streamlit image for the hotel UI service tag.
                    docker build -t "$LOGIN_SERVER/voyage-v2/hotel-streamlit:$IMAGE_TAG_VALUE" -f "MLops pipeline/Dockerfile.streamlit" .
                    # Build the shared Streamlit image for the gender UI service tag.
                    docker build -t "$LOGIN_SERVER/voyage-v2/gender-streamlit:$IMAGE_TAG_VALUE" -f "MLops pipeline/Dockerfile.streamlit" .
                    # Push the flight API image to ACR.
                    docker push "$LOGIN_SERVER/voyage-v2/flight-api:$IMAGE_TAG_VALUE"
                    # Push the hotel API image to ACR.
                    docker push "$LOGIN_SERVER/voyage-v2/hotel-api:$IMAGE_TAG_VALUE"
                    # Push the gender API image to ACR.
                    docker push "$LOGIN_SERVER/voyage-v2/gender-api:$IMAGE_TAG_VALUE"
                    # Push the flight Streamlit image to ACR.
                    docker push "$LOGIN_SERVER/voyage-v2/flight-streamlit:$IMAGE_TAG_VALUE"
                    # Push the hotel Streamlit image to ACR.
                    docker push "$LOGIN_SERVER/voyage-v2/hotel-streamlit:$IMAGE_TAG_VALUE"
                    # Push the gender Streamlit image to ACR.
                    docker push "$LOGIN_SERVER/voyage-v2/gender-streamlit:$IMAGE_TAG_VALUE"
                    # Make sure the artifact folder exists before saving release metadata.
                    mkdir -p "$PROJECT_ROOT/jenkins_artifacts"
                    # Save the chosen image tag so the next stage uses the exact same release value.
                    echo "$IMAGE_TAG_VALUE" > "$PROJECT_ROOT/jenkins_artifacts/jenkins_cd_image_tag.txt"
                '''
            }
        }

        stage('Ensure Existing AKS Is Running') {
            when {
                expression { params.DEPLOY_TO_AKS }
            }
            steps {
                sh '''
                    set -e
                    POWER_STATE="$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query powerState.code -o tsv)"
                    if [ "$POWER_STATE" != "Running" ]; then
                        if [ "$START_AKS_IF_STOPPED" != "true" ]; then
                            echo "AKS is $POWER_STATE and automatic start is disabled."
                            exit 4
                        fi
                        echo "Starting the existing AKS cluster before deployment."
                        az aks start -g "$RESOURCE_GROUP" -n "$AKS_NAME"
                    fi

                    POWER_STATE="$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query powerState.code -o tsv)"
                    if [ "$POWER_STATE" != "Running" ]; then
                        echo "AKS did not reach Running state: $POWER_STATE"
                        exit 5
                    fi
                '''
            }
        }

        // -----------------------------
        // 8. Deploy to AKS
        // -----------------------------
        stage('Deploy To AKS') {
            when {
                // Skip AKS work when the run is meant to be validation only.
                expression { params.DEPLOY_TO_AKS }
            }
            steps {
                sh '''
                    # Stop immediately if any deployment command fails.
                    set -e
                    # Move into the project folder so kubectl can see the local manifests.
                    cd "$PROJECT_ROOT"
                    # Read back the exact image tag prepared by the build stage.
                    IMAGE_TAG_VALUE="$(cat jenkins_artifacts/jenkins_cd_image_tag.txt)"
                    # Read the Azure Container Registry login server again inside this stage.
                    LOGIN_SERVER="$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)"
                    # Read the Application Insights connection string from Key Vault.
                    CONNECTION_STRING="$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name applicationinsights-connection-string --query value -o tsv)"
                    # Pull the AKS kubeconfig into the Jenkins container.
                    az aks get-credentials -g "$RESOURCE_GROUP" -n "$AKS_NAME" --overwrite-existing
                    # Ensure the target namespace exists before applying the rest of the manifests.
                    kubectl apply -f k8s/azure/namespace.yaml
                    # Create or update the shared monitoring secret used by every app deployment.
                    kubectl -n voyage-mlops create secret generic appinsights-connection \
                        --from-literal=connection-string="$CONNECTION_STRING" \
                        --dry-run=client -o yaml | kubectl apply -f -
                    # Apply the full Azure kustomization that now includes all three apps and the gateway.
                    kubectl apply -k k8s/azure
                    # Point the flight API deployment to the freshly pushed release image.
                    kubectl -n voyage-mlops set image deployment/flight-api "flight-api=$LOGIN_SERVER/voyage-v2/flight-api:$IMAGE_TAG_VALUE"
                    # Point the hotel API deployment to the freshly pushed release image.
                    kubectl -n voyage-mlops set image deployment/hotel-api "hotel-api=$LOGIN_SERVER/voyage-v2/hotel-api:$IMAGE_TAG_VALUE"
                    # Point the gender API deployment to the freshly pushed release image.
                    kubectl -n voyage-mlops set image deployment/gender-api "gender-api=$LOGIN_SERVER/voyage-v2/gender-api:$IMAGE_TAG_VALUE"
                    # Point the flight Streamlit deployment to the freshly pushed release image.
                    kubectl -n voyage-mlops set image deployment/flight-streamlit "flight-streamlit=$LOGIN_SERVER/voyage-v2/flight-streamlit:$IMAGE_TAG_VALUE"
                    # Point the hotel Streamlit deployment to the freshly pushed release image.
                    kubectl -n voyage-mlops set image deployment/hotel-streamlit "hotel-streamlit=$LOGIN_SERVER/voyage-v2/hotel-streamlit:$IMAGE_TAG_VALUE"
                    # Point the gender Streamlit deployment to the freshly pushed release image.
                    kubectl -n voyage-mlops set image deployment/gender-streamlit "gender-streamlit=$LOGIN_SERVER/voyage-v2/gender-streamlit:$IMAGE_TAG_VALUE"
                    # Wait until the flight API rollout finishes successfully.
                    kubectl -n voyage-mlops rollout status deployment/flight-api --timeout=300s
                    # Wait until the hotel API rollout finishes successfully.
                    kubectl -n voyage-mlops rollout status deployment/hotel-api --timeout=300s
                    # Wait until the gender API rollout finishes successfully.
                    kubectl -n voyage-mlops rollout status deployment/gender-api --timeout=300s
                    # Wait until the flight Streamlit rollout finishes successfully.
                    kubectl -n voyage-mlops rollout status deployment/flight-streamlit --timeout=300s
                    # Wait until the hotel Streamlit rollout finishes successfully.
                    kubectl -n voyage-mlops rollout status deployment/hotel-streamlit --timeout=300s
                    # Wait until the gender Streamlit rollout finishes successfully.
                    kubectl -n voyage-mlops rollout status deployment/gender-streamlit --timeout=300s
                    # Wait until the shared gateway rollout finishes successfully.
                    kubectl -n voyage-mlops rollout status deployment/voyage-gateway --timeout=300s
                    # Capture the latest cluster state as a Jenkins artifact for later review.
                    kubectl -n voyage-mlops get pods,svc -o wide > jenkins_artifacts/jenkins_cd_aks_status.txt
                '''
            }
        }

        // -----------------------------
        // 9. Smoke test AKS gateway
        // -----------------------------
        stage('AKS Gateway Smoke Test') {
            when {
                // Skip gateway checks when the run is meant to be validation only.
                expression { params.DEPLOY_TO_AKS }
            }
            steps {
                sh '''
                    # Stop immediately if any smoke-test command fails.
                    set -e
                    # Move into the project folder so artifact paths stay consistent.
                    cd "$PROJECT_ROOT"
                    # Remove any older temporary curl pod before creating a fresh one.
                    kubectl -n voyage-mlops delete pod voyage-gateway-smoke --ignore-not-found=true
                    # Start a short-lived curl pod that can call the internal gateway service.
                    kubectl -n voyage-mlops run voyage-gateway-smoke --image=curlimages/curl:8.8.0 --restart=Never --command -- sleep 300
                    # Wait until the curl pod becomes ready before running requests through it.
                    kubectl -n voyage-mlops wait --for=condition=Ready pod/voyage-gateway-smoke --timeout=120s
                    # Check the gateway health route through the in-cluster service.
                    kubectl -n voyage-mlops exec voyage-gateway-smoke -- curl -fsS http://voyage-gateway/health
                    # Check the flight API route through the shared gateway.
                    kubectl -n voyage-mlops exec voyage-gateway-smoke -- curl -fsS http://voyage-gateway/api/flight/health
                    # Check the hotel API route through the shared gateway.
                    kubectl -n voyage-mlops exec voyage-gateway-smoke -- curl -fsS http://voyage-gateway/api/hotel/health
                    # Check the gender API route through the shared gateway.
                    kubectl -n voyage-mlops exec voyage-gateway-smoke -- curl -fsS http://voyage-gateway/api/gender/health
                    # Check the flight Streamlit health route through the shared gateway.
                    kubectl -n voyage-mlops exec voyage-gateway-smoke -- curl -fsS http://voyage-gateway/app/flight/_stcore/health
                    # Check the hotel Streamlit health route through the shared gateway.
                    kubectl -n voyage-mlops exec voyage-gateway-smoke -- curl -fsS http://voyage-gateway/app/hotel/_stcore/health
                    # Check the gender Streamlit health route through the shared gateway.
                    kubectl -n voyage-mlops exec voyage-gateway-smoke -- curl -fsS http://voyage-gateway/app/gender/_stcore/health
                    # Save the public gateway service summary so the external IP is easy to find later.
                    kubectl -n voyage-mlops get svc voyage-gateway -o wide > jenkins_artifacts/jenkins_cd_gateway_service.txt
                    # Clean up the temporary curl pod after all smoke tests pass.
                    kubectl -n voyage-mlops delete pod voyage-gateway-smoke --ignore-not-found=true
                '''
            }
        }
    }

    // -----------------------------
    // 10. Archive CD evidence
    // -----------------------------
    post {
        always {
            dir("${env.PROJECT_ROOT}") {
                // Archive every Jenkins artifact folder that this pipeline touched.
                archiveArtifacts artifacts: 'jenkins_artifacts/**, monitoring_reports/**', allowEmptyArchive: true, fingerprint: true
            }
        }
        cleanup {
            script {
                sh 'rm -rf "$VENV_DIR"'
                if (params.DEPLOY_TO_AKS) {
                    sh 'kubectl -n voyage-mlops delete pod voyage-gateway-smoke --ignore-not-found=true >/dev/null 2>&1 || true'
                }
            }
        }
    }
}
