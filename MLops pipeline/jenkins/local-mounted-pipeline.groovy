
pipeline {
    agent any

    // -----------------------------
    // 1. Jenkins run behavior
    // -----------------------------
    options {
        // Add timestamps so Jenkins logs are easier to read during debugging.
        timestamps()
        // Avoid overlapping runs that could clash on Docker ports and artifacts.
        disableConcurrentBuilds()
    }

    // -----------------------------
    // 2. Manual pipeline switches
    // -----------------------------
    parameters {
        // Build images only when this extra cost is needed.
        booleanParam(name: 'BUILD_DOCKER_IMAGES', defaultValue: false, description: 'Build Docker images after validation.')
        // This flag turns on local restart + smoke test after image build.
        booleanParam(name: 'RUN_LOCAL_CD', defaultValue: false, description: 'Restart local Docker containers and verify app health after image build.')
        // Keep deployment blocked until cloud settings are confirmed.
        booleanParam(name: 'ENABLE_DEPLOYMENT', defaultValue: false, description: 'Keep this disabled until Kubernetes and Azure are confirmed.')
    }

    // -----------------------------
    // 3. Shared Jenkins paths
    // -----------------------------
    // Keep shared pipeline settings in one place.
    environment {
        // Mounted repo path available inside the Jenkins container.
        PROJECT_ROOT = '/workspace/voyage-analytics-mlops'
        // Root local training folder is mounted separately for promoted model checks.
        LOCAL_TRAINING_DIR = '/workspace/local_training'
        // Host path is needed when Jenkins prepares compose override volumes.
        HOST_PROJECT_ROOT = 'E:\\E almabetter projects\\1 Voyage_Analytics_MLOps\\MLops pipeline'
        // Disposable venv keeps Python installs isolated per run.
        VENV_DIR = '/tmp/voyage-jenkins-ci-venv'
        // Jenkins artifacts are saved here and then archived in post steps.
        JENKINS_OUTPUT_DIR = '/workspace/voyage-analytics-mlops/jenkins_artifacts'
        PYTHONUNBUFFERED = '1'
    }

    stages {
        // -----------------------------
        // 4. Confirm Jenkins agent
        // -----------------------------
        stage('Agent Check') {
            steps {
                // Run a shell command as part of the pipeline.
                sh '''
                    set -e
                    # Confirm the Jenkins agent can see Python and the mounted repo.
                    python3 --version
                    test -d "$PROJECT_ROOT"
                    test -f "$PROJECT_ROOT/Jenkinsfile"
                    mkdir -p "$JENKINS_OUTPUT_DIR"
                '''
            }
        }

        // -----------------------------
        // 5. Install Python dependencies
        // -----------------------------
        stage('Set Up Python') {
            steps {
                // Run a shell command as part of the pipeline.
                sh '''
                    set -e
                    # Rebuild the venv each run so stale packages do not affect CI.
                    rm -rf "$VENV_DIR"
                    python3 -m venv "$VENV_DIR"
                    . "$VENV_DIR/bin/activate"
                    python -m pip install --upgrade pip
                    cd "$PROJECT_ROOT"
                    pip install -r requirements.txt
                '''
            }
        }

        // -----------------------------
        // 6. Validate models and project files
        // -----------------------------
        stage('Validate Models And Project') {
            steps {
                // Run a shell command as part of the pipeline.
                sh '''
                    set -e
                    # Run the wide CI validator and the focused flight regression validator.
                    . "$VENV_DIR/bin/activate"
                    cd "$PROJECT_ROOT"
                    python scripts/validate_jenkins_ci.py --output-dir "$JENKINS_OUTPUT_DIR/ci_validation_from_jenkins_job"
                    python scripts/validate_flight_regression_workflow.py --output-dir "$JENKINS_OUTPUT_DIR/flight_validation_from_jenkins_job"
                '''
            }
        }

        // -----------------------------
        // 7. Validate Docker Compose config
        // -----------------------------
        stage('Docker Compose Config') {
            steps {
                // Run a shell command as part of the pipeline.
                sh '''
                    set -e
                    # Resolve compose config so broken references fail before deployment steps.
                    cd "$PROJECT_ROOT"
                    docker --version
                    docker compose version
                    docker compose config > "$JENKINS_OUTPUT_DIR/docker-compose-resolved-from-jenkins.yml"
                '''
            }
        }

        // -----------------------------
        // 8. Optional Docker build
        // -----------------------------
        stage('Docker Build') {
            when {
                // Build images when the user asked for image refresh or local CD.
                expression { params.BUILD_DOCKER_IMAGES || params.RUN_LOCAL_CD }
            }
            steps {
                // Run a shell command as part of the pipeline.
                sh '''
                    set -e
                    # Build only the serving apps used by the local browser flow.
                    cd "$PROJECT_ROOT"
                    docker compose build \
                        flight-api flight-streamlit \
                        hotel-api hotel-streamlit \
                        gender-api gender-streamlit
                '''
            }
        }

        // -----------------------------
        // 9. Optional local Docker CD
        // -----------------------------
        stage('Local Docker CD') {
            when {
                // Local CD here means restart containers and verify they come back healthy.
                expression { params.RUN_LOCAL_CD }
            }
            steps {
                // Run a shell command as part of the pipeline.
                sh '''
                    set -e
                    cd "$PROJECT_ROOT"

                    # Create a compose override so Jenkins uses the host datasets and joblib files.
                    cat > "$JENKINS_OUTPUT_DIR/docker-compose.jenkins-cd.yml" <<YAML
services:
  flight-api:
    volumes:
      - '${HOST_PROJECT_ROOT}\\dataset:/app/dataset:ro'
      - '${HOST_PROJECT_ROOT}\\joblib files:/app/joblib files:ro'
  flight-streamlit:
    volumes:
      - '${HOST_PROJECT_ROOT}\\dataset:/app/dataset:ro'
      - '${HOST_PROJECT_ROOT}\\joblib files:/app/joblib files:ro'
      - '${HOST_PROJECT_ROOT}\\.streamlit:/app/.streamlit:ro'
  hotel-api:
    volumes:
      - '${HOST_PROJECT_ROOT}\\dataset:/app/dataset:ro'
      - '${HOST_PROJECT_ROOT}\\joblib files:/app/joblib files:ro'
  hotel-streamlit:
    volumes:
      - '${HOST_PROJECT_ROOT}\\dataset:/app/dataset:ro'
      - '${HOST_PROJECT_ROOT}\\joblib files:/app/joblib files:ro'
      - '${HOST_PROJECT_ROOT}\\.streamlit:/app/.streamlit:ro'
  gender-api:
    volumes:
      - '${HOST_PROJECT_ROOT}\\dataset:/app/dataset:ro'
      - '${HOST_PROJECT_ROOT}\\joblib files:/app/joblib files:ro'
  gender-streamlit:
    volumes:
      - '${HOST_PROJECT_ROOT}\\dataset:/app/dataset:ro'
      - '${HOST_PROJECT_ROOT}\\joblib files:/app/joblib files:ro'
      - '${HOST_PROJECT_ROOT}\\.streamlit:/app/.streamlit:ro'
YAML

                    # Restart the six user-facing containers with the Jenkins override file.
                    docker compose -f docker-compose.yml -f "$JENKINS_OUTPUT_DIR/docker-compose.jenkins-cd.yml" up -d \
                        flight-api flight-streamlit \
                        hotel-api hotel-streamlit \
                        gender-api gender-streamlit

                    python3 - <<'PY'
import json
import time
import urllib.request

# Hit both APIs and Streamlit health endpoints until all of them respond.
checks = {
    "flight_api_health": "http://host.docker.internal:5002/health",
    "hotel_api_health": "http://host.docker.internal:5001/health",
    "gender_api_health": "http://host.docker.internal:5003/health",
    "flight_streamlit_health": "http://host.docker.internal:8501/app/flight/_stcore/health",
    "hotel_streamlit_health": "http://host.docker.internal:8502/app/hotel/_stcore/health",
    "gender_streamlit_health": "http://host.docker.internal:8503/app/gender/_stcore/health",
}

status = {}
for attempt in range(1, 19):
    failed = []
    for name, url in checks.items():
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                status[name] = response.status
        except Exception as error:
            status[name] = str(error)
            failed.append(name)
    if not failed:
        break
    time.sleep(10)
else:
    raise SystemExit(f"Local Docker CD health checks failed: {status}")

with open("jenkins_artifacts/local_cd_health_summary.json", "w", encoding="utf-8") as file:
    json.dump(status, file, indent=4)

print("Local Docker CD health checks passed.")
print(json.dumps(status, indent=4))
PY
                '''
            }
        }

        // -----------------------------
        // 10. Block accidental cloud deployment
        // -----------------------------
        stage('Deployment Gate') {
            steps {
                // Run custom Groovy logic for this part of the pipeline.
                script {
                    // Fail fast if someone tries to use this local CI job for cloud deployment.
                    if (params.ENABLE_DEPLOYMENT) {
                        error('Deployment is not enabled yet. Confirm Kubernetes and Azure settings first.')
                    }
                    echo 'Deployment is disabled for this run. CI validation is complete.'
                }
            }
        }
    }

    // -----------------------------
    // 11. Archive validation evidence
    // -----------------------------
    post {
        always {
            // Run a shell command as part of the pipeline.
            sh '''
                # Copy project artifacts into the Jenkins workspace before archiving.
                rm -rf "$WORKSPACE/jenkins_artifacts"
                cp -R "$PROJECT_ROOT/jenkins_artifacts" "$WORKSPACE/jenkins_artifacts" 2>/dev/null || true
            '''
            archiveArtifacts artifacts: 'jenkins_artifacts/**', allowEmptyArchive: true, fingerprint: true
        }
    }
}
