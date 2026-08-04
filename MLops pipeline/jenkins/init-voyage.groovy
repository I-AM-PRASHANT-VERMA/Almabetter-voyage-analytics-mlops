
import hudson.model.ParametersDefinitionProperty
import hudson.model.BooleanParameterDefinition
import hudson.model.FreeStyleProject
import hudson.tasks.Shell
import hudson.plugins.git.BranchSpec
import hudson.plugins.git.GitSCM
import hudson.plugins.git.UserRemoteConfig
import hudson.plugins.git.extensions.impl.WipeWorkspace
import hudson.triggers.SCMTrigger
import jenkins.model.Jenkins
import hudson.security.HudsonPrivateSecurityRealm
import hudson.security.FullControlOnceLoggedInAuthorizationStrategy
import org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition
import org.jenkinsci.plugins.workflow.job.WorkflowJob
import com.cloudbees.jenkins.GitHubPushTrigger

// -----------------------------
// 1. Read Jenkins instance and credentials
// -----------------------------
// Read the live Jenkins instance so this bootstrap script can create users and jobs.
def instance = Jenkins.get()

// Allow local admin credentials to come from container env vars.
def adminId = System.getenv("JENKINS_ADMIN_ID") ?: "prashant_jenkins"
def adminPassword = System.getenv("JENKINS_ADMIN_PASSWORD") ?: "change-me-local"

// -----------------------------
// 2. Configure local Jenkins security
// -----------------------------
// Create the local Jenkins login user only when it does not already exist.
def realm = new HudsonPrivateSecurityRealm(false)
if (realm.getUser(adminId) == null) {
    realm.createAccount(adminId, adminPassword)
}
instance.setSecurityRealm(realm)

// Block anonymous access so the browser UI always requires login.
def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
strategy.setAllowAnonymousRead(false)
instance.setAuthorizationStrategy(strategy)

// -----------------------------
// 3. Create local CI job
// -----------------------------
// This is the main validation job used for local CI checks.
def jobName = "voyage-analytics-mlops-ci"
def job = instance.getItem(jobName)

if (job == null) {
    job = instance.createProject(FreeStyleProject, jobName)
}

// Keep the job description practical so the Jenkins dashboard stays readable.
job.setDescription("Voyage Analytics CI job. It checks out GitHub main, validates the tracked models and project files, and verifies Docker Compose without deploying to Azure.")

// Replace old parameter definitions so rerunning this init script stays idempotent.
job.removeProperty(ParametersDefinitionProperty)
job.addProperty(new ParametersDefinitionProperty(
    new BooleanParameterDefinition("BUILD_DOCKER_IMAGES", false, "Build Docker images after validation."),
    new BooleanParameterDefinition("ENABLE_DEPLOYMENT", false, "Keep this disabled until Kubernetes and Azure settings are confirmed.")
))

// Always validate the public GitHub main branch instead of the mounted host checkout.
def repoUrl = "https://github.com/I-AM-PRASHANT-VERMA/Almabetter-voyage-analytics-mlops.git"
def scm = new GitSCM(
    [new UserRemoteConfig(repoUrl, null, null, null)],
    [new BranchSpec("*/main")],
    false,
    [],
    null,
    null,
    [new WipeWorkspace()]
)
job.setScm(scm)

// Accept GitHub push events when a webhook is available. Polling is the local fallback.
def existingTriggers = job.getTriggers().values()
if (!existingTriggers.any { trigger -> trigger instanceof GitHubPushTrigger }) {
    job.addTrigger(new GitHubPushTrigger())
}

def pollingTrigger = existingTriggers.find { trigger -> trigger instanceof SCMTrigger }
if (pollingTrigger == null) {
    job.addTrigger(new SCMTrigger("H/5 * * * *"))
} else if (pollingTrigger.spec != "H/5 * * * *") {
    job.removeTrigger(pollingTrigger.descriptor)
    job.addTrigger(new SCMTrigger("H/5 * * * *"))
}

// -----------------------------
// 4. Define CI shell steps
// -----------------------------
// Clear any older shell builder and keep one current validation script.
job.getBuildersList().clear()
job.getBuildersList().add(new Shell('''
set -e

# Jenkins checks out the repository before this shell step runs.
PROJECT_ROOT="$WORKSPACE/MLops pipeline"
LOCAL_TRAINING_DIR="$WORKSPACE/local_training"
# Use a temporary venv inside the container so host files stay untouched.
VENV_DIR="/tmp/voyage-jenkins-ci-venv"
# Store validation outputs where Jenkins can archive them later.
JENKINS_OUTPUT_DIR="$WORKSPACE/jenkins_artifacts"
export LOCAL_TRAINING_DIR

python3 --version
test -d "$PROJECT_ROOT"
test -f "$PROJECT_ROOT/Jenkinsfile"
test -f "$LOCAL_TRAINING_DIR/train_flight_price.py"

# Recreate the virtual environment on each run for a clean dependency state.
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Install project dependencies before running validation scripts.
cd "$PROJECT_ROOT"
pip install -r requirements.txt

# Run the main project checks and the dedicated flight workflow checks.
python scripts/validate_jenkins_ci.py --output-dir "$JENKINS_OUTPUT_DIR/ci_validation_from_jenkins_job"
python scripts/validate_flight_regression_workflow.py --output-dir "$JENKINS_OUTPUT_DIR/flight_validation_from_jenkins_job"

# Save the fully resolved compose file when Docker CLI is available in the container.
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose --profile mlops --profile mlflow config > "$JENKINS_OUTPUT_DIR/docker-compose-resolved-from-jenkins.yml"
else
    echo "Docker CLI is not available inside this Jenkins container." > "$JENKINS_OUTPUT_DIR/docker-validation-note.txt"
fi

# Optional image build flag for heavier CI runs.
if [ "$BUILD_DOCKER_IMAGES" = "true" ]; then
    docker compose --profile mlops --profile mlflow build \
        flight-api flight-streamlit \
        hotel-api hotel-streamlit \
        gender-api gender-streamlit \
        flight-validation \
        mlflow-ui mlflow-training
fi

# Deployment stays blocked here until CD settings are intentionally enabled.
if [ "$ENABLE_DEPLOYMENT" = "true" ]; then
    echo "Deployment is intentionally blocked until Kubernetes and Azure settings are confirmed."
    exit 2
fi

echo "Voyage Analytics Jenkins CI completed."
'''))

job.save()

// -----------------------------
// 5. Create AKS CD pipeline job
// -----------------------------
// This second job reads the Groovy pipeline file and handles Docker + AKS CD.
def cdJobName = "voyage-analytics-mlops-cd"
def cdJob = instance.getItem(cdJobName)

if (cdJob == null) {
    cdJob = instance.createProject(WorkflowJob, cdJobName)
}

// Load the CD pipeline script from the mounted repo so source control stays the single truth.
def cdPipelineFile = new File("/workspace/voyage-analytics-mlops/jenkins/azure-aks-cd-pipeline.groovy")
def cdPipelineScript = cdPipelineFile.exists() ? cdPipelineFile.text : """
// Start the Jenkins pipeline definition.
pipeline {
    agent any
    stages {
        stage('Missing Pipeline File') {
            steps {
                error('jenkins/azure-aks-cd-pipeline.groovy was not found in the mounted workspace.')
            }
        }
    }
}
"""

cdJob.setDescription("Voyage Analytics Azure CD job. It is disabled while the Azure environment is stopped.")
cdJob.setDefinition(new CpsFlowDefinition(cdPipelineScript, true))
cdJob.setDisabled(true)
cdJob.save()

// -----------------------------
// 6. Save Jenkins configuration
// -----------------------------
// Persist the updated Jenkins security and job configuration.
instance.save()
