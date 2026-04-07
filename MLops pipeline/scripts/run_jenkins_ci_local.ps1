
# -----------------------------
# 1. Resolve project and Python runtime
# -----------------------------
$ErrorActionPreference = "Stop"  # stop immediately when a validation command fails.

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
# Move into the project folder before running commands.
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Project virtual environment was not found: $Python"
}

# -----------------------------
# 2. Run local validation checks
# -----------------------------
# Run the Jenkins validation command for the project.
& $Python scripts\validate_jenkins_ci.py --output-dir jenkins_artifacts\ci_validation
# Run the Jenkins validation command for the project.
& $Python scripts\validate_flight_regression_workflow.py --output-dir jenkins_artifacts\flight_validation

# -----------------------------
# 3. Check Docker availability
# -----------------------------
$dockerAvailable = $false
try {
    docker --version | Out-Null
    docker compose version | Out-Null
    $dockerAvailable = $true
}
catch {
    $dockerAvailable = $false
}

# -----------------------------
# 4. Validate Docker Compose config
# -----------------------------
if ($dockerAvailable) {
    docker compose --profile mlops --profile mlflow config > jenkins_artifacts\docker-compose-resolved.yml
    Write-Host "Docker Compose config validation completed."
}
else {
    # Run the Jenkins validation command for the project.
    New-Item -ItemType Directory -Force -Path jenkins_artifacts | Out-Null
    "Docker validation skipped because Docker was not available." | Set-Content jenkins_artifacts\docker-validation-note.txt
    Write-Host "Docker was not available, so Docker validation was skipped."
}

# Run the Jenkins validation command for the project.
Write-Host "Local Jenkins-equivalent CI checks completed."
