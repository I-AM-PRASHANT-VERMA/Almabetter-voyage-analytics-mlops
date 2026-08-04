
# -----------------------------
# 1. Base Jenkins image
# -----------------------------
# Start from the official Jenkins LTS image with Java 17.
FROM jenkins/jenkins:lts-jdk17

USER root

# -----------------------------
# 2. Install local CI tools
# -----------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        curl \
        ca-certificates \
        docker-cli \
        docker-compose \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 3. Install Azure deployment tools
# -----------------------------
# Azure CLI and kubectl are needed only for the Azure CD job.
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && az aks install-cli --install-location /usr/local/bin/kubectl

# -----------------------------
# 4. Install Jenkins plugins
# -----------------------------
# Install only the plugins needed for the local pipeline demo.
RUN jenkins-plugin-cli --plugins \
        workflow-aggregator \
        git \
        github \
        credentials-binding \
        ws-cleanup \
        timestamper \
        pipeline-stage-view

# -----------------------------
# 5. Copy startup configuration
# -----------------------------
# Seed the local Jenkins user and jobs.
COPY jenkins/init-voyage.groovy /usr/share/jenkins/ref/init.groovy.d/init-voyage.groovy

# -----------------------------
# 6. Keep Docker access available
# -----------------------------
# The local Jenkins container needs Docker CLI access for the CD demo.
# It runs as root only in this local setup so it can use the mounted Docker socket.
USER root
