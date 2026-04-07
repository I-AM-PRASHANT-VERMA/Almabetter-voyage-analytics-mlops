
# -----------------------------
# 1. Base Jenkins image
# -----------------------------
FROM jenkins/jenkins:lts-jdk17  # start from the official Jenkins LTS image with Java 17.

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
        credentials-binding \
        ws-cleanup \
        timestamper \
        pipeline-stage-view

# -----------------------------
# 5. Copy startup configuration
# -----------------------------
COPY jenkins/init-voyage.groovy /usr/share/jenkins/ref/init.groovy.d/init-voyage.groovy  # seed the local Jenkins user and demo job.

# -----------------------------
# 6. Keep Docker access available
# -----------------------------
# The local Jenkins container needs Docker CLI access for the CD demo.
# It runs as root only in this local setup so it can use the mounted Docker socket.
USER root
