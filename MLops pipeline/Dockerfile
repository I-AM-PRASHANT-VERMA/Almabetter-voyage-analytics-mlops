# -----------------------------
# 1. Base runtime image
# -----------------------------
# This image is only for serving the model through Flask, not for retraining.
# Slim Python image keeps the Flask API container smaller for local and cloud runs.
FROM python:3.11.15-slim-bookworm

# Keep Python output cleaner inside the container.
ENV PYTHONDONTWRITEBYTECODE=1

# Stream logs immediately so Docker, Jenkins, and Azure logs show live output.
ENV PYTHONUNBUFFERED=1

# Everything in this image runs from /app.
WORKDIR /app

# -----------------------------
# 2. Install runtime dependencies
# -----------------------------
# XGBoost needs libgomp at runtime, so install it once during the image build.
# The apt cache is removed in the same layer so the final image stays smaller.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 3. Install Python dependencies
# -----------------------------
# Copy requirements first so dependency installs stay cached across source edits.
COPY ["MLops pipeline/requirements.txt", "./"]

# Install the Python packages used by the Flask app and the shared model loader.
# These packages include Flask, pandas, joblib, model libraries, and monitoring support.
RUN pip install --no-cache-dir -r requirements.txt

# Keep the model path ready so Azure File mounts land in the expected place.
# The folder name matches the existing project artifact folder.
RUN mkdir -p "/app/joblib files"

# -----------------------------
# 4. Copy service files
# -----------------------------
# Only copy the pieces this service actually needs at runtime.
# Training development workflows, Airflow files, and CI files are not needed inside this API image.
COPY ["MLops pipeline/flask_apps", "/app/flask_apps"]

# Shared monitoring setup is imported by the Flask apps.
COPY ["MLops pipeline/monitoring.py", "/app/monitoring.py"]

# The API uses the dataset for metadata, samples, and route summaries.
# The prediction itself still comes from the saved model, not from retraining here.
COPY ["dataset", "/app/dataset"]

# Include the local model artifact for standalone image runs; Compose and AKS can mount over it.
COPY ["MLops pipeline/joblib files", "/app/joblib files"]

# -----------------------------
# 5. Expose and start service
# -----------------------------
# The Flask flight app listens on port 5002.
# Docker Compose maps this internal port to the host when running locally.
EXPOSE 5002

# Start the flight price Flask app when the container comes up.
# Compose can still add health checks, volumes, and environment variables around this command.
CMD ["python", "flask_apps/flight_price_flask_app/app.py"]
