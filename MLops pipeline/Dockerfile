# Base image for the flight Flask service.
FROM python:3.11.15-slim-bookworm

# Keep Python output cleaner inside the container.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Everything in this image runs from /app.
WORKDIR /app

# XGBoost needs libgomp at runtime, so install it once during the image build.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements first so dependency installs stay cached across source edits.
COPY ["requirements.txt", "./"]

# Install the Python packages used by the Flask app and the shared model loader.
RUN pip install --no-cache-dir -r requirements.txt

# The model folder is mounted from the host, but creating it here keeps the path ready.
RUN mkdir -p "/app/joblib files"

# Only copy the pieces this service actually needs at runtime.
COPY ["flask_apps", "/app/flask_apps"]
COPY ["dataset", "/app/dataset"]
COPY ["joblib files", "/app/joblib files"]

# The Flask flight app listens on port 5002.
EXPOSE 5002

# Start the flight price Flask app when the container comes up.
CMD ["python", "flask_apps/flight_price_flask_app/app.py"]
