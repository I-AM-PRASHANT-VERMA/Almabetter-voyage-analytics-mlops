import json
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import jsonify, render_template_string, request
from sklearn.exceptions import InconsistentVersionWarning
from werkzeug.exceptions import HTTPException

from monitoring import configure_flask_monitoring


# -----------------------------
# 1. Resolve shared Flask paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # points to the MLops pipeline folder
REPO_ROOT = BASE_DIR.parent  # fallback root for shared dataset files
DEFAULT_DATASET_DIR = (
    BASE_DIR / "dataset" / "travel_capstone"  # first try dataset inside pipeline folder
    if (BASE_DIR / "dataset" / "travel_capstone").exists()
    else REPO_ROOT / "dataset" / "travel_capstone"  # fallback to repo-level dataset folder
)
DATASET_DIR = Path(os.getenv("VOYAGE_DATASET_DIR", DEFAULT_DATASET_DIR))  # allow Docker/Kubernetes to override dataset location
JOBLIB_DIR = BASE_DIR / "joblib files"  # shared model artifact folder for the Flask apps
LOGGER = logging.getLogger("voyage_flask_apps")  # common logger used by shared helpers


def display_path(path):
    """Return a short path for API responses without assuming one project root."""
    resolved_path = Path(path).resolve()

    for root in (BASE_DIR, REPO_ROOT):
        try:
            return str(resolved_path.relative_to(root))
        except ValueError:
            continue

    return str(resolved_path)


ERROR_PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ service_name }} - {{ status_code }}</title>
    <style>
        body {
            font-family: "Segoe UI", Tahoma, sans-serif;
            background: #f5f7fb;
            color: #172033;
            margin: 0;
            padding: 2rem;
        }
        .shell {
            max-width: 720px;
            margin: 2rem auto;
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }
        .code {
            color: #1769e0;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.85rem;
        }
        h1 {
            margin: 0.5rem 0 0.75rem;
            font-size: 1.8rem;
        }
        p {
            line-height: 1.6;
            color: #4f5f75;
        }
    </style>
</head>
<body>
    <div class="shell">
        <div class="code">{{ status_code }}</div>
        <h1>{{ service_name }}</h1>
        <p>{{ message }}</p>
    </div>
</body>
</html>
"""


class AssetLoadError(RuntimeError):
    """Raised when a runtime dataset or model artifact is missing or unreadable."""


# -----------------------------
# 2. Check required runtime file
# -----------------------------
def ensure_runtime_file(path, label):
    path = Path(path)  # normalize string paths before checking them

    if not path.exists():
        raise AssetLoadError(f"{label} was not found: {path}")  # fail before model or dataset loading starts

    if path.is_file() and path.stat().st_size == 0:
        raise AssetLoadError(f"{label} is empty: {path}")  # empty artifacts should not be used at runtime


# -----------------------------
# 3. Decide JSON or browser response
# -----------------------------
def prefers_json_response():
    if request.path != "/":
        return True  # API-style routes should return JSON errors

    if request.is_json:
        return True  # JSON requests should receive JSON responses

    best_match = request.accept_mimetypes.best_match(["application/json", "text/html"])
    return best_match == "application/json"  # browser page keeps HTML when HTML is preferred


# -----------------------------
# 4. Build shared error response
# -----------------------------
def build_error_response(service_name, message, status_code):
    if prefers_json_response():
        # API clients should get a small predictable JSON error payload.
        return jsonify({"status": "error", "app_name": service_name, "message": message}), status_code

    # Browser users get a readable HTML page instead of raw JSON.
    return (
        render_template_string(
            ERROR_PAGE_TEMPLATE,
            service_name=service_name,
            message=message,
            status_code=status_code,
        ),
        status_code,
    )


# -----------------------------
# 5. Register shared Flask errors
# -----------------------------
def register_error_handlers(app, service_name):
    configure_flask_monitoring(app, service_name)  # attach request logging and monitoring hooks

    @app.errorhandler(AssetLoadError)
    def handle_asset_error(error):
        # Missing model or dataset files mean the service is temporarily unavailable.
        app.logger.warning(
            "Runtime asset error.",
            extra={
                "event": "runtime_asset_error",
                "service_name": service_name,
                "path": request.path,
                "detail": str(error),
            },
        )
        return build_error_response(service_name, str(error), 503)

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        message = error.description or "The request could not be completed."  # keep the client message readable
        # Normal Flask errors such as 404 or 400 are logged in the same shape.
        app.logger.warning(
            "HTTP request failed.",
            extra={
                "event": "http_error",
                "service_name": service_name,
                "path": request.path,
                "status_code": error.code or 500,
                "detail": message,
            },
        )
        return build_error_response(service_name, message, error.code or 500)

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        # Keep unexpected errors out of the response body but preserve details in logs.
        app.logger.exception(
            "Unexpected application error.",
            extra={"event": "unexpected_error", "service_name": service_name, "path": request.path},
            exc_info=error,
        )
        return build_error_response(
            service_name,
            "The service ran into an unexpected error while processing the request.",
            500,
        )

    return app  # returning app keeps the helper easy to chain in each Flask app


# -----------------------------
# 6. Build health check response
# -----------------------------
def build_health_response(service_name, asset_loader):
    try:
        asset_loader()  # health should confirm the required model and dataset can load
    except AssetLoadError as error:
        # A failed asset load should make health checks fail clearly.
        LOGGER.warning(
            "Health check failed.",
            extra={"event": "health_check_failed", "service_name": service_name, "detail": str(error)},
        )
        return (
            jsonify(
                {
                    "status": "error",  # service is reachable but not ready
                    "app_name": service_name,
                    "assets_loaded": False,  # Docker/Kubernetes can use this for readiness checks
                    "message": str(error),
                }
            ),
            503,
        )

    LOGGER.info("Health check passed.", extra={"event": "health_check_ok", "service_name": service_name})
    return jsonify({"status": "ok", "app_name": service_name, "assets_loaded": True})  # healthy service response


# -----------------------------
# 7. Ignore safe sklearn version warning
# -----------------------------
def ignore_model_version_warning():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)  # older saved models can warn but still load


# -----------------------------
# 8. Read request payload
# -----------------------------
def read_request_data(request):
    json_payload = request.get_json(silent=True)  # JSON is preferred for API clients

    if isinstance(json_payload, dict):
        return json_payload

    form_payload = request.form.to_dict()  # browser forms reach the same route handlers
    if form_payload:
        return form_payload

    return request.args.to_dict()  # GET requests can still pass query-string inputs


# -----------------------------
# 9. Parse positive integer inputs
# -----------------------------
def read_positive_int(raw_value, default_value):
    try:
        parsed_value = int(raw_value)  # convert query/form text into an integer
    except (TypeError, ValueError):
        return default_value  # use safe fallback when input is missing or invalid

    if parsed_value > 0:
        return parsed_value  # accept only positive values

    return default_value  # zero or negative values fall back to the default


# -----------------------------
# 10. Convert dataframe to API records
# -----------------------------
def dataframe_to_records(dataframe):
    if dataframe.empty:
        return []  # keep JSON responses simple when there is no matching data

    return json.loads(dataframe.to_json(orient="records"))  # pandas values become JSON-safe Python records


# -----------------------------
# 11. Load and cache joblib artifact
# -----------------------------
@lru_cache(maxsize=None)
def load_joblib_file(model_path):
    ignore_model_version_warning()  # avoid noisy sklearn version warnings during model loading
    ensure_runtime_file(model_path, "Model artifact")  # check artifact before joblib tries to load it

    try:
        return joblib.load(model_path)  # model is cached after the first successful load
    except Exception as error:
        raise AssetLoadError(f"Model artifact could not be loaded: {Path(model_path)}") from error


# -----------------------------
# 12. Load and cache CSV data
# -----------------------------
@lru_cache(maxsize=None)
def load_csv_file(csv_path):
    ensure_runtime_file(csv_path, "Dataset file")  # fail early if the dataset is missing or empty

    try:
        return pd.read_csv(csv_path)  # source datasets are static while the Flask app is running
    except Exception as error:
        raise AssetLoadError(f"Dataset file could not be loaded: {Path(csv_path)}") from error
