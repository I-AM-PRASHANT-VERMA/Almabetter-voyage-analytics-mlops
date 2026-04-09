import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import g, request

# Azure Monitor is optional locally, but enabled in AKS when the package and connection string exist.
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
except ImportError:  # pragma: no cover - local fallback when extras are not installed yet
    configure_azure_monitor = None


# Keep standard LogRecord fields out of the custom JSON payload loop.
_STANDARD_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}

_MONITORING_READY = False

# Runtime logs are written under the project folder unless an env var overrides it.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = BASE_DIR / "runtime_logs"


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        # -----------------------------
        # 1. Format one log record as JSON
        # -----------------------------
        # Convert standard log records into structured JSON for local files and Azure export.
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "service": os.getenv("OTEL_SERVICE_NAME", "voyage-service"),
            "environment": os.getenv("APP_ENVIRONMENT", "local"),
        }

        # Add only custom fields passed through logger.extra.
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class AlertLevelFilter(logging.Filter):
    def filter(self, record):
        # -----------------------------
        # 2. Keep only alert-level logs
        # -----------------------------
        # Only warning and error records should enter the alert log file.
        return record.levelno >= logging.WARNING


def _resolve_log_level():
    # -----------------------------
    # 3. Read logging level
    # -----------------------------
    # Allow APP_LOG_LEVEL to tune console and file verbosity without code changes.
    log_level = os.getenv("APP_LOG_LEVEL", "INFO").upper()

    return getattr(logging, log_level, logging.INFO)


def _resolve_log_dir():
    # -----------------------------
    # 4. Resolve local log folder
    # -----------------------------
    # LOCAL_LOG_DIR is useful when logs need to be redirected outside the repo.
    configured_path = os.getenv("LOCAL_LOG_DIR", "").strip()

    if configured_path:
        return Path(configured_path)

    return DEFAULT_LOG_DIR


def _build_file_handler(log_path):
    # -----------------------------
    # 5. Build rotating JSON file handler
    # -----------------------------
    # Rotate files so long-running local services do not grow one infinite log file.
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonLogFormatter())

    return file_handler


def _build_alert_handler(log_path):
    # -----------------------------
    # 6. Build alert-only file handler
    # -----------------------------
    # Reuse the JSON rotating handler and then narrow it to warning/error events.
    alert_handler = _build_file_handler(log_path)
    alert_handler.addFilter(AlertLevelFilter())

    return alert_handler


def configure_service_monitoring(service_name):
    # -----------------------------
    # 7. Configure service-level monitoring
    # -----------------------------
    global _MONITORING_READY

    # Avoid re-registering handlers when Flask reloads modules.
    if _MONITORING_READY:
        return

    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
    os.environ.setdefault("APP_ENVIRONMENT", "local")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(_resolve_log_level())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonLogFormatter())
    root_logger.addHandler(console_handler)

    log_dir = _resolve_log_dir()
    # Keep one shared runtime log, one per-service log, and one alert-only log.
    root_logger.addHandler(_build_file_handler(log_dir / "voyage-runtime.jsonl"))
    root_logger.addHandler(_build_file_handler(log_dir / f"{service_name}.jsonl"))
    root_logger.addHandler(_build_alert_handler(log_dir / "voyage-alerts.jsonl"))

    logging.getLogger("werkzeug").setLevel(logging.INFO)
    logging.captureWarnings(True)

    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()

    # If Azure Application Insights is configured, export the same telemetry there too.
    if connection_string and configure_azure_monitor is not None:
        configure_azure_monitor(connection_string=connection_string)
        root_logger.info(
            "Azure Monitor exporter enabled.",
            extra={"event": "monitoring_initialized", "service_name": service_name},
        )
    elif connection_string and configure_azure_monitor is None:
        root_logger.warning(
            "Azure Monitor package is missing, so telemetry export is off.",
            extra={"event": "monitoring_export_disabled", "service_name": service_name},
        )
    else:
        root_logger.info(
            "Structured logging is running without Azure Monitor export.",
            extra={
                "event": "monitoring_local_mode",
                "service_name": service_name,
                "log_dir": str(log_dir),
            },
        )

    _MONITORING_READY = True


def configure_flask_monitoring(app, service_name):
    # -----------------------------
    # 8. Add Flask request monitoring
    # -----------------------------
    # Apply shared logging first, then attach request timing hooks to this Flask app.
    configure_service_monitoring(service_name)

    @app.before_request
    def start_request_timer():
        # -----------------------------
        # 9. Start request timer
        # -----------------------------
        # Track when the request started and assign a request id for tracing.
        g.request_started_at = time.perf_counter()
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    @app.after_request
    def log_request_summary(response):
        # -----------------------------
        # 10. Log completed request
        # -----------------------------
        # Emit one structured summary log after each request finishes.
        started_at = getattr(g, "request_started_at", None)
        duration_ms = None

        if started_at is not None:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

        log_payload = {
            "event": "request_completed",
            "service_name": service_name,
            "request_id": getattr(g, "request_id", ""),
            "method": request.method,
            "path": request.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }

        # Escalate log level based on the response code.
        if response.status_code >= 500:
            app.logger.error("Request completed with a server error.", extra=log_payload)
        elif response.status_code >= 400:
            app.logger.warning("Request completed with a client error.", extra=log_payload)
        else:
            app.logger.info("Request completed.", extra=log_payload)

        response.headers["X-Request-ID"] = getattr(g, "request_id", "")

        return response

    return app
