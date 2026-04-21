import sys

from pathlib import Path


# Make sure the project root is importable when this script is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Reuse the shared ngrok bootstrapper and the existing Flask app object.
from ngrok_apps.common import start_ngrok_for_app

from flask_apps.flight_price_flask_app.app import app


# Convenience entry point for exposing the local flight API.
if __name__ == "__main__":
    start_ngrok_for_app(app, 5002, "Flight Price Flask API")
