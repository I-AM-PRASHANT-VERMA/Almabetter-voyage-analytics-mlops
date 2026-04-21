import json

import warnings

from functools import lru_cache

from pathlib import Path

import joblib

import pandas as pd

from sklearn.exceptions import InconsistentVersionWarning


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset" / "travel_capstone"

JOBLIB_DIR = BASE_DIR / "joblib files"


# The saved gender classifier can warn about sklearn version drift while still loading fine.
def ignore_model_version_warning():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def read_request_data(request):
    # The Flask endpoints accept JSON, form posts, and query-string calls so
    # the browser pages and the API routes can share the same handlers.
    json_payload = request.get_json(silent=True)

    if isinstance(json_payload, dict):
        return json_payload

    form_payload = request.form.to_dict()

    if form_payload:
        return form_payload

    return request.args.to_dict()


def read_positive_int(raw_value, default_value):
    # Most API routes use top_n style inputs, so normalize that parsing in one place.
    try:
        parsed_value = int(raw_value)

    except (TypeError, ValueError):
        return default_value

    if parsed_value > 0:
        return parsed_value

    return default_value


# JSON-serializable records keep Flask responses and Jinja tables consistent.
def dataframe_to_records(dataframe):
    if dataframe.empty:
        return []

    return json.loads(dataframe.to_json(orient="records"))


# Cache model loads because the Flask apps reuse the same saved artifacts on every request.
@lru_cache(maxsize=None)
def load_joblib_file(model_path):
    ignore_model_version_warning()

    return joblib.load(model_path)


# Cache CSV loads for the same reason: the source datasets are static while the app is running.
@lru_cache(maxsize=None)
def load_csv_file(csv_path):
    return pd.read_csv(csv_path)
