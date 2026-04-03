import json
import logging
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to the MLops pipeline folder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # allow this nested app to import shared Flask helpers


from flask_apps.common import (
    DATASET_DIR,
    JOBLIB_DIR,
    build_health_response,
    dataframe_to_records,
    display_path,
    load_csv_file,
    load_joblib_file,
    read_request_data,
    register_error_handlers,
)


# -----------------------------
# 1. Create Flask app and artifact paths
# -----------------------------
app = Flask(__name__, template_folder="templates")  # serves both the browser page and API routes
register_error_handlers(app, "Flight Price Flask API")  # attach shared monitoring and error handlers

MODEL_PATH = JOBLIB_DIR / "flight_price_model.joblib"  # promoted model produced by the training pipeline
FLIGHTS_DATA_PATH = DATASET_DIR / "flights.csv"  # source dataset used for dropdowns and route summaries
METADATA_PATH = JOBLIB_DIR / "flight_price_model_metadata.json"  # optional training metadata shown by /model-info

MODEL_FEATURE_COLUMNS = [
    "time",
    "year",
    "month",
    "day",
    "from",
    "to",
    "flightType",
    "agency",
]  # exact raw feature order expected by the saved sklearn pipeline

LOGGER = logging.getLogger("voyage.flight_api")


# -----------------------------
# 2. Build raw model input row
# -----------------------------
def build_prediction_input(departure_city, arrival_city, flight_type, agency, travel_date, travel_time):
    input_row = {
        "time": float(travel_time),  # flight duration
        "year": int(travel_date.year),  # year extracted from selected travel date
        "month": int(travel_date.month),  # month extracted from selected travel date
        "day": int(travel_date.day),  # day extracted from selected travel date
        "from": departure_city,  # raw origin category for the model encoder
        "to": arrival_city,  # raw destination category for the model encoder
        "flightType": flight_type,  # raw cabin type category
        "agency": agency,  # raw agency category
    }
    return pd.DataFrame([input_row], columns=MODEL_FEATURE_COLUMNS)  # keep the trained column order


# -----------------------------
# 3. Read request payload
# -----------------------------
def parse_prediction_payload(payload):
    if all(column in payload for column in MODEL_FEATURE_COLUMNS):
        # Raw API payload already matches the model feature contract.
        travel_date = date(int(payload["year"]), int(payload["month"]), int(payload["day"]))
        return {
            "departure_city": str(payload["from"]).strip(),  # map raw model column to internal name
            "arrival_city": str(payload["to"]).strip(),  # map raw model column to internal name
            "flight_type": str(payload["flightType"]).strip(),  # keep API name compatible with training data
            "agency": str(payload["agency"]).strip(),  # booking agency used as categorical feature
            "travel_date": travel_date,  # rebuilt from year, month, and day
            "travel_time": float(payload["time"]),  # model expects numeric duration
        }

    travel_date_text = str(payload.get("travel_date", "")).strip()  # legacy browser-style payload
    return {
        "departure_city": str(payload.get("departure_city", "")).strip(),  # browser field name
        "arrival_city": str(payload.get("arrival_city", "")).strip(),  # browser field name
        "flight_type": str(payload.get("flight_type", "")).strip(),  # browser field name
        "agency": str(payload.get("agency", "")).strip(),  # browser field name
        "travel_date": pd.to_datetime(travel_date_text).date(),  # convert form date text into a date object
        "travel_time": float(payload.get("travel_time")),  # convert form input into numeric duration
    }


# -----------------------------
# 4. Validate route and input fields
# -----------------------------
def validate_city(city_name, city_options, label):
    if city_name not in city_options:
        valid_cities = ", ".join(city_options)
        raise ValueError(f"{label} must be one of: {valid_cities}")  # reject unknown city names before prediction


def validate_flight_inputs(assets, departure_city, arrival_city, flight_type, agency, travel_time=None):
    validate_city(departure_city, assets["city_options"], "departure_city")  # origin must exist in training data
    validate_city(arrival_city, assets["city_options"], "arrival_city")  # destination must exist in training data

    if departure_city == arrival_city:
        raise ValueError("departure_city and arrival_city must be different.")  # same-city route is not useful

    if flight_type not in assets["flight_type_options"]:
        valid_types = ", ".join(assets["flight_type_options"])  # show allowed cabin types in the error
        raise ValueError(f"flight_type must be one of: {valid_types}")

    if agency not in assets["agency_options"]:
        valid_agencies = ", ".join(assets["agency_options"])  # show allowed agencies in the error
        raise ValueError(f"agency must be one of: {valid_agencies}")

    if travel_time is not None and travel_time <= 0:
        raise ValueError("travel_time must be greater than 0.")  # duration cannot be zero or negative


# -----------------------------
# 5. Build route summary table
# -----------------------------
def build_route_summary(flights_df):
    route_summary_df = (
        flights_df.groupby(["from", "to", "flightType"], as_index=False)  # one summary row per route and cabin type
        .agg(
            avg_price=("price", "mean"),  # historical average ticket price
            avg_time=("time", "mean"),  # historical average flight time
            avg_distance=("distance", "mean"),  # historical average route distance
            trip_count=("travelCode", "count"),  # number of records behind the summary
        )
    )
    route_summary_df["avg_price"] = route_summary_df["avg_price"].round(2)  # keep UI/API numbers readable
    route_summary_df["avg_time"] = route_summary_df["avg_time"].round(2)  # keep UI/API numbers readable
    route_summary_df["avg_distance"] = route_summary_df["avg_distance"].round(2)  # keep UI/API numbers readable
    return route_summary_df  # used by both the browser page and /route-summary endpoint


# -----------------------------
# 6. Load optional training metadata
# -----------------------------
def load_model_metadata():
    if not METADATA_PATH.exists():
        return {}  # API can still run when metadata has not been generated yet

    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))  # expose training details when metadata exists


# -----------------------------
# 7. Load and cache runtime assets
# -----------------------------
@lru_cache(maxsize=1)
def load_flight_assets():
    model = load_joblib_file(MODEL_PATH)  # cached model artifact
    flights_df = load_csv_file(FLIGHTS_DATA_PATH)  # cached dataset used for options and summaries
    route_summary_df = build_route_summary(flights_df)  # precompute route context once

    city_options = sorted(flights_df["from"].dropna().unique().tolist())  # dropdown source for origin/destination
    flight_type_options = sorted(flights_df["flightType"].dropna().unique().tolist())  # dropdown cabin types
    agency_options = sorted(flights_df["agency"].dropna().unique().tolist())  # dropdown agencies
    dataset_min_date = str(pd.to_datetime(flights_df["date"]).min().date())  # lower date limit for UI
    dataset_max_date = str(pd.to_datetime(flights_df["date"]).max().date())  # upper date limit for UI
    average_price = round(float(flights_df["price"].mean()), 2)  # baseline comparison shown in UI/API

    return {
        "model": model,  # trained sklearn pipeline
        "flights_df": flights_df,  # full dataset for UI stats
        "route_summary_df": route_summary_df,  # grouped route-level context
        "city_options": city_options,  # dropdown values
        "flight_type_options": flight_type_options,  # dropdown values
        "agency_options": agency_options,  # dropdown values
        "dataset_min_date": dataset_min_date,  # UI date minimum
        "dataset_max_date": dataset_max_date,  # UI date maximum
        "average_price": average_price,  # overall dataset average
    }


# -----------------------------
# 8. Render browser prediction page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    assets = load_flight_assets()  # reuse cached model, dataset, and dropdown values

    default_departure_city = assets["city_options"][0] if assets["city_options"] else ""  # first city for initial page load
    default_arrival_city = assets["city_options"][1] if len(assets["city_options"]) > 1 else default_departure_city  # second city avoids same-route default
    default_flight_type = assets["flight_type_options"][0] if assets["flight_type_options"] else ""  # first known cabin type
    default_agency = assets["agency_options"][0] if assets["agency_options"] else ""  # first known agency

    departure_city = str(request.values.get("departure_city", default_departure_city)).strip()  # form value or default
    arrival_city = str(request.values.get("arrival_city", default_arrival_city)).strip()  # form value or default
    flight_type = str(request.values.get("flight_type", default_flight_type)).strip()  # form value or default
    agency = str(request.values.get("agency", default_agency)).strip()  # form value or default
    travel_date_text = str(request.values.get("travel_date", assets["dataset_min_date"])).strip()  # date field text

    default_travel_time = round(float(assets["flights_df"]["time"].median()), 2)  # realistic default duration from data
    travel_time_text = request.values.get("travel_time", default_travel_time)  # form value or median fallback

    if departure_city not in assets["city_options"]:
        departure_city = default_departure_city  # avoid rendering invalid dropdown state
    if arrival_city not in assets["city_options"]:
        arrival_city = default_arrival_city  # avoid rendering invalid dropdown state
    if flight_type not in assets["flight_type_options"]:
        flight_type = default_flight_type  # avoid rendering invalid dropdown state
    if agency not in assets["agency_options"]:
        agency = default_agency  # avoid rendering invalid dropdown state

    try:
        travel_time = float(travel_time_text)  # browser input arrives as text
    except (TypeError, ValueError):
        travel_time = default_travel_time  # keep page usable when input is invalid

    selected_route_df = assets["route_summary_df"][
        (assets["route_summary_df"]["from"] == departure_city)
        & (assets["route_summary_df"]["to"] == arrival_city)
        & (assets["route_summary_df"]["flightType"] == flight_type)
    ]  # matching historical summary for the selected route

    predicted_price = None  # page starts without a prediction on GET
    page_error = ""  # populated only when browser prediction fails

    if request.method == "POST":
        try:
            travel_date = pd.to_datetime(travel_date_text).date()  # convert submitted date before feature building
            prediction_input_df = build_prediction_input(
                departure_city=departure_city,
                arrival_city=arrival_city,
                flight_type=flight_type,
                agency=agency,
                travel_date=travel_date,
                travel_time=travel_time,
            )
            predicted_price = round(float(assets["model"].predict(prediction_input_df)[0]), 2)  # run model prediction
            travel_date_text = str(travel_date)  # normalize date text after parsing
        except Exception:
            LOGGER.warning(
                "Browser prediction failed.",
                extra={
                    "event": "browser_prediction_failed",
                    "service_name": "Flight Price Flask API",
                    "departure_city": departure_city,
                    "arrival_city": arrival_city,
                    "flight_type": flight_type,
                },
            )
            page_error = "The app could not create a prediction from the submitted values. Please check the date and travel time fields."  # browser-safe error

    return render_template(
        "index.html",  # browser template for the Flask UI
        app_name="Flight Price Prediction",
        model_file=display_path(MODEL_PATH),  # show which promoted model file is used
        dataset_file=display_path(FLIGHTS_DATA_PATH),  # show which dataset feeds the UI options
        record_count=int(assets["flights_df"].shape[0]),  # dataset size shown in the page summary
        route_count=int(assets["route_summary_df"].shape[0]),  # number of grouped route summaries
        average_price=assets["average_price"],  # overall price benchmark
        city_options=assets["city_options"],  # dropdown options
        flight_type_options=assets["flight_type_options"],  # dropdown options
        agency_options=assets["agency_options"],  # dropdown options
        dataset_min_date=assets["dataset_min_date"],  # date input lower limit
        dataset_max_date=assets["dataset_max_date"],  # date input upper limit
        departure_city=departure_city,  # selected form value
        arrival_city=arrival_city,  # selected form value
        flight_type=flight_type,  # selected form value
        agency=agency,  # selected form value
        travel_date=travel_date_text,  # selected form value
        travel_time=travel_time,  # selected form value
        predicted_price=predicted_price,  # model output shown after POST
        page_error=page_error,  # browser error message if prediction fails
        route_summary=dataframe_to_records(selected_route_df),  # route context table
    )


# -----------------------------
# 9. Show API overview
# -----------------------------
@app.get("/api")
def api_overview():
    assets = load_flight_assets()  # include live dataset counts in the overview
    return jsonify(
        {
            "app_name": "Flight Price Flask API",  # service name for manual API checks
            "model_file": display_path(MODEL_PATH),  # promoted model currently served
            "dataset_file": display_path(FLIGHTS_DATA_PATH),  # dataset used for options/context
            "available_routes": {  # quick route reference for testers
                "GET /": "Open the browser-based Flask page.",
                "GET /api": "Open the JSON overview for this app.",
                "GET /health": "Check whether the API is running.",
                "GET /model-info": "Return metadata for the saved regression model when available.",
                "GET /metadata": "Return the dropdown options and dataset date range.",
                "GET or POST /route-summary": "Return the historical route summary for one route and cabin type.",
                "GET or POST /predict": "Return the predicted flight price for one request payload.",
            },
            "sample_payload": {  # raw payload format accepted by /predict
                "time": 1.76,
                "year": 2019,
                "month": 9,
                "day": 26,
                "from": "Recife (PE)",
                "to": "Florianopolis (SC)",
                "flightType": "firstClass",
                "agency": "FlyingDrops",
            },
            "summary": {  # small dataset summary for API users
                "record_count": int(assets["flights_df"].shape[0]),
                "route_count": int(assets["route_summary_df"].shape[0]),
                "average_price": assets["average_price"],
            },
        }
    )


# -----------------------------
# 10. Health check endpoint
# -----------------------------
@app.get("/health")
def health():
    return build_health_response("Flight Price Flask API", load_flight_assets)  # used by Docker/Kubernetes probes


# -----------------------------
# 11. Model metadata endpoint
# -----------------------------
@app.get("/model-info")
def model_info():
    model_metadata = load_model_metadata()  # read metadata created by training/promotion flow
    if not model_metadata:
        return jsonify(
            {
                "message": "Model metadata is not available yet. Run the local training pipeline first.",
                "metadata_file": display_path(METADATA_PATH),  # expected metadata location
            }
        )

    return jsonify(model_metadata)  # expose selected model, metrics, and registry details when available


# -----------------------------
# 12. UI metadata endpoint
# -----------------------------
@app.get("/metadata")
def metadata():
    assets = load_flight_assets()  # dropdown options come from the loaded dataset
    return jsonify(
        {
            "city_options": assets["city_options"],  # valid origin and destination values
            "flight_type_options": assets["flight_type_options"],  # valid cabin type values
            "agency_options": assets["agency_options"],  # valid agency values
            "dataset_min_date": assets["dataset_min_date"],  # date lower bound for clients
            "dataset_max_date": assets["dataset_max_date"],  # date upper bound for clients
        }
    )


# -----------------------------
# 13. Route summary endpoint
# -----------------------------
@app.route("/route-summary", methods=["GET", "POST"])
def route_summary():
    assets = load_flight_assets()  # load cached route summary and valid option lists
    payload = read_request_data(request)  # accept JSON, form data, or query params

    departure_city = str(payload.get("departure_city", "")).strip()  # required route input
    arrival_city = str(payload.get("arrival_city", "")).strip()  # required route input
    flight_type = str(payload.get("flight_type", "")).strip()  # required cabin type input

    if not departure_city or not arrival_city or not flight_type:
        return jsonify({"error": "Please provide departure_city, arrival_city, and flight_type."}), 400  # missing request fields

    try:
        validate_city(departure_city, assets["city_options"], "departure_city")  # origin must be known
        validate_city(arrival_city, assets["city_options"], "arrival_city")  # destination must be known
        if departure_city == arrival_city:
            raise ValueError("departure_city and arrival_city must be different.")
        if flight_type not in assets["flight_type_options"]:
            valid_types = ", ".join(assets["flight_type_options"])  # show valid cabin types in error
            raise ValueError(f"flight_type must be one of: {valid_types}")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400  # validation errors are client-side issues

    selected_route_df = assets["route_summary_df"][
        (assets["route_summary_df"]["from"] == departure_city)
        & (assets["route_summary_df"]["to"] == arrival_city)
        & (assets["route_summary_df"]["flightType"] == flight_type)
    ]  # filter the precomputed route summary table

    if selected_route_df.empty:
        return jsonify({"error": "No historical summary was found for the selected route and flight type."}), 404  # route not found in history

    return jsonify({"route_summary": dataframe_to_records(selected_route_df)})  # return historical context records


# -----------------------------
# 14. Prediction endpoint
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    assets = load_flight_assets()  # model and dataset context are cached after first load
    payload = read_request_data(request)  # supports JSON, form, and query-string inputs

    try:
        parsed_payload = parse_prediction_payload(payload)  # normalize raw API or browser-style input
    except Exception:
        return jsonify({"error": "Please provide raw fields: time, year, month, day, from, to, flightType, and agency."}), 400  # malformed payload

    departure_city = parsed_payload["departure_city"]  # normalized origin
    arrival_city = parsed_payload["arrival_city"]  # normalized destination
    flight_type = parsed_payload["flight_type"]  # normalized cabin type
    agency = parsed_payload["agency"]  # normalized agency
    travel_date = parsed_payload["travel_date"]  # normalized date object
    travel_time = parsed_payload["travel_time"]  # normalized numeric duration

    if not all([departure_city, arrival_city, flight_type, agency]):
        return jsonify({"error": "Please provide from, to, flightType, and agency."}), 400  # required categorical fields

    try:
        validate_flight_inputs(
            assets=assets,
            departure_city=departure_city,
            arrival_city=arrival_city,
            flight_type=flight_type,
            agency=agency,
            travel_time=travel_time,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400  # return validation message directly to client

    prediction_input_df = build_prediction_input(
        departure_city=departure_city,
        arrival_city=arrival_city,
        flight_type=flight_type,
        agency=agency,
        travel_date=travel_date,
        travel_time=travel_time,
    )  # one-row DataFrame in the trained model's expected column order
    predicted_price = float(assets["model"].predict(prediction_input_df)[0])  # generate prediction from saved model

    LOGGER.info(
        "Prediction generated.",
        extra={
            "event": "flight_prediction_generated",
            "service_name": "Flight Price Flask API",
            "departure_city": departure_city,
            "arrival_city": arrival_city,
            "flight_type": flight_type,
            "agency": agency,
        },
    )  # log useful request context without logging the full payload

    selected_route_df = assets["route_summary_df"][
        (assets["route_summary_df"]["from"] == departure_city)
        & (assets["route_summary_df"]["to"] == arrival_city)
        & (assets["route_summary_df"]["flightType"] == flight_type)
    ]  # gives prediction context from historical route data

    return jsonify(
        {
            "request": {  # echo normalized request values for transparency
                "departure_city": departure_city,
                "arrival_city": arrival_city,
                "flight_type": flight_type,
                "agency": agency,
                "travel_date": str(travel_date),
                "travel_time": travel_time,
            },
            "predicted_price": round(predicted_price, 2),  # final model output
            "overall_average_price": assets["average_price"],  # dataset-level comparison value
            "route_summary": dataframe_to_records(selected_route_df),  # historical context for the same route
            "model_input_preview": dataframe_to_records(prediction_input_df),  # shows exact model input row
        }
    )


# -----------------------------
# 15. Local development entry point
# -----------------------------
if __name__ == "__main__":
    LOGGER.info("Starting Flask API server.", extra={"event": "service_start", "service_name": "Flight Price Flask API"})
    app.run(host="0.0.0.0", port=5002, debug=False)  # local development server
