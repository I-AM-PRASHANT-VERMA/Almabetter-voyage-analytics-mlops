import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


# This script is the single official local training entry point for flight price regression.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("FLIGHT_DATA_DIR", PROJECT_ROOT / "dataset" / "travel_capstone"))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
LOG_DIR = OUTPUT_DIR / "logs"
MLRUNS_DIR = OUTPUT_DIR / "mlruns"

TARGET_COLUMN = "price"
RAW_COLUMNS = ["date", "time", "from", "to", "flightType", "agency"]
MODEL_COLUMNS = ["time", "year", "month", "day", "from", "to", "flightType", "agency"]
NUMERIC_COLUMNS = ["time", "year", "month", "day"]
CATEGORICAL_COLUMNS = ["from", "to", "flightType", "agency"]
PRICE_GROUP_COLUMNS = ["from", "to", "flightType", "agency"]

RANDOM_STATE = 42
TEST_RATIO = 0.20
CV_SPLITS = 3
# Keep tuning intentionally light so the script stays practical on a normal local machine.
TUNING_ITERATIONS = int(os.getenv("TUNING_ITERATIONS", "5"))
TRAIN_SAMPLE_ROWS = int(os.getenv("TRAIN_SAMPLE_ROWS", "0"))


def prepare_folders():
    # -----------------------------
    # 1. Prepare local training workspace
    # -----------------------------
    for folder in [MODEL_DIR, METRICS_DIR, LOG_DIR, MLRUNS_DIR]:  # keep all training outputs in one place
        folder.mkdir(parents=True, exist_ok=True)  # create missing folders without touching existing files


def start_log_file():
    # -----------------------------
    # 2. Start a simple run log
    # -----------------------------
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # use one timestamp for model, metrics, and logs
    log_file = LOG_DIR / f"flight_price_training_{run_stamp}.log"  # keep terminal messages in a readable file
    return run_stamp, log_file


def log_message(message, log_file):
    print(message)  # show progress in the terminal while the script runs
    with log_file.open("a", encoding="utf-8") as file:  # open the current run log in append mode
        file.write(message + "\n")  # save the same message for later checking


def load_flight_data(log_file):
    # -----------------------------
    # 3. Load the local flight dataset
    # -----------------------------
    flight_file = DATA_DIR / "flights.csv"  # local dataset path used by this project

    if not flight_file.exists():  # stop early when the dataset path is wrong
        raise FileNotFoundError(f"Flight dataset not found: {flight_file}")

    flights = pd.read_csv(flight_file)  # load the flight records used for price prediction
    log_message(f"Loaded flights.csv with shape: {flights.shape}", log_file)  # record row and column count
    return flights


def build_modeling_dataset(flights, log_file):
    # -----------------------------
    # 4. Build the modeling dataset
    # -----------------------------
    flight_source = flights.copy()  # keep raw data unchanged while preparing model data
    flight_source["date"] = pd.to_datetime(flight_source["date"])  # convert date text into real datetime values

    model_df = flight_source[RAW_COLUMNS + [TARGET_COLUMN]].copy()  # keep only columns needed for training
    model_df["year"] = model_df["date"].dt.year  # use year as a numeric time feature
    model_df["month"] = model_df["date"].dt.month  # use month to capture seasonal price movement
    model_df["day"] = model_df["date"].dt.day  # use day to keep basic date-level variation

    rows_before = len(model_df)  # count rows before ML-specific duplicate cleanup
    model_df = model_df.drop_duplicates(subset=MODEL_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)  # remove repeated same-input same-price rows
    rows_after = len(model_df)  # count rows after duplicate cleanup

    model_df = model_df.sort_values("date").reset_index(drop=True)  # place older records before newer records

    if TRAIN_SAMPLE_ROWS > 0:  # use this only for a quick test run on a smaller sample
        model_df = model_df.tail(TRAIN_SAMPLE_ROWS).reset_index(drop=True)  # keep latest rows so time order still makes sense
        log_message(f"Using sample rows for quick test: {len(model_df)}", log_file)  # show that this is not a full run

    split_index = int(len(model_df) * (1 - TEST_RATIO))  # calculate the 80 percent time-based split point
    train_df = model_df.iloc[:split_index].copy()  # older rows become training data
    test_df = model_df.iloc[split_index:].copy()  # newer rows become holdout test data

    X_train = train_df[MODEL_COLUMNS]  # training input columns
    y_train = train_df[TARGET_COLUMN]  # training target price
    X_test = test_df[MODEL_COLUMNS]  # holdout input columns
    y_test = test_df[TARGET_COLUMN]  # holdout actual price

    train_keys = set(map(tuple, X_train.astype(str).values))  # convert train rows into comparable keys
    test_keys = list(map(tuple, X_test.astype(str).values))  # convert test rows into comparable keys
    overlap_count = sum(key in train_keys for key in test_keys)  # count exact test inputs already seen in train
    overlap_rate = overlap_count / max(len(test_keys), 1)  # turn overlap count into a safe percentage

    train_price_groups = set(map(tuple, train_df[PRICE_GROUP_COLUMNS].astype(str).values))  # keep route-agency-price pattern keys from train
    test_price_groups = list(map(tuple, test_df[PRICE_GROUP_COLUMNS].astype(str).values))  # keep the same pattern keys from test
    price_group_overlap = sum(key in train_price_groups for key in test_price_groups) / max(len(test_price_groups), 1)  # check repeated pricing patterns

    dataset_info = {  # keep dataset details for MLflow and metadata
        "rows_before_duplicate_cleanup": int(rows_before),
        "rows_after_duplicate_cleanup": int(rows_after),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_overlap_rate": float(overlap_rate),
        "price_group_overlap_rate": float(price_group_overlap),
        "train_start_date": str(train_df["date"].min()),
        "train_end_date": str(train_df["date"].max()),
        "test_start_date": str(test_df["date"].min()),
        "test_end_date": str(test_df["date"].max()),
    }

    for key, value in dataset_info.items():  # print every important split detail once
        log_message(f"{key}: {value}", log_file)  # save the split detail in terminal and log file

    return model_df, X_train, X_test, y_train, y_test, dataset_info


def build_group_holdout_dataset(model_df, log_file):
    # -----------------------------
    # 5. Build stricter group holdout data
    # -----------------------------
    group_labels = model_df[PRICE_GROUP_COLUMNS].astype(str).agg("|".join, axis=1)  # create one label for each route-agency-flight pattern
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE)  # hold out complete pricing groups
    train_index, test_index = next(splitter.split(model_df, groups=group_labels))  # split rows without sharing pricing groups

    group_train_df = model_df.iloc[train_index].sort_values("date").copy()  # keep train groups in time order for readability
    group_test_df = model_df.iloc[test_index].sort_values("date").copy()  # keep unseen test groups in time order

    X_group_train = group_train_df[MODEL_COLUMNS]  # group-holdout training inputs
    y_group_train = group_train_df[TARGET_COLUMN]  # group-holdout training target
    X_group_test = group_test_df[MODEL_COLUMNS]  # unseen-group test inputs
    y_group_test = group_test_df[TARGET_COLUMN]  # unseen-group actual prices

    train_groups = set(group_train_df[PRICE_GROUP_COLUMNS].astype(str).agg("|".join, axis=1))  # groups seen during group training
    test_groups = set(group_test_df[PRICE_GROUP_COLUMNS].astype(str).agg("|".join, axis=1))  # groups reserved for strict testing
    group_overlap = len(train_groups.intersection(test_groups)) / max(len(test_groups), 1)  # this should be zero

    group_info = {  # keep strict validation details for MLflow and metadata
        "group_holdout_train_rows": int(len(X_group_train)),
        "group_holdout_test_rows": int(len(X_group_test)),
        "group_holdout_train_groups": int(len(train_groups)),
        "group_holdout_test_groups": int(len(test_groups)),
        "group_holdout_overlap_rate": float(group_overlap),
    }

    for key, value in group_info.items():  # print the strict split details once
        log_message(f"{key}: {value}", log_file)  # save group split details in the log

    return X_group_train, X_group_test, y_group_train, y_group_test, group_info


def build_preprocessor():
    # -----------------------------
    # 6. Build preprocessing steps
    # -----------------------------
    return ColumnTransformer(  # apply separate transformations to numeric and categorical columns
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),  # scale numeric columns before linear/tree models use them
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),  # encode text columns safely
        ]
    )


def make_pipeline(model):
    # A single saved pipeline is easier to deploy than separate preprocess + model objects.
    return Pipeline(  # save preprocessing and model together in one object
        steps=[
            ("preprocess", build_preprocessor()),  # convert raw input columns into model-ready columns
            ("model", model),  # train the selected algorithm after preprocessing
        ]
    )


def regression_metrics(y_true, y_pred):
    # -----------------------------
    # 7. Calculate regression metrics
    # -----------------------------
    mse_value = mean_squared_error(y_true, y_pred)  # calculate squared prediction error

    return {  # return plain Python floats so MLflow and JSON can store them cleanly
        "mae": float(mean_absolute_error(y_true, y_pred)),  # average absolute price error
        "mse": float(mse_value),  # average squared error
        "rmse": float(np.sqrt(mse_value)),  # error in original price scale
        "r2_score": float(r2_score(y_true, y_pred)),  # how much price variation the model explains
    }


def is_suspicious(metrics):
    # Extremely good scores are flagged because they can hide leakage or memorization.
    return bool(metrics["rmse"] < 5 or metrics["r2_score"] > 0.9995)  # flag scores that are too close to lookup-table behavior


def model_search_space():
    # -----------------------------
    # 8. Define candidate models and light tuning ranges
    # -----------------------------
    spaces = {  # keep each model with its small tuning space
        "Ridge Regression": {
            "estimator": Ridge(),  # regularized linear baseline
            "params": {
                "model__alpha": [0.1, 1.0, 5.0, 10.0, 25.0],  # try a few regularization strengths
            },
        },
        "Decision Tree Regularized": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),  # single tree with controlled depth
            "params": {
                "model__max_depth": [8, 10, 12, 14, 16],  # limit how deep the tree can grow
                "model__min_samples_leaf": [5, 10, 20],  # keep leaves large enough to avoid memorizing rows
                "model__min_samples_split": [10, 20, 40],  # require enough rows before a split happens
            },
        },
        "Random Forest Regularized": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),  # ensemble of controlled trees
            "params": {
                "model__n_estimators": [80, 120],  # try a practical tree count for local training
                "model__max_depth": [12, 14, 16],  # control tree depth to reduce overfitting
                "model__min_samples_leaf": [3, 5, 8],  # smooth predictions by keeping more rows per leaf
                "model__min_samples_split": [5, 10, 20],  # avoid very tiny splits
            },
        },
        "Hist Gradient Boosting": {
            "estimator": HistGradientBoostingRegressor(random_state=RANDOM_STATE),  # fast sklearn boosting model
            "params": {
                "model__max_iter": [120, 180, 220],  # try a few boosting round counts
                "model__learning_rate": [0.05, 0.08, 0.1],  # control how strongly each boosting step learns
                "model__max_leaf_nodes": [31, 63],  # control tree complexity inside boosting
                "model__l2_regularization": [0.05, 0.1, 0.2],  # add regularization to reduce overfitting
            },
        },
    }

    if XGBRegressor is not None:  # add XGBoost only when the package is available
        spaces["XGBoost Regularized"] = {
            "estimator": XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),  # boosted tree model for regression
            "params": {
                "model__n_estimators": [150, 250],  # try moderate tree counts
                "model__learning_rate": [0.03, 0.05, 0.08],  # try controlled learning rates
                "model__max_depth": [4, 5, 6],  # keep trees shallow enough for generalization
                "model__subsample": [0.8, 0.9],  # train trees on partial row samples
                "model__colsample_bytree": [0.8, 0.9],  # train trees on partial feature samples
                "model__reg_alpha": [0.0, 0.1],  # try light L1 regularization
                "model__reg_lambda": [1.0, 2.0],  # try light L2 regularization
            },
        }

    return spaces


def tune_and_evaluate_model(
    model_name,
    setup,
    X_train,
    X_test,
    y_train,
    y_test,
    X_group_train,
    X_group_test,
    y_group_train,
    y_group_test,
    dataset_info,
    group_info,
    log_file,
):
    # -----------------------------
    # 9. Train, tune, evaluate, and track one model
    # -----------------------------
    # Train on the time split first because that is closest to real future prediction usage.
    pipeline = make_pipeline(setup["estimator"])  # combine preprocessing and the candidate model
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)  # validate in date order instead of random order

    search = RandomizedSearchCV(  # run light hyperparameter tuning
        estimator=pipeline,  # tune the complete preprocessing-plus-model pipeline
        param_distributions=setup["params"],  # use the small parameter ranges defined above
        n_iter=TUNING_ITERATIONS,  # keep tuning small for local machine safety
        scoring="neg_root_mean_squared_error",  # choose the model with the lowest validation RMSE
        cv=cv,  # use time-aware validation splits
        random_state=RANDOM_STATE,  # keep search reproducible
        n_jobs=-1,  # use available CPU cores
        refit=True,  # retrain the best setting on the full training data
    )

    run_name = model_name.lower().replace(" ", "_")  # readable MLflow run name

    with mlflow.start_run(run_name=run_name):  # track this candidate as one MLflow run
        start_time = time.time()  # measure how long this model takes
        log_message(f"Training started: {model_name}", log_file)  # show progress before fitting

        search.fit(X_train, y_train)  # train multiple tuned versions using time-based folds
        best_model = search.best_estimator_  # keep the best tuned pipeline
        time_predictions = best_model.predict(X_test)  # test the tuned model on future-style holdout data
        time_metrics = regression_metrics(y_test, time_predictions)  # calculate time-split holdout metrics

        # Refit the same tuned model on unseen pricing groups to get a stricter quality check.
        group_model = make_pipeline(clone(search.best_estimator_.named_steps["model"]))  # rebuild the best model for unseen-group validation
        group_model.fit(X_group_train, y_group_train)  # train on groups that will not appear in the strict test set
        group_predictions = group_model.predict(X_group_test)  # predict prices for unseen route-agency-flight groups
        group_metrics = regression_metrics(y_group_test, group_predictions)  # calculate stricter unseen-group metrics

        cv_rmse = -cross_val_score(  # re-check the selected model across time-based folds
            best_model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        metrics = {  # keep both easy and strict validation scores in one record
            "time_mae": time_metrics["mae"],
            "time_mse": time_metrics["mse"],
            "time_rmse": time_metrics["rmse"],
            "time_r2_score": time_metrics["r2_score"],
            "group_mae": group_metrics["mae"],
            "group_mse": group_metrics["mse"],
            "group_rmse": group_metrics["rmse"],
            "group_r2_score": group_metrics["r2_score"],
        }
        metrics["cv_rmse_mean"] = float(cv_rmse.mean())  # average cross-validation RMSE
        metrics["cv_rmse_std"] = float(cv_rmse.std())  # stability of RMSE across folds
        metrics["suspicious_score"] = is_suspicious(time_metrics)  # mark easy split scores that look too perfect
        metrics["selection_metric"] = metrics["group_rmse"]  # use stricter unseen-group RMSE for ranking
        metrics["training_seconds"] = round(time.time() - start_time, 2)  # store runtime for this model

        # Store enough context in MLflow so later comparison does not depend on development workflow memory.
        mlflow.log_params(dataset_info)  # save dataset and split details
        mlflow.log_params(group_info)  # save strict group-holdout split details
        mlflow.log_params({"model_name": model_name})  # save model name
        mlflow.log_params({key.replace("model__", ""): value for key, value in search.best_params_.items()})  # save best tuned settings
        mlflow.log_metrics(metrics)  # save model metrics

        log_message(  # print the main result in one readable line
            f"Finished {model_name}: time_RMSE={metrics['time_rmse']:.4f}, group_RMSE={metrics['group_rmse']:.4f}",
            log_file,
        )

    return {  # return one record for comparison and saving
        "model_name": model_name,
        "model": best_model,
        "best_params": search.best_params_,
        **metrics,
    }


def save_results(results, dataset_info, group_info, run_stamp, X_test, y_test, log_file):
    # -----------------------------
    # 10. Save model comparison and best model
    # -----------------------------
    comparison_rows = [{key: value for key, value in row.items() if key != "model"} for row in results]  # remove model objects from CSV data
    comparison_df = pd.DataFrame(comparison_rows).sort_values(["selection_metric", "time_rmse"], ascending=[True, True]).reset_index(drop=True)  # rank by strict unseen-group RMSE first

    comparison_file = METRICS_DIR / f"flight_price_model_comparison_{run_stamp}.csv"  # timestamped comparison path
    comparison_df.to_csv(comparison_file, index=False)  # save model comparison table

    # The first ranked row becomes the promoted model for local serving and CI checks.
    best_row = comparison_df.iloc[0].to_dict()  # select the top-ranked model row
    best_result = next(row for row in results if row["model_name"] == best_row["model_name"])  # get the matching trained pipeline

    model_file = MODEL_DIR / f"flight_price_model_{run_stamp}.joblib"  # timestamped model version
    latest_model_file = MODEL_DIR / "flight_price_model_latest.joblib"  # stable deployment model path
    joblib.dump(best_result["model"], model_file)  # save permanent model version
    joblib.dump(best_result["model"], latest_model_file)  # update latest model copy

    # -----------------------------
    # 11. Save feature importance and worst prediction reports
    # -----------------------------
    final_predictions = best_result["model"].predict(X_test)  # reuse the selected model on the time-based holdout set
    prediction_report = X_test.copy()  # keep input columns beside prediction errors
    prediction_report["actual_price"] = y_test.values  # attach actual prices for review
    prediction_report["predicted_price"] = final_predictions  # attach model predictions for review
    prediction_report["absolute_error"] = (prediction_report["actual_price"] - prediction_report["predicted_price"]).abs()  # calculate absolute error

    worst_predictions_file = METRICS_DIR / f"flight_price_worst_predictions_{run_stamp}.csv"  # timestamped worst prediction report
    prediction_report.sort_values("absolute_error", ascending=False).head(50).to_csv(worst_predictions_file, index=False)  # save the largest mistakes

    feature_importance_file = METRICS_DIR / f"flight_price_feature_importance_{run_stamp}.csv"  # timestamped feature importance report
    feature_importance_saved = False  # track whether the final model exposes importance values

    try:
        # Only tree-style models expose direct feature importance in this workflow.
        feature_names = best_result["model"].named_steps["preprocess"].get_feature_names_out()  # get final transformed feature names
        trained_estimator = best_result["model"].named_steps["model"]  # get the trained algorithm from the pipeline

        if hasattr(trained_estimator, "feature_importances_"):  # tree-based models expose feature importance directly
            importance_values = trained_estimator.feature_importances_  # read importance scores from the trained model
            importance_df = pd.DataFrame({"feature": feature_names, "importance": importance_values})  # make a readable table
            importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)  # show strongest features first
            importance_df.to_csv(feature_importance_file, index=False)  # save feature importance report
            feature_importance_saved = True  # mark the report as available
    except Exception as error:
        log_message(f"Feature importance report skipped: {error}", log_file)  # keep training successful even if report fails

    metadata = {  # keep model selection details in a readable JSON file
        "project_name": "voyage_analytics",
        "model_name": "flight_price_model",
        "selected_model": best_result["model_name"],
        "version_id": run_stamp,
        "model_file": str(model_file),
        "latest_model_file": str(latest_model_file),
        "comparison_file": str(comparison_file),
        "feature_importance_file": str(feature_importance_file) if feature_importance_saved else None,
        "worst_predictions_file": str(worst_predictions_file),
        "target_column": TARGET_COLUMN,
        "model_columns": MODEL_COLUMNS,
        "split_strategy": "time_based_holdout",
        "strict_validation": "group_holdout_by_from_to_flightType_agency",
        "tuning_method": "light_randomized_search_cv",
        "cv_splits": CV_SPLITS,
        "tuning_iterations": TUNING_ITERATIONS,
        "dataset_info": dataset_info,
        "group_holdout_info": group_info,
        "best_params": best_result["best_params"],
        "metrics": {key: best_result[key] for key in best_result if key not in ["model", "best_params"]},
    }

    metadata_file = METRICS_DIR / f"flight_price_model_{run_stamp}_metadata.json"  # timestamped metadata path
    latest_metadata_file = METRICS_DIR / "flight_price_model_latest_metadata.json"  # stable metadata path
    metadata_file.write_text(json.dumps(metadata, indent=4), encoding="utf-8")  # save timestamped metadata
    latest_metadata_file.write_text(json.dumps(metadata, indent=4), encoding="utf-8")  # update latest metadata copy

    # Create one explicit MLflow run for the final promoted model artifacts.
    with mlflow.start_run(run_name=f"register_best_model_{run_stamp}"):  # keep final model registration as a separate MLflow run
        mlflow.log_params({"selected_model": best_result["model_name"], "version_id": run_stamp})  # record final selected model
        mlflow.log_metrics({key: value for key, value in best_result.items() if isinstance(value, (int, float, bool))})  # record final metrics
        mlflow.log_artifact(str(model_file), artifact_path="model")  # attach joblib file to MLflow
        mlflow.log_artifact(str(metadata_file), artifact_path="metadata")  # attach metadata file to MLflow
        mlflow.log_artifact(str(worst_predictions_file), artifact_path="reports")  # attach worst prediction report to MLflow
        if feature_importance_saved:
            mlflow.log_artifact(str(feature_importance_file), artifact_path="reports")  # attach feature importance report to MLflow
        # Cloudpickle supports the trusted project pipeline types used by sklearn and XGBoost.
        mlflow.sklearn.log_model(
            best_result["model"],
            artifact_path="sklearn_model",
            serialization_format="cloudpickle",
        )

    loaded_model = joblib.load(latest_model_file)  # reload latest model to prove the saved file works
    sanity_predictions = loaded_model.predict(X_test.head(5))  # predict first five holdout examples

    sanity_df = pd.DataFrame(  # build a small actual-vs-predicted table
        {
            "actual_price": y_test.head(5).values,
            "predicted_price": sanity_predictions,
        }
    )
    sanity_file = METRICS_DIR / f"flight_price_sanity_check_{run_stamp}.csv"  # timestamped sanity-check output
    sanity_df.to_csv(sanity_file, index=False)  # save sanity-check table

    log_message(f"Best model: {best_result['model_name']}", log_file)  # print selected model
    log_message(f"Comparison saved: {comparison_file}", log_file)  # print comparison file path
    log_message(f"Model saved: {model_file}", log_file)  # print versioned model path
    log_message(f"Latest model saved: {latest_model_file}", log_file)  # print stable model path
    log_message(f"Metadata saved: {metadata_file}", log_file)  # print metadata path
    log_message(f"Worst predictions saved: {worst_predictions_file}", log_file)  # print worst prediction report path
    if feature_importance_saved:
        log_message(f"Feature importance saved: {feature_importance_file}", log_file)  # print feature importance report path
    log_message(f"Sanity check saved: {sanity_file}", log_file)  # print sanity-check path


def main():
    # -----------------------------
    # 12. Run the local training workflow
    # -----------------------------
    prepare_folders()  # create output folders before writing files
    run_stamp, log_file = start_log_file()  # create timestamp and log path for this run

    # Default to local MLflow files, but allow Docker/Airflow/Jenkins to pass a server URI.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.resolve().as_uri())  # local by default, remote when Docker passes a server URI
    mlflow.set_tracking_uri(tracking_uri)  # store runs in the selected MLflow backend
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "flight-price-local-training")  # allow Airflow to use its own experiment
    mlflow.set_experiment(experiment_name)  # group all flight price runs under one experiment

    flights = load_flight_data(log_file)  # read the local flight dataset
    model_df, X_train, X_test, y_train, y_test, dataset_info = build_modeling_dataset(flights, log_file)  # prepare model-ready train and test data
    X_group_train, X_group_test, y_group_train, y_group_test, group_info = build_group_holdout_dataset(model_df, log_file)  # prepare stricter unseen-group data

    results = []  # collect successful model results here
    for model_name, setup in model_search_space().items():  # train every candidate model
        try:
            result = tune_and_evaluate_model(
                model_name,
                setup,
                X_train,
                X_test,
                y_train,
                y_test,
                X_group_train,
                X_group_test,
                y_group_train,
                y_group_test,
                dataset_info,
                group_info,
                log_file,
            )  # run tuning, time-split evaluation, and group-holdout evaluation
            results.append(result)  # keep successful model result for final comparison
        except Exception as error:
            log_message(f"Skipped {model_name}: {error}", log_file)  # continue even if one model fails

    # Stop the run only if every candidate failed, not when just one model breaks.
    if not results:  # fail clearly only when every model failed
        raise RuntimeError("No model completed successfully.")

    save_results(results, dataset_info, group_info, run_stamp, X_test, y_test, log_file)  # save comparison, best model, metadata, and sanity check


if __name__ == "__main__":
    main()
