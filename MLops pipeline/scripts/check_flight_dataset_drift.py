import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


# -----------------------------
# 1. Resolve dataset and report paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # points to the MLops pipeline folder
REPO_ROOT = PROJECT_ROOT.parent  # fallback root for shared dataset and baseline files

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv"  # first try the pipeline dataset path
    if (PROJECT_ROOT / "dataset" / "travel_capstone" / "flights.csv").exists()
    else REPO_ROOT / "dataset" / "travel_capstone" / "flights.csv"  # fallback to the repository dataset path
)
DEFAULT_BASELINE_PATH = (
    PROJECT_ROOT / "dataset" / "baselines" / "flight_dataset_baseline.json"  # first try pipeline baseline storage
    if (PROJECT_ROOT / "dataset" / "baselines" / "flight_dataset_baseline.json").exists()
    else REPO_ROOT / "dataset" / "baselines" / "flight_dataset_baseline.json"  # fallback to repo-level baseline storage
)
DATASET_PATH = Path(os.getenv("FLIGHT_DATASET_PATH", DEFAULT_DATASET_PATH))  # allow Airflow/Docker to override the dataset path
DEFAULT_REPORT_PATH = PROJECT_ROOT / "jenkins_artifacts" / "data_drift" / "flight_dataset_drift_summary.json"  # stable drift report path

CATEGORICAL_COLUMNS = ["from", "to", "flightType", "agency"]  # compare route, type, and agency mix
NUMERIC_COLUMNS = ["price", "time", "distance"]  # compare key numeric averages
REQUIRED_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS  # fail clearly if any important column is missing


# -----------------------------
# 2. Print readable paths
# -----------------------------
def display_path(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))  # prefer short paths inside the pipeline folder
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))  # use repo-relative paths for shared files
        except ValueError:
            return str(path)  # keep full path for anything outside the project roots


# -----------------------------
# 3. Calculate safe percentage change
# -----------------------------
def relative_change(current_value, baseline_value):
    if baseline_value == 0:
        return 0.0 if current_value == 0 else 1.0  # avoid division by zero while still flagging a new non-zero value
    return abs(current_value - baseline_value) / abs(baseline_value)  # standard relative change formula


# -----------------------------
# 4. Build current dataset snapshot
# -----------------------------
def build_snapshot(dataset_path):
    flights_df = pd.read_csv(dataset_path)  # load the current flight dataset

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in flights_df.columns]  # validate the columns used by drift checks
    if missing_columns:
        raise ValueError(f"Flight dataset is missing required columns: {missing_columns}")  # stop with a clear schema error

    return {
        "dataset_file": display_path(dataset_path),  # store the dataset used for this snapshot
        "captured_at": datetime.now().isoformat(timespec="seconds"),  # timestamp helps audit when the check ran
        "row_count": int(len(flights_df)),  # row count drift can indicate a changed dataset extract
        "columns": list(flights_df.columns),  # keep full layout so schema changes are visible
        "missing_values": {column: int(value) for column, value in flights_df.isna().sum().items()},  # track data quality shifts
        "numeric_means": {
            column: round(float(flights_df[column].mean()), 6) for column in NUMERIC_COLUMNS  # compare average price/time/distance
        },
        "category_proportions": {
            column: {
                str(category): round(float(proportion), 6)
                for category, proportion in flights_df[column].value_counts(normalize=True).items()  # convert counts into proportions
            }
            for column in CATEGORICAL_COLUMNS  # compare distribution for each important categorical feature
        },
    }


# -----------------------------
# 5. Compare category distributions
# -----------------------------
def category_distribution_change(current_values, baseline_values):
    categories = set(current_values).union(baseline_values)  # include categories that exist in only one snapshot
    return sum(
        abs(current_values.get(category, 0.0) - baseline_values.get(category, 0.0))
        for category in categories
    ) / 2  # total variation distance for a simple distribution-change score


# -----------------------------
# 6. Compare current snapshot with baseline
# -----------------------------
def compare_snapshots(current_snapshot, baseline_snapshot):
    checks = {
        "row_count_change": relative_change(
            current_snapshot["row_count"], baseline_snapshot["row_count"]  # compare dataset size
        ),
        "numeric_mean_changes": {
            column: relative_change(
                current_snapshot["numeric_means"][column],  # current numeric average
                baseline_snapshot["numeric_means"][column],  # baseline numeric average
            )
            for column in NUMERIC_COLUMNS
        },
        "category_distribution_changes": {
            column: category_distribution_change(
                current_snapshot["category_proportions"][column],  # current category mix
                baseline_snapshot["category_proportions"][column],  # baseline category mix
            )
            for column in CATEGORICAL_COLUMNS
        },
        "column_layout_matches": current_snapshot["columns"] == baseline_snapshot["columns"],  # catch schema/order changes
    }

    drift_detected = (
        checks["row_count_change"] > 0.10  # row count changed by more than 10 percent
        or any(value > 0.10 for value in checks["numeric_mean_changes"].values())  # numeric averages shifted enough to flag
        or any(value > 0.10 for value in checks["category_distribution_changes"].values())  # categorical mix shifted enough to flag
        or not checks["column_layout_matches"]  # schema changed
    )
    return checks, drift_detected  # return both detailed checks and the final decision


# -----------------------------
# 7. Save JSON output
# -----------------------------
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)  # create report/baseline folder if needed
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")  # save readable JSON for review and automation


# -----------------------------
# 8. Run complete drift check
# -----------------------------
def run_check(args):
    current_snapshot = build_snapshot(args.dataset)  # profile the current dataset
    baseline_exists = args.baseline.exists()  # decide whether to load or create the baseline

    if not baseline_exists or args.refresh_baseline:
        write_json(args.baseline, current_snapshot)  # create or replace the baseline snapshot
        baseline_snapshot = current_snapshot  # compare current data with itself after baseline creation
        baseline_status = "created" if not baseline_exists else "refreshed"  # make the report clear
    else:
        baseline_snapshot = json.loads(args.baseline.read_text(encoding="utf-8"))  # load the existing reference snapshot
        baseline_status = "loaded"  # baseline already existed and was reused

    checks, drift_detected = compare_snapshots(current_snapshot, baseline_snapshot)  # run all drift comparisons
    report = {
        "baseline_status": baseline_status,  # tells whether baseline was loaded, created, or refreshed
        "baseline_file": display_path(args.baseline),  # baseline path used for comparison
        "drift_detected": drift_detected,  # final drift decision
        "checks": checks,  # detailed row, numeric, category, and schema checks
        "current_snapshot": current_snapshot,  # saved so the report can be inspected later
    }
    write_json(args.report, report)  # save the drift report for Airflow/Jenkins/manual review

    print(f"Dataset drift detected: {drift_detected}")  # visible status in terminal or Airflow logs
    print(f"Baseline status: {baseline_status}")  # shows whether the baseline was reused or updated
    print(f"Report file: {display_path(args.report)}")  # tells where the JSON report was written

    if drift_detected and args.fail_on_drift:
        raise SystemExit("Dataset drift crossed the configured threshold.")  # optional hard stop for automated pipelines


# -----------------------------
# 9. Read command line options
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compare the current flight dataset with its baseline.")  # CLI wrapper for local and pipeline runs
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)  # custom dataset path
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)  # custom baseline path
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)  # custom output report path
    parser.add_argument("--refresh-baseline", action="store_true")  # replace baseline with the current dataset snapshot
    parser.add_argument("--fail-on-drift", action="store_true")  # exit with failure when drift is detected
    return parser.parse_args()  # return parsed CLI options


# -----------------------------
# 10. Script entry point
# -----------------------------
if __name__ == "__main__":
    run_check(parse_args())  # run the drift check only when executed as a script
