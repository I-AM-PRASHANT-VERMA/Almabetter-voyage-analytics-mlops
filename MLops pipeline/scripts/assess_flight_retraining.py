import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_DATASET = REPO_ROOT / "dataset" / "travel_capstone" / "flights.csv"
DEFAULT_METADATA = PROJECT_ROOT / "joblib files" / "flight_price_model_metadata.json"
DEFAULT_DRIFT_REPORT = PROJECT_ROOT / "jenkins_artifacts" / "data_drift" / "flight_dataset_drift_summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "jenkins_artifacts" / "automation" / "flight_retraining_decision.json"

RETRAINING_PATHS = (
    "dataset/travel_capstone/flights.csv",
    "local_training/",
    "mlops pipeline/scripts/run_flight_price_mlflow_experiments.py",
    "mlops pipeline/requirements",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8", newline=None) as file:
        for chunk in iter(lambda: file.read(1024 * 1024), ""):
            digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_changed_files(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_retraining_path(path: str) -> bool:
    normalized = path.lower().lstrip("./")
    return any(normalized == prefix or normalized.startswith(prefix) for prefix in RETRAINING_PATHS)


def assess(dataset: Path, metadata_path: Path, drift_path: Path, changed_files: list[str], force: bool) -> dict:
    if not dataset.exists() or dataset.stat().st_size == 0:
        raise FileNotFoundError(f"Flight dataset is missing or empty: {dataset}")

    metadata = read_json(metadata_path)
    drift_report = read_json(drift_path)
    dataset_hash = file_sha256(dataset)
    trained_hash = metadata.get("dataset_info", {}).get("dataset_sha256")
    relevant_changes = sorted(path for path in changed_files if is_retraining_path(path))

    reasons = []
    if force:
        reasons.append("manual force flag")
    if not trained_hash:
        reasons.append("serving metadata has no dataset fingerprint")
    elif trained_hash != dataset_hash:
        reasons.append("flight dataset fingerprint changed")
    if drift_report.get("drift_detected") is True:
        reasons.append("dataset drift threshold crossed")
    if relevant_changes:
        reasons.append("training inputs changed in Git")

    return {
        "retrain_required": bool(reasons),
        "reasons": reasons,
        "dataset_sha256": dataset_hash,
        "trained_dataset_sha256": trained_hash,
        "drift_detected": drift_report.get("drift_detected"),
        "relevant_changes": relevant_changes,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Decide whether the flight model needs retraining.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--drift-report", type=Path, default=DEFAULT_DRIFT_REPORT)
    parser.add_argument("--changed-files-file", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--properties-output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    decision = assess(
        dataset=args.dataset,
        metadata_path=args.metadata,
        drift_path=args.drift_report,
        changed_files=read_changed_files(args.changed_files_file),
        force=args.force,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=4), encoding="utf-8")

    if args.properties_output:
        args.properties_output.parent.mkdir(parents=True, exist_ok=True)
        value = "true" if decision["retrain_required"] else "false"
        args.properties_output.write_text(f"RETRAIN_REQUIRED={value}\n", encoding="utf-8")

    print(json.dumps(decision, indent=4))
    print(f"Decision file: {args.output}")


if __name__ == "__main__":
    main()
