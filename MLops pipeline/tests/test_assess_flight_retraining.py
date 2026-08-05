import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "assess_flight_retraining.py"
SPEC = importlib.util.spec_from_file_location("assess_flight_retraining", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AssessFlightRetrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset = self.root / "flights.csv"
        self.metadata = self.root / "metadata.json"
        self.drift = self.root / "drift.json"
        self.dataset.write_text("date,price\n2026-01-01,100\n", encoding="utf-8")
        self.dataset_hash = MODULE.file_sha256(self.dataset)
        self.metadata.write_text(
            json.dumps({"dataset_info": {"dataset_sha256": self.dataset_hash}}),
            encoding="utf-8",
        )
        self.drift.write_text(json.dumps({"drift_detected": False}), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def assess(self, changed_files=None, force=False):
        return MODULE.assess(
            dataset=self.dataset,
            metadata_path=self.metadata,
            drift_path=self.drift,
            changed_files=changed_files or [],
            force=force,
        )

    def test_same_dataset_does_not_retrain(self):
        self.assertFalse(self.assess()["retrain_required"])

    def test_changed_dataset_retrains(self):
        self.dataset.write_text("date,price\n2026-01-01,150\n", encoding="utf-8")
        self.assertIn("flight dataset fingerprint changed", self.assess()["reasons"])

    def test_drift_retrains(self):
        self.drift.write_text(json.dumps({"drift_detected": True}), encoding="utf-8")
        self.assertIn("dataset drift threshold crossed", self.assess()["reasons"])

    def test_training_source_change_retrains(self):
        decision = self.assess(["local_training/train_flight_price.py"])
        self.assertIn("training inputs changed in Git", decision["reasons"])

    def test_unrelated_change_does_not_retrain(self):
        self.assertFalse(self.assess(["docs/README.md"])["retrain_required"])


if __name__ == "__main__":
    unittest.main()
