import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "trigger_jenkins_cd.py"
SPEC = importlib.util.spec_from_file_location("trigger_jenkins_cd", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TriggerJenkinsCdTests(unittest.TestCase):
    def test_true_values_enable_deployment(self):
        for value in ["true", "TRUE", "1", "yes", "on"]:
            self.assertTrue(MODULE.is_enabled(value))

    def test_false_values_keep_deployment_disabled(self):
        for value in [None, "", "false", "0", "disabled"]:
            self.assertFalse(MODULE.is_enabled(value))

    def test_trigger_passes_safe_aks_parameters(self):
        class FakeResponse:
            def __init__(self, payload=b"", location=None):
                self.payload = payload
                self.headers = {"Location": location} if location else {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return self.payload

        responses = [
            FakeResponse(json.dumps({"crumbRequestField": "Jenkins-Crumb", "crumb": "test"}).encode("utf-8")),
            FakeResponse(location="http://jenkins/queue/item/1/"),
        ]

        with patch.object(MODULE.urllib.request, "urlopen", side_effect=responses) as urlopen:
            location = MODULE.trigger_cd("http://jenkins", "voyage-analytics-mlops-cd", "user", "password")

        self.assertEqual(location, "http://jenkins/queue/item/1/")
        build_request = urlopen.call_args_list[1].args[0]
        self.assertIn("DEPLOY_TO_AKS=true", build_request.full_url)
        self.assertIn("START_AKS_IF_STOPPED=true", build_request.full_url)


if __name__ == "__main__":
    unittest.main()
