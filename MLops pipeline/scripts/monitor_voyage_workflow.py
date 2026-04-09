"""Local workflow monitor for the Voyage Analytics MLOps project."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from http.client import RemoteDisconnected
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "monitoring_reports"
DEFAULT_TIMEOUT = 10

# -----------------------------
# 1. Define HTTP health checks
# -----------------------------
HTTP_CHECKS = [
    {
        "name": "MLflow UI",
        "url": "http://127.0.0.1:5004/health",
        "method": "GET",
    },
    {
        "name": "Airflow health",
        "url": "http://127.0.0.1:8080/api/v2/monitor/health",
        "method": "GET",
    },
    {
        "name": "Jenkins home",
        "url": "http://127.0.0.1:8081/",
        "method": "GET",
        "allow_status": [200, 403],
    },
    {
        "name": "Flight API health",
        "url": "http://127.0.0.1:5002/health",
        "method": "GET",
    },
    {
        "name": "Hotel API health",
        "url": "http://127.0.0.1:5001/health",
        "method": "GET",
    },
    {
        "name": "Gender API health",
        "url": "http://127.0.0.1:5003/health",
        "method": "GET",
    },
    {
        "name": "Flight Streamlit health",
        "url": "http://127.0.0.1:8501/app/flight/_stcore/health",
        "method": "GET",
    },
    {
        "name": "Hotel Streamlit health",
        "url": "http://127.0.0.1:8502/app/hotel/_stcore/health",
        "method": "GET",
    },
    {
        "name": "Gender Streamlit health",
        "url": "http://127.0.0.1:8503/app/gender/_stcore/health",
        "method": "GET",
    },
    # Check the common gateway itself before testing any routed app paths through it.
    {
        "name": "Gateway health",
        "url": "http://127.0.0.1:8090/health",
        "method": "GET",
    },
    # Check that the gateway can forward traffic to the flight API.
    {
        "name": "Gateway flight API health",
        "url": "http://127.0.0.1:8090/api/flight/health",
        "method": "GET",
    },
    # Check that the gateway can forward traffic to the hotel API.
    {
        "name": "Gateway hotel API health",
        "url": "http://127.0.0.1:8090/api/hotel/health",
        "method": "GET",
    },
    # Check that the gateway can forward traffic to the gender API.
    {
        "name": "Gateway gender API health",
        "url": "http://127.0.0.1:8090/api/gender/health",
        "method": "GET",
    },
    # Check that the gateway can forward traffic to the flight Streamlit app.
    {
        "name": "Gateway flight Streamlit health",
        "url": "http://127.0.0.1:8090/app/flight/_stcore/health",
        "method": "GET",
    },
    # Check that the gateway can forward traffic to the hotel Streamlit app.
    {
        "name": "Gateway hotel Streamlit health",
        "url": "http://127.0.0.1:8090/app/hotel/_stcore/health",
        "method": "GET",
    },
    # Check that the gateway can forward traffic to the gender Streamlit app.
    {
        "name": "Gateway gender Streamlit health",
        "url": "http://127.0.0.1:8090/app/gender/_stcore/health",
        "method": "GET",
    },
]

# -----------------------------
# 2. Define prediction smoke checks
# -----------------------------
PREDICTION_CHECKS = [
    {
        "name": "Flight prediction",
        "url": "http://127.0.0.1:5002/predict",
        "payload": {
            "time": 1.76,
            "year": 2019,
            "month": 9,
            "day": 26,
            "from": "Recife (PE)",
            "to": "Florianopolis (SC)",
            "flightType": "firstClass",
            "agency": "FlyingDrops",
        },
        "required_keys": ["predicted_price"],
    },
    {
        "name": "Hotel recommendation",
        "url": "http://127.0.0.1:5001/popular-hotels?top_n=3",
        "payload": None,
        "required_keys": ["results"],
    },
    {
        "name": "Gender prediction",
        "url": "http://127.0.0.1:5003/predict",
        "payload": {"name": "Priya"},
        "required_keys": ["predicted_gender"],
    },
]

# -----------------------------
# 3. Define expected Docker containers
# -----------------------------
DOCKER_CONTAINERS = [
    "1-project-flight-price-voyage-api",
    "1-project-flight-price-voyage-streamlit",
    "1-project-hotel-recommendation-voyage-api",
    "1-project-hotel-recommendation-voyage-streamlit",
    "1-project-gender-classification-voyage-api",
    "1-project-gender-classification-voyage-streamlit",
    # Include the shared nginx front door in the local container health list.
    "1-project-voyage-analytics-gateway",
    "1-project-voyage-analytics-mlflow-ui",
    "1-project-voyage-analytics-jenkins",
]


# -----------------------------
# 4. Format report timestamp
# -----------------------------
def utc_now_text() -> str:
    # Use UTC so reports are easy to compare across tools and machines.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -----------------------------
# 5. Run external command safely
# -----------------------------
def run_command(command: list[str], timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    # Run external tools like Docker and kubectl without opening another shell.
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as error:
        return {
            "ok": False,
            "stdout": "",
            "stderr": str(error),
            "return_code": None,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "ok": False,
            "stdout": error.stdout or "",
            "stderr": f"Command timed out after {timeout} seconds.",
            "return_code": None,
        }

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "return_code": completed.returncode,
    }


# -----------------------------
# 6. Build Jenkins auth header
# -----------------------------
def build_basic_auth_header(username: str | None, password: str | None) -> dict[str, str]:
    # Jenkins can be checked with credentials when they are provided through env vars.
    if not username or not password:
        return {}

    token = f"{username}:{password}".encode("utf-8")
    encoded_token = base64.b64encode(token).decode("utf-8")

    return {"Authorization": f"Basic {encoded_token}"}


# -----------------------------
# 7. Call HTTP endpoint
# -----------------------------
def call_http(
    name: str,
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    allow_status: list[int] | None = None,
) -> dict[str, Any]:
    # Convert JSON payloads into bytes only when the check needs a POST request.
    body = json.dumps(payload).encode("utf-8") if payload is not None else None

    # Keep headers small and explicit so the APIs receive normal JSON requests.
    request_headers = {"Accept": "application/json"}

    # Add Content-Type only for JSON request bodies.
    if payload is not None:
        request_headers["Content-Type"] = "application/json"

    # Merge optional auth headers after the default headers are ready.
    if headers:
        request_headers.update(headers)

    # Build a urllib request so the script stays dependency-free.
    request = Request(url=url, data=body, headers=request_headers, method=method)

    # Treat these response codes as success for this specific check.
    accepted_statuses = set(allow_status or [200])

    started_at = time.perf_counter()

    try:
        # Open this resource in a controlled context.
        with urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            status_code = int(response.status)
    except HTTPError as error:
        response_text = error.read().decode("utf-8", errors="replace")
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        status_code = int(error.code)
    except (RemoteDisconnected, ConnectionResetError, OSError) as error:
        return {
            "name": name,
            "ok": False,
            "status_code": None,
            "duration_ms": None,
            "message": str(error),
            "url": url,
        }
    except URLError as error:
        return {
            "name": name,
            "ok": False,
            "status_code": None,
            "duration_ms": None,
            "message": str(error.reason),
            "url": url,
        }
    except TimeoutError:
        return {
            "name": name,
            "ok": False,
            "status_code": None,
            "duration_ms": None,
            "message": f"Request timed out after {timeout} seconds.",
            "url": url,
        }

    return {
        "name": name,
        "ok": status_code in accepted_statuses,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "message": response_text[:500],
        "response_text": response_text,
        "url": url,
    }


# -----------------------------
# 8. Check local services
# -----------------------------
def check_http_services(timeout: int) -> list[dict[str, Any]]:
    # Check every browser/API surface that should be reachable on localhost.
    results = []

    # Jenkins may need credentials, so read them from env instead of hard-coding them.
    jenkins_headers = build_basic_auth_header(
        os.getenv("JENKINS_USER"),
        os.getenv("JENKINS_TOKEN") or os.getenv("JENKINS_PASSWORD"),
    )

    # Loop through each item that needs the same handling.
    for check in HTTP_CHECKS:
        headers = jenkins_headers if check["name"].startswith("Jenkins") else None
        result = call_http(
            name=check["name"],
            url=check["url"],
            method=check.get("method", "GET"),
            timeout=timeout,
            headers=headers,
            allow_status=check.get("allow_status"),
        )
        results.append(result)

    return results


# -----------------------------
# 9. Check model predictions
# -----------------------------
def check_predictions(timeout: int) -> list[dict[str, Any]]:
    # Run one realistic prediction/recommendation request for each model-facing API.
    results = []

    # Loop through each item that needs the same handling.
    for check in PREDICTION_CHECKS:
        method = "POST" if check["payload"] is not None else "GET"
        result = call_http(
            name=check["name"],
            url=check["url"],
            method=method,
            payload=check["payload"],
            timeout=timeout,
        )

        if result["ok"]:
            try:
                response_json = json.loads(result.get("response_text", result["message"]))
                missing_keys = [
                    key for key in check["required_keys"] if key not in response_json
                ]
                result["ok"] = not missing_keys
                result["message"] = "Required response keys found." if not missing_keys else f"Missing keys: {missing_keys}"
                result["response_preview"] = response_json
            except json.JSONDecodeError:
                result["ok"] = False
                result["message"] = "Response was not valid JSON."

        results.append(result)

    return results


# -----------------------------
# 10. Check Docker containers
# -----------------------------
def check_docker_containers(timeout: int) -> list[dict[str, Any]]:
    # Inspect named containers because Docker Compose may include optional profiles.
    results = []

    # Loop through each item that needs the same handling.
    for container_name in DOCKER_CONTAINERS:
        command = [
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            container_name,
        ]
        output = run_command(command, timeout=timeout)

        if not output["ok"]:
            results.append(
                {
                    "name": container_name,
                    "ok": False,
                    "status": "missing",
                    "health": "unknown",
                    "message": output["stderr"] or output["stdout"],
                }
            )
            continue

        status, health = output["stdout"].split("|", maxsplit=1)
        is_running = status == "running"
        is_healthy = health in {"healthy", "none"}

        results.append(
            {
                "name": container_name,
                "ok": is_running and is_healthy,
                "status": status,
                "health": health,
                "message": "Container is usable." if is_running and is_healthy else "Container is not ready.",
            }
        )

    return results


# -----------------------------
# 11. Check Jenkins job through API
# -----------------------------
def check_jenkins_job(timeout: int) -> dict[str, Any]:
    # Jenkins exposes the last build status through a small JSON endpoint.
    local_result = check_jenkins_job_from_files()

    if local_result["ok"]:
        return local_result

    job_url = os.getenv(
        "JENKINS_JOB_URL",
        "http://127.0.0.1:8081/job/voyage-analytics-mlops-cd/lastBuild/api/json",
    )

    headers = build_basic_auth_header(
        os.getenv("JENKINS_USER"),
        os.getenv("JENKINS_TOKEN") or os.getenv("JENKINS_PASSWORD"),
    )

    result = call_http(
        name="Jenkins CD last build",
        url=job_url,
        method="GET",
        timeout=timeout,
        headers=headers,
        allow_status=[200],
    )

    if not result["ok"]:
        return result

    try:
        build_data = json.loads(result.get("response_text", result["message"]))
    except json.JSONDecodeError:
        result["ok"] = False
        result["message"] = "Jenkins returned a non-JSON response."
        return result

    build_result = build_data.get("result")
    result["ok"] = build_result == "SUCCESS"
    result["build_number"] = build_data.get("number")
    result["build_result"] = build_result
    result["message"] = f"Last Jenkins result: {build_result}"

    return result


# -----------------------------
# 12. Check Jenkins job from local files
# -----------------------------
def check_jenkins_job_from_files() -> dict[str, Any]:
    # Read the local Jenkins build files so monitoring does not need a password.
    builds_dir = PROJECT_ROOT / "jenkins" / "jenkins_home" / "jobs" / "voyage-analytics-mlops-cd" / "builds"

    # If Jenkins has not been initialized yet, fall back to the HTTP check.
    if not builds_dir.exists():
        return {
            "name": "Jenkins CD last build",
            "ok": False,
            "message": "Local Jenkins build folder was not found.",
        }

    # Keep only numeric build folders because Jenkins also stores permalink files here.
    build_numbers = [
        int(path.name)
        # Loop through each item that needs the same handling.
        for path in builds_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]

    # No build means the CD pipeline has not run yet.
    if not build_numbers:
        return {
            "name": "Jenkins CD last build",
            "ok": False,
            "message": "No Jenkins CD builds were found.",
        }

    # The highest folder number is the latest completed or attempted build.
    latest_build_number = max(build_numbers)

    # Jenkins stores the build result in this XML file.
    build_xml_path = builds_dir / str(latest_build_number) / "build.xml"

    # Missing XML means the build folder is incomplete.
    if not build_xml_path.exists():
        return {
            "name": "Jenkins CD last build",
            "ok": False,
            "build_number": latest_build_number,
            "message": f"Build XML was not found for build {latest_build_number}.",
        }

    # Parse XML instead of using string matching so the result read stays reliable.
    build_xml = ElementTree.parse(build_xml_path)

    # Jenkins stores the final build status in the first direct result tag.
    result_node = build_xml.getroot().find("result")

    # A missing result usually means the build is still running.
    build_result = result_node.text if result_node is not None else "RUNNING"

    return {
        "name": "Jenkins CD last build",
        "ok": build_result == "SUCCESS",
        "build_number": latest_build_number,
        "build_result": build_result,
        "message": f"Last local Jenkins CD build #{latest_build_number}: {build_result}",
    }


# -----------------------------
# 13. Check Airflow health
# -----------------------------
def check_airflow_health(timeout: int) -> dict[str, Any]:
    # Airflow health tells us whether metadatabase, scheduler, and triggerer are alive.
    result = call_http(
        name="Airflow component health",
        url="http://127.0.0.1:8080/api/v2/monitor/health",
        method="GET",
        timeout=timeout,
    )

    if not result["ok"]:
        return result

    try:
        health_data = json.loads(result.get("response_text", result["message"]))
    except json.JSONDecodeError:
        result["ok"] = False
        result["message"] = "Airflow health response was not valid JSON."
        return result

    unhealthy_parts = [
        name
        # Loop through each item that needs the same handling.
        for name, detail in health_data.items()
        if isinstance(detail, dict) and detail.get("status") != "healthy"
    ]

    result["ok"] = not unhealthy_parts
    result["message"] = "Airflow components are healthy." if not unhealthy_parts else f"Unhealthy parts: {unhealthy_parts}"
    result["response_preview"] = health_data

    return result


# -----------------------------
# 14. Check Kubernetes resources
# -----------------------------
def check_kubernetes(timeout: int) -> dict[str, Any]:
    # Kubernetes should show all Voyage pods as Running or Completed.
    command = [
        "kubectl",
        "-n",
        "voyage-mlops",
        "get",
        "pods",
        "-o",
        "json",
    ]

    output = run_command(command, timeout=timeout)

    if not output["ok"]:
        return {
            "name": "Kubernetes pods",
            "ok": False,
            "message": output["stderr"] or output["stdout"],
        }

    pod_data = json.loads(output["stdout"])
    bad_pods = []

    # Loop through each item that needs the same handling.
    for pod in pod_data.get("items", []):
        pod_name = pod["metadata"]["name"]
        pod_phase = pod["status"].get("phase", "Unknown")
        pod_ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            # Loop through each item that needs the same handling.
            for condition in pod["status"].get("conditions", [])
        )

        if pod_phase not in {"Running", "Succeeded"}:
            bad_pods.append(f"{pod_name}:{pod_phase}")
        # Handle this alternate condition.
        elif pod_phase == "Running" and not pod_ready:
            bad_pods.append(f"{pod_name}:not-ready")

    return {
        "name": "Kubernetes pods",
        "ok": not bad_pods,
        "message": "All Kubernetes pods are ready." if not bad_pods else f"Problem pods: {bad_pods}",
    }


# -----------------------------
# 15. Check dataset drift
# -----------------------------
def check_dataset_drift(timeout: int, output_dir: Path) -> dict[str, Any]:
    # Reuse the existing project drift script so the monitor follows the same rules as CI.
    report_path = output_dir / "flight_dataset_drift_summary.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "check_flight_dataset_drift.py"),
        "--report",
        str(report_path),
    ]

    output = run_command(command, timeout=timeout)

    return {
        "name": "Flight dataset drift",
        "ok": output["ok"],
        "message": output["stdout"] or output["stderr"],
        "report_path": str(report_path),
    }


# -----------------------------
# 16. Flatten report results
# -----------------------------
def flatten_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    # Convert grouped checks into one list so summary and markdown writing stay simple.
    all_results = []

    # Loop through each item that needs the same handling.
    for group_name in ["http", "predictions", "docker"]:
        all_results.extend(report["checks"].get(group_name, []))

    # Loop through each item that needs the same handling.
    for group_name in ["jenkins", "airflow", "kubernetes", "dataset_drift"]:
        all_results.append(report["checks"][group_name])

    return all_results


# -----------------------------
# 17. Write JSON report
# -----------------------------
def write_json_report(report: dict[str, Any], output_dir: Path) -> Path:
    # JSON is the machine-readable report for Jenkins or future automation.
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latest_workflow_health.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


# -----------------------------
# 18. Write Markdown report
# -----------------------------
def write_markdown_report(report: dict[str, Any], output_dir: Path) -> Path:
    # Markdown is the human-readable report for quick local review.
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latest_workflow_health.md"
    lines = [
        "# Voyage Workflow Health Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Overall status: **{report['overall_status']}**",
        "",
        "| Check | Status | Message |",
        "| --- | --- | --- |",
    ]

    # Loop through each item that needs the same handling.
    for result in flatten_results(report):
        status = "PASS" if result.get("ok") else "FAIL"
        message = str(result.get("message", "")).replace("\n", " ")[:220]
        lines.append(f"| {result.get('name')} | {status} | {message} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# -----------------------------
# 19. Build full monitor report
# -----------------------------
def build_report(args: argparse.Namespace) -> dict[str, Any]:
    # Collect every local workflow check into one structured report.
    output_dir = Path(args.output_dir)

    checks = {
        "http": check_http_services(args.timeout),
        "predictions": check_predictions(args.timeout),
        "docker": check_docker_containers(args.timeout),
        "jenkins": check_jenkins_job(args.timeout),
        "airflow": check_airflow_health(args.timeout),
        "kubernetes": check_kubernetes(args.timeout),
        "dataset_drift": check_dataset_drift(args.timeout, output_dir),
    }

    report = {
        "generated_at": utc_now_text(),
        "project_root": str(PROJECT_ROOT),
        "checks": checks,
    }

    failed_checks = [
        result["name"]
        # Loop through each item that needs the same handling.
        for result in flatten_results(report)
        if not result.get("ok")
    ]

    report["overall_status"] = "healthy" if not failed_checks else "attention_needed"
    report["failed_checks"] = failed_checks

    return report


# -----------------------------
# 20. Read CLI options
# -----------------------------
def parse_args() -> argparse.Namespace:
    # Keep CLI options small because this monitor is meant to be run often.
    parser = argparse.ArgumentParser(
        description="Check the local Voyage Analytics MLOps workflow health."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where JSON and Markdown health reports will be saved.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for each network or CLI check.",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Return exit code 1 when any workflow check fails.",
    )
    return parser.parse_args()


# -----------------------------
# 21. Run monitor command
# -----------------------------
def main() -> int:
    # Parse the command line before any checks run.
    args = parse_args()

    # Build the full health report from all local workflow checks.
    report = build_report(args)

    # Save the machine-readable report first for automation.
    json_path = write_json_report(report, Path(args.output_dir))

    # Save the readable report next for manual project review.
    markdown_path = write_markdown_report(report, Path(args.output_dir))

    # Print one concise console summary for quick local alerts.
    print(f"Workflow status: {report['overall_status']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    # Print failed checks clearly so the terminal itself behaves like an alert.
    if report["failed_checks"]:
        print("Failed checks:")
        # Loop through each item that needs the same handling.
        for check_name in report["failed_checks"]:
            print(f"- {check_name}")

    # Let Jenkins or scheduled jobs fail when alert mode is enabled.
    if args.fail_on_alert and report["failed_checks"]:
        return 1

    return 0


# -----------------------------
# 22. Script entry point
# -----------------------------
if __name__ == "__main__":
    # Stop execution with a clear error when the input is invalid.
    raise SystemExit(main())
