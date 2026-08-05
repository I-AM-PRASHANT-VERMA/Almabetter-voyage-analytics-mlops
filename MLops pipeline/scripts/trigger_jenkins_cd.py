import argparse
import base64
import http.cookiejar
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}


def is_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def read_decision(path: Path | None) -> dict:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_opener():
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def request(opener, url: str, username: str, password: str, method: str = "GET", headers: dict | None = None, data: bytes | None = None):
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    request_headers = {"Authorization": f"Basic {token}"}
    request_headers.update(headers or {})
    http_request = urllib.request.Request(url, method=method, headers=request_headers, data=data)
    return opener.open(http_request, timeout=30)


def get_crumb(opener, jenkins_url: str, username: str, password: str) -> dict[str, str]:
    crumb_url = f"{jenkins_url.rstrip('/')}/crumbIssuer/api/json"
    try:
        with request(opener, crumb_url, username, password) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {payload["crumbRequestField"]: payload["crumb"]}
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise


def trigger_cd(jenkins_url: str, job_name: str, username: str, password: str) -> str:
    form_data = urllib.parse.urlencode(
        {
            "DEPLOY_TO_AKS": "true",
            "START_AKS_IF_STOPPED": "true",
            "IMAGE_TAG": "",
        }
    ).encode("utf-8")
    job_path = urllib.parse.quote(job_name, safe="")
    build_url = f"{jenkins_url.rstrip('/')}/job/{job_path}/buildWithParameters"
    opener = build_opener()
    headers = get_crumb(opener, jenkins_url, username, password)
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    with request(opener, build_url, username, password, method="POST", headers=headers, data=form_data) as response:
        return response.headers.get("Location", build_url)


def parse_args():
    parser = argparse.ArgumentParser(description="Trigger the gated Voyage Analytics Jenkins CD job.")
    parser.add_argument("--decision-file", type=Path)
    parser.add_argument("--only-if-retraining-required", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not is_enabled(os.getenv("VOYAGE_AZURE_DEPLOYMENT_ENABLED")):
        print("Azure CD trigger skipped: VOYAGE_AZURE_DEPLOYMENT_ENABLED is false.")
        return

    decision = read_decision(args.decision_file)
    if args.only_if_retraining_required and not decision.get("retrain_required"):
        print("Azure CD trigger skipped: retraining was not required.")
        return

    username = os.getenv("JENKINS_ADMIN_ID", "").strip()
    password = os.getenv("JENKINS_ADMIN_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("Jenkins credentials are required when the Azure CD switch is enabled.")

    location = trigger_cd(
        jenkins_url=os.getenv("JENKINS_INTERNAL_URL", "http://localhost:8080"),
        job_name=os.getenv("JENKINS_CD_JOB", "voyage-analytics-mlops-cd"),
        username=username,
        password=password,
    )
    print(f"Azure CD job accepted by Jenkins: {location}")


if __name__ == "__main__":
    main()
