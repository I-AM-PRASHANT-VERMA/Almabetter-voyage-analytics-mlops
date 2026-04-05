import os
import subprocess
import sys

from pathlib import Path

from pyngrok import ngrok


# -----------------------------
# 1. Resolve Streamlit app path
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # point to the pipeline folder.
STREAMLIT_APP_PATH = PROJECT_ROOT / "streamlit" / "flight_price_app.py"
STREAMLIT_PORT = 8501


# -----------------------------
# 2. Start ngrok and Streamlit
# -----------------------------
def main():
    auth_token = os.getenv("NGROK_AUTHTOKEN", "").strip()

    if auth_token:
        ngrok.set_auth_token(auth_token)

    public_tunnel = ngrok.connect(addr=STREAMLIT_PORT, proto="http")

    print("Flight Price Streamlit website is starting.")
    print(f"Local URL: http://127.0.0.1:{STREAMLIT_PORT}")
    print(f"Public URL: {public_tunnel.public_url}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP_PATH),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(STREAMLIT_PORT),
        "--server.headless",
        "true",
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


# -----------------------------
# 3. Script entry point
# -----------------------------
if __name__ == "__main__":
    main()

