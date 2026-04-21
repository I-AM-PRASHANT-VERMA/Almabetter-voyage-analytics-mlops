import os

import sys

from pathlib import Path

from pyngrok import ngrok


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def start_ngrok_for_app(flask_app, port, app_title):
    # One helper keeps the three ngrok launchers consistent and avoids
    # repeating the same tunnel setup in every small wrapper script.
    auth_token = os.getenv("NGROK_AUTHTOKEN", "").strip()

    # Use the caller's ngrok account when a token is available.
    if auth_token:
        ngrok.set_auth_token(auth_token)

    # Open the public tunnel before Flask starts so the terminal can print the URL right away.
    public_tunnel = ngrok.connect(addr=port, proto="http")

    print(f"{app_title} is starting.")

    print(f"Local URL: http://127.0.0.1:{port}")

    print(f"Public URL: {public_tunnel.public_url}")

    # Disable the reloader here so ngrok does not end up with duplicate Flask processes.
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
