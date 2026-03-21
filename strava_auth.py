#!/usr/bin/env python3
"""
strava_auth.py
==============
One-time OAuth setup for Strava. Reads credentials from a .env file in the
same directory, opens your browser for authorization, then automatically
writes the refresh token back to the .env file.

Usage:
    python strava_auth.py

Or override credentials via flags:
    python strava_auth.py --client-id YOUR_ID --client-secret YOUR_SECRET

.env file format (same directory as this script):
    STRAVA_CLIENT_ID=12345
    STRAVA_CLIENT_SECRET=abc...
    STRAVA_REFRESH_TOKEN=   ← this will be written automatically
    OPENTOPO_API_KEY=ghi... ← left untouched
"""

import os
import sys
import argparse
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import requests

REDIRECT_PORT = 8765
REDIRECT_URI  = f"http://localhost:{REDIRECT_PORT}"
SCOPE         = "activity:read_all"
AUTH_URL      = "https://www.strava.com/oauth/authorize"
TOKEN_URL     = "https://www.strava.com/oauth/token"

ENV_FILE = Path(__file__).parent / ".env"


# ─── .env helpers ─────────────────────────────────────────────────────────────

def load_env_file(path: Path) -> dict:
    """Parse a .env file into a dict, preserving order and comments."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, val = stripped.partition("=")
            result[key.strip()] = val.strip()
    return result


def write_env_file(path: Path, updates: dict):
    """
    Write key=value pairs back to the .env file.
    - Updates existing keys in-place (preserving order and comments).
    - Appends any new keys that didn't exist before.
    """
    lines = path.read_text().splitlines() if path.exists() else []
    updated_keys = set()

    # Update existing lines
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Append any keys that weren't already in the file
    for key, val in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")

    path.write_text("\n".join(new_lines) + "\n")


# ─── OAuth flow ───────────────────────────────────────────────────────────────

_auth_code = None
_error     = None


class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _auth_code, _error
        params = parse_qs(urlparse(self.path).query)
        if "error" in params:
            _error = params["error"][0]
            self._respond("Access denied. You can close this tab.")
        elif "code" in params:
            _auth_code = params["code"][0]
            self._respond("✓ Authorized! You can close this tab and return to your terminal.")
        else:
            self._respond("Unexpected response. Check your terminal.")

    def _respond(self, message):
        body = f"<html><body style='font-family:sans-serif;padding:2em'><h2>{message}</h2></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


def wait_for_code() -> str:
    server = HTTPServer(("localhost", REDIRECT_PORT), OAuthHandler)
    server.handle_request()
    if _error:
        print(f"\nERROR: Strava returned: {_error}")
        sys.exit(1)
    return _auth_code


def exchange_code(code: str, client_id: str, client_secret: str) -> dict:
    resp = requests.post(TOKEN_URL, data={
        "client_id":     client_id,
        "client_secret": client_secret,
        "code":          code,
        "grant_type":    "authorization_code",
    })
    resp.raise_for_status()
    return resp.json()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load .env first so its values are available as fallbacks
    env_vars = load_env_file(ENV_FILE)
    if env_vars:
        print(f"  Loaded {ENV_FILE}")

    p = argparse.ArgumentParser(description="Strava one-time OAuth setup")
    p.add_argument("--client-id",     default=env_vars.get("STRAVA_CLIENT_ID")     or os.environ.get("STRAVA_CLIENT_ID"))
    p.add_argument("--client-secret", default=env_vars.get("STRAVA_CLIENT_SECRET") or os.environ.get("STRAVA_CLIENT_SECRET"))
    args = p.parse_args()

    for name, val in [("--client-id", args.client_id), ("--client-secret", args.client_secret)]:
        if not val:
            print(f"\nERROR: {name} is required.")
            if ENV_FILE.exists():
                print(f"  Add it to {ENV_FILE}:  {name.lstrip('-').upper().replace('-','_')}=your_value")
            else:
                print(f"  Create a .env file next to this script with:")
                print(f"    STRAVA_CLIENT_ID=your_id")
                print(f"    STRAVA_CLIENT_SECRET=your_secret")
            sys.exit(1)

    auth_url = (
        f"{AUTH_URL}"
        f"?client_id={args.client_id}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=force"
        f"&scope={SCOPE}"
    )

    print(f"\nOpening Strava authorization in your browser …")
    print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")

    threading.Timer(0.5, lambda: webbrowser.open(auth_url)).start()

    print("Waiting for you to approve access in your browser …")
    code = wait_for_code()
    print(f"  ✓ Authorization code received")

    print("Exchanging code for tokens …")
    tokens = exchange_code(code, args.client_id, args.client_secret)

    refresh_token = tokens.get("refresh_token")
    athlete       = tokens.get("athlete", {})

    print(f"  ✓ Authenticated as: {athlete.get('firstname','')} {athlete.get('lastname','')}")
    print(f"  Scope: {tokens.get('scope','')}")

    # Write refresh token back to .env
    write_env_file(ENV_FILE, {"STRAVA_REFRESH_TOKEN": refresh_token})
    print(f"\n  ✓ STRAVA_REFRESH_TOKEN written to {ENV_FILE}")
    print(f"    You're all set — just run strava_to_3d.py")


if __name__ == "__main__":
    main()
