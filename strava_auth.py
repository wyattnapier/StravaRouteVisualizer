#!/usr/bin/env python3
"""
strava_auth.py
==============
One-time OAuth setup for Strava. Opens your browser, waits for you to
approve access, then automatically exchanges the code for tokens and
prints your refresh token.

Usage:
    python strava_auth.py --client-id YOUR_ID --client-secret YOUR_SECRET

Or with env vars already set:
    python strava_auth.py
"""

import os
import sys
import argparse
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests

REDIRECT_PORT = 8765
REDIRECT_URI  = f"http://localhost:{REDIRECT_PORT}"
SCOPE         = "activity:read_all"
AUTH_URL      = "https://www.strava.com/oauth/authorize"
TOKEN_URL     = "https://www.strava.com/oauth/token"


# Shared state between the HTTP handler and main thread
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
        pass  # silence server logs


def wait_for_code() -> str:
    """Start a local server, block until the OAuth redirect arrives."""
    server = HTTPServer(("localhost", REDIRECT_PORT), OAuthHandler)
    server.handle_request()  # handles exactly one request then stops
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


def main():
    p = argparse.ArgumentParser(description="Strava one-time OAuth setup")
    p.add_argument("--client-id",     default=os.environ.get("STRAVA_CLIENT_ID"))
    p.add_argument("--client-secret", default=os.environ.get("STRAVA_CLIENT_SECRET"))
    args = p.parse_args()

    for name, val in [("--client-id", args.client_id), ("--client-secret", args.client_secret)]:
        if not val:
            print(f"ERROR: {name} is required (or set env var "
                  f"{name.lstrip('-').upper().replace('-','_')})")
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

    # Open browser slightly after server is ready
    threading.Timer(0.5, lambda: webbrowser.open(auth_url)).start()

    print("Waiting for authorization (approve in your browser) …")
    code = wait_for_code()
    print(f"  ✓ Authorization code received")

    print("Exchanging code for tokens …")
    tokens = exchange_code(code, args.client_id, args.client_secret)

    refresh_token = tokens.get("refresh_token")
    access_token  = tokens.get("access_token")
    athlete       = tokens.get("athlete", {})

    print(f"\n  ✓ Authenticated as: {athlete.get('firstname','')} {athlete.get('lastname','')}")
    print(f"  Scope: {tokens.get('scope','')}")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Success! Add these to your shell (e.g. ~/.zshrc):          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  export STRAVA_CLIENT_ID={args.client_id:<36}║
║  export STRAVA_CLIENT_SECRET={args.client_secret:<32}║
║  export STRAVA_REFRESH_TOKEN={refresh_token:<32}║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    print("Or just run this in your terminal right now:")
    print(f"  export STRAVA_CLIENT_ID={args.client_id}")
    print(f"  export STRAVA_CLIENT_SECRET={args.client_secret}")
    print(f"  export STRAVA_REFRESH_TOKEN={refresh_token}")


if __name__ == "__main__":
    main()
