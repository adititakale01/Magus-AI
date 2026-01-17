"""
Run the Freight Agent API with ngrok tunnel.

This creates a public URL that your teammates can access from anywhere!

Usage:
    python run_with_ngrok.py

The script will print a public URL like:
    https://abc123.ngrok-free.app

Share this URL with your team!
"""

import os
import sys
import threading
import time
from dotenv import load_dotenv

load_dotenv()

def main():
    # Import here to avoid issues if pyngrok not installed
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("ERROR: pyngrok not installed. Run: pip install pyngrok")
        sys.exit(1)

    # Check for ngrok auth token (optional but recommended for longer sessions)
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if auth_token:
        conf.get_default().auth_token = auth_token
        print("[ngrok] Auth token configured")
    else:
        print("[ngrok] No auth token (sessions limited to 2 hours)")
        print("[ngrok] Get free token at: https://dashboard.ngrok.com/get-started/your-authtoken")
        print()

    # Start ngrok tunnel
    port = int(os.getenv("API_PORT", 5001))

    print(f"[ngrok] Starting tunnel to port {port}...")
    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url

    print()
    print("=" * 60)
    print("FREIGHT AGENT API - PUBLIC URL")
    print("=" * 60)
    print()
    print(f"  PUBLIC URL: {public_url}")
    print()
    print("  Share this with your teammate!")
    print()
    print("  Endpoints:")
    print(f"    GET  {public_url}/health")
    print(f"    POST {public_url}/api/quote")
    print(f"    POST {public_url}/api/quote/file")
    print(f"    GET  {public_url}/api/emails")
    print(f"    GET  {public_url}/api/rate-sheets")
    print()
    print("=" * 60)
    print()

    # Now start the Flask app
    # Import here to ensure ngrok starts first
    from api import app

    print("[Flask] Starting API server...")
    print("[Flask] Press Ctrl+C to stop")
    print()

    try:
        # Run Flask (this blocks)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[ngrok] Shutting down tunnel...")
        ngrok.disconnect(public_url)
        ngrok.kill()
        print("[ngrok] Done!")


if __name__ == "__main__":
    main()
