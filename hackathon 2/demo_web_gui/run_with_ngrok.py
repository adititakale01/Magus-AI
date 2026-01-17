from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

import ngrok
import uvicorn


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_port(host: str, port: int, timeout_s: float = 10.0) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if _port_is_open(host, port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the API server and expose it via ngrok (Python SDK).")
    parser.add_argument("--host", default=os.getenv("API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Do not start uvicorn (assume the API server is already running on host:port).",
    )
    args = parser.parse_args(argv)

    bind_host = str(args.host)
    connect_host = bind_host
    if connect_host in {"0.0.0.0", "::"}:
        connect_host = "127.0.0.1"

    authtoken = (os.getenv("NGROK_AUTHTOKEN") or "").strip()
    if not authtoken:
        sys.stderr.write("Missing NGROK_AUTHTOKEN. Set it in your shell and re-run.\n")
        return 2

    ngrok.set_auth_token(authtoken)

    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None

    if not args.no_start and not _port_is_open(connect_host, args.port):
        config = uvicorn.Config(
            "api_server:app",
            host=bind_host,
            port=args.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        _wait_port(connect_host, args.port, timeout_s=15.0)

    try:
        listener = ngrok.forward(f"http://{connect_host}:{args.port}", "http")
    except Exception as e:
        sys.stderr.write(f"Failed to start ngrok tunnel: {e}\n")
        if server is not None:
            server.should_exit = True
        return 1

    public = str(getattr(listener, "url", "") or "").rstrip("/")
    print(f"ngrok URL: {public}")
    print(f"API base: {public}/api/v1")
    print(f"Docs: {public}/docs")

    try:
        while True:
            time.sleep(0.8)
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            listener.close()
        except Exception:
            pass
        try:
            ngrok.kill()
        except Exception:
            pass
        if server is not None:
            server.should_exit = True


if __name__ == "__main__":
    raise SystemExit(main())
