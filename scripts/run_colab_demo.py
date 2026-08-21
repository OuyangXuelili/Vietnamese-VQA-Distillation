"""Run the Gradio demo and expose it through a resilient Colab tunnel.

The Gradio server is intentionally kept separate from the public tunnel. A
temporary ``gradio.live`` outage can therefore no longer stop the application
or make the notebook look as if model loading failed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_URL_RE = re.compile(r"https://[a-z0-9-]+\.lhr\.life", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Colab VQA demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--reconnect-delay", type=int, default=5)
    return parser.parse_args()


def server_is_ready(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True only when Gradio's JSON config endpoint is healthy."""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/config", timeout=timeout
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def public_url_is_ready(url: str, timeout: int = 45) -> bool:
    """Verify that the tunnel serves Gradio JSON, not an HTML 404/504 page."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                f"{url.rstrip('/')}/config",
                headers={"User-Agent": "vi-vqa-colab-healthcheck/1.0"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and isinstance(payload.get("components"), list):
                    return True
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(1)
    return False


def wait_for_server(
    server: subprocess.Popen[str] | None,
    host: str,
    port: int,
    timeout: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_is_ready(host, port):
            print(f"GRADIO_SERVER_READY http://{host}:{port}", flush=True)
            return
        if server is not None and server.poll() is not None:
            break
        time.sleep(1)

    details = ""
    if log_path.exists():
        details = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    raise RuntimeError(
        "Gradio did not become ready within the startup timeout.\n" + details
    )


def start_server(host: str, port: int, log_path: Path) -> subprocess.Popen[str] | None:
    """Start Gradio when needed; reuse a healthy server after notebook reruns."""
    if server_is_ready(host, port):
        print(f"REUSING_GRADIO_SERVER http://{host}:{port}", flush=True)
        return None

    log_handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            sys.executable,
            "-u",
            "app.py",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def tunnel_command(port: int) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ConnectTimeout=20",
        "-T",
        "-R",
        f"80:localhost:{port}",
        "nokey@localhost.run",
    ]


def stream_tunnel(port: int) -> int:
    """Run one tunnel session and print a prominent, clickable demo URL."""
    tunnel = subprocess.Popen(
        tunnel_command(port),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert tunnel.stdout is not None

    announced: set[str] = set()
    try:
        for line in tunnel.stdout:
            print(line, end="", flush=True)
            for url in PUBLIC_URL_RE.findall(line):
                if url in announced:
                    continue
                announced.add(url)
                print(f"VERIFYING_PUBLIC_URL: {url}", flush=True)
                if not public_url_is_ready(url):
                    print(
                        "PUBLIC_URL_FAILED_HEALTHCHECK; reconnecting tunnel...",
                        flush=True,
                    )
                    return 1
                print("\n" + "=" * 72, flush=True)
                print(f"DEMO_URL: {url}", flush=True)
                print("Giữ cell này chạy để đường dẫn tiếp tục hoạt động.", flush=True)
                print("=" * 72 + "\n", flush=True)
    finally:
        if tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
    return tunnel.returncode or 0


def main() -> None:
    args = parse_args()
    log_path = Path(tempfile.gettempdir()) / "vi-vqa-gradio.log"
    server = start_server(args.host, args.port, log_path)

    try:
        wait_for_server(
            server,
            args.host,
            args.port,
            args.startup_timeout,
            log_path,
        )
        attempt = 0
        while True:
            if server is not None and server.poll() is not None:
                raise RuntimeError(
                    f"Gradio server stopped unexpectedly (exit {server.returncode})."
                )
            if server is None and not server_is_ready(args.host, args.port):
                raise RuntimeError("The reused Gradio server is no longer responding.")
            attempt += 1
            print(f"STARTING_PUBLIC_TUNNEL attempt={attempt}", flush=True)
            exit_code = stream_tunnel(args.port)
            print(
                f"Tunnel stopped (exit {exit_code}); reconnecting in "
                f"{args.reconnect_delay}s...",
                flush=True,
            )
            time.sleep(args.reconnect_delay)
    except KeyboardInterrupt:
        print("\nStopping Colab demo...", flush=True)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
