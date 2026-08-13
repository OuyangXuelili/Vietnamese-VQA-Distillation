"""Launch the demo with Gradio's official public share link.

Older versions of this helper started a second ``localhost.run`` SSH tunnel.
That tunnel served the static page but did not reliably proxy Gradio's
heartbeat/stream endpoints, so uploads and submissions could fail with
``Failed to fetch``.  Gradio's own ``--share`` link proxies those endpoints
correctly and is all the Colab demo needs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Colab VQA demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Run locally without creating a temporary public Gradio URL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-u",
        "app.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_share:
        command.append("--share")

    print("Khởi động demo Gradio; giữ cell này chạy để đường dẫn tiếp tục hoạt động.", flush=True)
    try:
        raise SystemExit(subprocess.call(command, cwd=ROOT))
    except KeyboardInterrupt:
        print("\nĐã dừng demo.", flush=True)


if __name__ == "__main__":
    main()
