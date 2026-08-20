"""Capture a privacy-conscious software/hardware record for P2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def redact_local_identity(value: str | None) -> str | None:
    """Remove URL credentials and common user-home components from text."""
    if value is None:
        return None
    sanitized = re.sub(
        r"(?i)([a-z][a-z0-9+.-]*://)([^/@\s]+)@",
        r"\1<redacted>@",
        str(value),
    )
    sanitized = re.sub(
        r"(?i)(file:///+[a-z]:/users/)[^/]+",
        r"\1<redacted>",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)([a-z]:\\users\\)[^\\]+",
        r"\1<redacted>",
        sanitized,
    )
    sanitized = re.sub(
        r"(file:///+(?:home|users)/)[^/]+", r"\1<redacted>", sanitized
    )
    sanitized = re.sub(r"(/(?:home|Users)/)[^/]+", r"\1<redacted>", sanitized)
    return sanitized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-environment-confirmed", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    payload: Dict[str, Any] = {
        "schema_version": 2,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_experiment_environment_confirmed": args.historical_environment_confirmed,
        "notes": args.notes,
        "python": sys.version,
        "python_executable": redact_local_identity(sys.executable),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "conda_environment": {
            "name": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefix": redact_local_identity(os.environ.get("CONDA_PREFIX")),
        },
        "packages": {
            name: package_version(name) for name in ("numpy", "torch", "monai", "nibabel", "matplotlib")
        },
    }
    try:
        import torch

        payload["pytorch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except ImportError:
        payload["pytorch_runtime"] = None

    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu_query = []
    payload["nvidia_smi"] = gpu_query

    conda_history = Path(sys.prefix) / "conda-meta" / "history"
    if conda_history.is_file():
        history_bytes = conda_history.read_bytes()
        payload["conda_history"] = {
            "bytes": len(history_bytes),
            "sha256": hashlib.sha256(history_bytes).hexdigest(),
            "mtime_utc": datetime.fromtimestamp(
                conda_history.stat().st_mtime, timezone.utc
            ).isoformat(),
        }
    else:
        payload["conda_history"] = None

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        freeze = []
    payload["pip_freeze"] = [redact_local_identity(line) for line in freeze]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
