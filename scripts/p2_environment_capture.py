"""Capture a privacy-conscious software/hardware record for P2 evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-environment-confirmed", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "historical_experiment_environment_confirmed": args.historical_environment_confirmed,
        "notes": args.notes,
        "python": sys.version,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
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
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        freeze = []
    payload["pip_freeze"] = freeze

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
