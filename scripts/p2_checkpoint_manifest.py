"""Create a metadata-only SHA-256 manifest for rigorous P2 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


EXPECTED = (
    "UNet3D",
    "NanoMambaUNet",
    "Ablation_NoMamba_UNet",
    "Ablation_HalfMamba_UNet",
    "AttentionUNet",
    "SegResNet16",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state_dict", {})
        record.update(
            {
                "model_name": checkpoint.get("model_name"),
                "epoch": checkpoint.get("epoch"),
                "val_mean_dice": checkpoint.get("val_mean_dice"),
                "config": json_safe(checkpoint.get("config", {})),
                "state_dict_tensors": len(state),
                "state_dict_numel": int(
                    sum(tensor.numel() for tensor in state.values() if hasattr(tensor, "numel"))
                ),
            }
        )
    except Exception as exc:  # The hash remains useful if metadata loading fails.
        record["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-checkpoint-set-confirmed", action="store_true")
    args = parser.parse_args()

    records = []
    missing = []
    for model in EXPECTED:
        path = args.checkpoint_dir / f"best_{model}.pth"
        if path.is_file():
            records.append(inspect_checkpoint(path))
        else:
            missing.append(path.name)
    payload = {
        "schema_version": 1,
        "historical_checkpoint_set_confirmed": args.historical_checkpoint_set_confirmed,
        "all_expected_present": not missing,
        "missing": missing,
        "checkpoints": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
