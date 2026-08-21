"""Restore missing ACDC full-cine files and prove patient identity before use.

The local project may contain complete ED/ES endpoint images while a subset of
the original ``patientNNN_4d.nii.gz`` files is absent.  This utility first
searches the local ACDC tree for misplaced copies.  With ``--allow-download``,
it can then retrieve only the known-missing files from a pinned public mirror.

No recovered file is installed until all of these checks pass:

* exact byte length and SHA-256 for a pinned download;
* four-dimensional NIfTI shape and ``Info.cfg`` ``NbFrame``;
* spatial shape and affine agreement with the local ED and ES images; and
* normalized ED/ES image agreement with the corresponding cine frames.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PINNED_REPOSITORY = "msepulvedagodoy/acdc"
PINNED_REVISION = "067262d5b40f9c976f7139c13416ace5a3314f42"
PINNED_MIRRORS: Tuple[Tuple[str, str], ...] = (
    (PINNED_REPOSITORY, PINNED_REVISION),
    ("MedOtter/ACDC", "aa51609e58f92f9b657dcca7babebd172b9b36b6"),
)
PINNED_CINE_FILES: Dict[str, Dict[str, object]] = {
    "patient002": {
        "bytes": 17_559_867,
        "sha256": "b074ae28371ad30c21805e93a2c15eaa9c1fe92ae6c08789ef36b49ab8e76446",
    },
    "patient042": {
        "bytes": 18_791_701,
        "sha256": "b47ab2c97d8035fe6fdf55d39542cc0a19537a30cd8f13af393e6703b58c353c",
    },
    "patient049": {
        "bytes": 14_130_645,
        "sha256": "492ae35c12644e7437ad191c5b0a9a9f80ac13b9120d031cdba4bda53d430fcc",
    },
    "patient066": {
        "bytes": 16_715_374,
        "sha256": "f00752148ebc8bf449600c80021cca8c2db90ed2413a932e5eabe9180ebd87a0",
    },
    "patient071": {
        "bytes": 11_945_021,
        "sha256": "2ef02fe147b81746cd50a29df91e7e798e8087963898ec93d56e1f35ae848d43",
    },
    "patient073": {
        "bytes": 16_865_597,
        "sha256": "7dc09cdfacba9f26e93cc3c82e42bb15ef50266c9947b9013e6731e802658832",
    },
}


def load_cine_analysis_module() -> Any:
    path = Path(__file__).with_name("23_spatiotemporal_cine_analysis.py")
    spec = importlib.util.spec_from_file_location("cine_analysis_for_recovery", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import cine analysis helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_url(patient_id: str, mirror_index: int = 0) -> str:
    repository, revision = PINNED_MIRRORS[mirror_index]
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/training/"
        f"{patient_id}/{patient_id}_4d.nii.gz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_file_contract(patient_id: str, mirror_index: int = 0) -> Dict[str, object]:
    if patient_id not in PINNED_CINE_FILES:
        raise KeyError(
            f"No pinned recovery file is approved for {patient_id}; "
            "download the official ACDC archive instead of guessing a source"
        )
    contract = dict(PINNED_CINE_FILES[patient_id])
    repository, revision = PINNED_MIRRORS[mirror_index]
    contract.update(
        {
            "repository": repository,
            "revision": revision,
            "url": source_url(patient_id, mirror_index),
        }
    )
    return contract


def download_pinned_file(
    patient_id: str,
    destination: Path,
    mirror_index: int = 0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Dict[str, object]:
    contract = pinned_file_contract(patient_id, mirror_index)
    expected_bytes = int(contract["bytes"])
    expected_sha256 = str(contract["sha256"])
    request = urllib.request.Request(
        str(contract["url"]),
        headers={
            "User-Agent": "NanoMamba-P2-cine-recovery/1.0",
            "Accept": "application/octet-stream",
        },
    )
    digest = hashlib.sha256()
    received = 0
    next_progress = 10
    with opener(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            percent = int(100 * received / expected_bytes)
            if percent >= next_progress:
                print(f"    download {min(percent, 100):3d}% ({received / 2**20:.1f} MiB)")
                next_progress += 10

    observed_sha256 = digest.hexdigest()
    if received != expected_bytes:
        raise ValueError(
            f"Downloaded byte count mismatch for {patient_id}: "
            f"{received} != {expected_bytes}"
        )
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Downloaded SHA-256 mismatch for {patient_id}: {observed_sha256}"
        )
    return {
        **contract,
        "observed_bytes": received,
        "observed_sha256": observed_sha256,
    }


def validate_cine_identity(
    cine_path: Path,
    patient_dir: Path,
    analysis: Any,
    nib: Any,
    endpoint_mae_tolerance: float,
) -> Dict[str, object]:
    info_path = patient_dir / "Info.cfg"
    info = analysis.parse_info_cfg_text(info_path.read_text(encoding="utf-8-sig"))
    endpoints = analysis.discover_endpoint_pairs(patient_dir)
    cine_image = nib.load(str(cine_path))
    cine_shape = tuple(int(value) for value in cine_image.shape)
    expected_frames = int(info["NbFrame"])
    if len(cine_shape) != 4 or cine_shape[-1] != expected_frames:
        raise ValueError(
            f"{patient_dir.name} recovered cine shape={cine_shape}, expected "
            f"(X, Y, Z, {expected_frames})"
        )

    native_shape = cine_shape[:3]
    cine_affine = np.asarray(cine_image.affine, dtype=np.float64)
    endpoint_mae: Dict[str, float] = {}
    for phase in ("ED", "ES"):
        frame = int(info[phase])
        if frame not in endpoints:
            raise ValueError(f"{patient_dir.name} is missing the {phase} endpoint pair")
        endpoint_image = nib.load(str(endpoints[frame]["image"]))
        endpoint_shape = tuple(int(value) for value in endpoint_image.shape)
        if endpoint_shape != native_shape:
            raise ValueError(
                f"{patient_dir.name} {phase} shape={endpoint_shape}, "
                f"cine spatial shape={native_shape}"
            )
        if not np.allclose(
            np.asarray(endpoint_image.affine, dtype=np.float64),
            cine_affine,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(f"{patient_dir.name} {phase} affine differs from recovered cine")
        cine_frame = np.asarray(cine_image.dataobj[..., frame - 1], dtype=np.float32)
        endpoint_values = np.asarray(endpoint_image.dataobj, dtype=np.float32)
        mae = analysis.normalized_input_mae(cine_frame, endpoint_values)
        endpoint_mae[phase] = mae
        if mae > endpoint_mae_tolerance:
            raise ValueError(
                f"{patient_dir.name} {phase} normalized MAE={mae:.8f} exceeds "
                f"{endpoint_mae_tolerance}; recovered cine does not match local endpoints"
            )

    return {
        "cine_shape": list(cine_shape),
        "reference_ed_frame": int(info["ED"]),
        "reference_es_frame": int(info["ES"]),
        "endpoint_normalized_mae": endpoint_mae,
    }


def local_cine_candidates(search_roots: Iterable[Path], patient_id: str) -> List[Path]:
    candidates: List[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    continue
            except OSError:
                continue
            relative_text = str(path.relative_to(root)).lower()
            if patient_id.lower() not in relative_text or "4d" not in relative_text:
                continue
            if (
                ".download." in path.name.lower()
                or ".recovery." in path.name.lower()
                or path.name.lower().endswith(".partial")
            ):
                continue
            parts = [part.lower() for part in path.relative_to(root).parts]
            if any(part.endswith((".nii", ".nii.gz")) for part in parts):
                candidates.append(path.resolve())
    return sorted(set(candidates), key=lambda path: str(path).lower())


def choose_valid_local_candidate(
    candidates: Sequence[Path],
    patient_dir: Path,
    target: Path,
    analysis: Any,
    nib: Any,
    endpoint_mae_tolerance: float,
) -> Optional[Tuple[Path, Dict[str, object]]]:
    valid: List[Tuple[Path, Dict[str, object], str]] = []
    for candidate in candidates:
        if candidate == target.resolve():
            continue
        try:
            validation = validate_cine_identity(
                candidate,
                patient_dir,
                analysis,
                nib,
                endpoint_mae_tolerance,
            )
            valid.append((candidate, validation, sha256(candidate)))
        except Exception:
            continue
    if not valid:
        return None
    hashes = {item[2] for item in valid}
    if len(hashes) != 1:
        raise ValueError(
            f"Multiple different local full cines match {patient_dir.name}: "
            + ", ".join(str(item[0]) for item in valid)
        )
    return valid[0][0], valid[0][1]


def install_from_path(
    source: Path,
    target: Path,
    patient_dir: Path,
    analysis: Any,
    nib: Any,
    endpoint_mae_tolerance: float,
) -> Dict[str, object]:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {target}")
    temporary = target.with_name(f"{target.stem}.recovery.nii.gz")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copy2(source, temporary)
        validation = validate_cine_identity(
            temporary,
            patient_dir,
            analysis,
            nib,
            endpoint_mae_tolerance,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return validation


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\AI_FYP"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--endpoint-mae-tolerance", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = (
        args.data_dir
        or project_root / "Data" / "ACDC" / "database" / "training"
    ).resolve()
    split_path = (
        args.split
        or repo_root / "evidence" / "rigorous_patient_split" / "patient_split_seed42.json"
    ).resolve()
    if args.endpoint_mae_tolerance < 0.0:
        raise ValueError("--endpoint-mae-tolerance must be non-negative")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ACDC training directory is missing: {data_dir}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Patient split is missing: {split_path}")

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for cine recovery validation") from exc
    analysis = load_cine_analysis_module()
    split = analysis.load_split(split_path)
    patient_ids = sorted(split["val_patients"], key=analysis.natural_sort_key)
    search_roots = [path.resolve() for path in args.search_root]
    default_search_root = data_dir.parent.parent
    if default_search_root not in search_roots:
        search_roots.append(default_search_root)

    already_present: List[Dict[str, object]] = []
    missing: List[str] = []
    for patient_id in patient_ids:
        patient_dir = data_dir / patient_id
        info = analysis.parse_info_cfg_text(
            (patient_dir / "Info.cfg").read_text(encoding="utf-8-sig")
        )
        try:
            cine_path = analysis.discover_cine_nifti(
                patient_dir, int(info["NbFrame"]), nib
            )
            validation = validate_cine_identity(
                cine_path,
                patient_dir,
                analysis,
                nib,
                args.endpoint_mae_tolerance,
            )
            already_present.append(
                {
                    "patient_id": patient_id,
                    "path": str(cine_path),
                    "sha256": sha256(cine_path),
                    **validation,
                }
            )
        except FileNotFoundError:
            missing.append(patient_id)

    print(
        f"ACDC cine inventory: {len(already_present)}/{len(patient_ids)} validation "
        f"patients complete; {len(missing)} require recovery"
    )
    if missing:
        print("Missing: " + ", ".join(missing))

    recovered: List[Dict[str, object]] = []
    for index, patient_id in enumerate(missing, start=1):
        patient_dir = data_dir / patient_id
        target = patient_dir / f"{patient_id}_4d.nii.gz"
        print(f"[{index}/{len(missing)}] Recovering {patient_id}")
        local = choose_valid_local_candidate(
            local_cine_candidates(search_roots, patient_id),
            patient_dir,
            target,
            analysis,
            nib,
            args.endpoint_mae_tolerance,
        )
        source_record: Dict[str, object]
        if local is not None:
            source_path, _ = local
            print(f"    using validated local copy: {source_path}")
            validation = install_from_path(
                source_path,
                target,
                patient_dir,
                analysis,
                nib,
                args.endpoint_mae_tolerance,
            )
            source_record = {
                "kind": "local_validated_copy",
                "path": str(source_path),
                "sha256": sha256(source_path),
            }
        else:
            if not args.allow_download:
                raise FileNotFoundError(
                    f"No matching local full cine found for {patient_id}. Rerun with "
                    "--allow-download or restore it from the official ACDC archive."
                )
            contract = pinned_file_contract(patient_id)
            print(
                f"    downloading pinned {int(contract['bytes']) / 2**20:.1f} MiB file"
            )
            temporary_download = target.with_name(
                f"{patient_id}_4d.download.nii.gz"
            )
            if temporary_download.exists():
                temporary_download.unlink()
            try:
                download_record: Optional[Dict[str, object]] = None
                last_error: Optional[Exception] = None
                attempts = (0, 0, 1)
                for attempt, mirror_index in enumerate(attempts, start=1):
                    try:
                        download_record = download_pinned_file(
                            patient_id,
                            temporary_download,
                            mirror_index=mirror_index,
                        )
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if temporary_download.exists():
                            temporary_download.unlink()
                        if attempt < len(attempts):
                            next_repository = PINNED_MIRRORS[attempts[attempt]][0]
                            print(
                                f"    attempt {attempt} failed; retrying via "
                                f"{next_repository}: {exc}"
                            )
                            time.sleep(2 * attempt)
                if last_error is not None or download_record is None:
                    raise RuntimeError(
                        f"Unable to download {patient_id} after 3 attempts"
                    ) from last_error
                validation = install_from_path(
                    temporary_download,
                    target,
                    patient_dir,
                    analysis,
                    nib,
                    args.endpoint_mae_tolerance,
                )
                source_record = {
                    "kind": "pinned_public_mirror",
                    "repository": download_record["repository"],
                    "revision": download_record["revision"],
                    "url": download_record["url"],
                    "expected_sha256": download_record["sha256"],
                    "expected_bytes": download_record["bytes"],
                }
            finally:
                if temporary_download.exists():
                    temporary_download.unlink()

        installed_sha256 = sha256(target)
        print(
            f"    validated and installed: ED MAE="
            f"{float(validation['endpoint_normalized_mae']['ED']):.8f}, ES MAE="
            f"{float(validation['endpoint_normalized_mae']['ES']):.8f}"
        )
        recovered.append(
            {
                "patient_id": patient_id,
                "destination": str(target),
                "bytes": target.stat().st_size,
                "sha256": installed_sha256,
                "source": source_record,
                "validation": validation,
            }
        )

    # Re-run the same all-patient resolver used by the final experiment.
    final_inputs = analysis.preflight_patient_inputs(data_dir, patient_ids, nib)
    if len(final_inputs) != len(patient_ids):
        raise RuntimeError("Recovery finished but all-patient preflight is incomplete")

    output_dir = project_root / "experiment_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = output_dir / f"acdc_cine_recovery_{timestamp}.json"
    manifest = {
        "schema_version": 1,
        "status": "all_validation_cines_verified",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_patients": len(patient_ids),
        "already_present": already_present,
        "recovered": recovered,
        "endpoint_mae_tolerance": args.endpoint_mae_tolerance,
        "pinned_download_sources": [
            {"repository": repository, "revision": revision}
            for repository, revision in PINNED_MIRRORS
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nACDC CINE RECOVERY: PASS")
    print(f"All {len(patient_ids)} validation cines passed identity checks")
    print(f"Recovery manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
