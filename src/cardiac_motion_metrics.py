"""Pure numerical utilities for segmentation-derived cardiac motion analysis.

The functions in this module deliberately avoid PyTorch, MONAI, nibabel, and
Matplotlib.  This keeps the scientific definitions independently testable and
allows the full cine inference script to fail early on malformed metadata or
non-physiological numerical inputs.

Frame numbers exposed by this module are one-based, matching ACDC ``Info.cfg``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


FOREGROUND_CLASSES = (1, 2, 3)
CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def parse_info_cfg_text(text: str) -> Dict[str, object]:
    """Parse and validate the small ACDC ``Info.cfg`` record.

    The dataset uses one-based ED/ES frame numbers.  ``NbFrame`` is required so
    that accidental selection of a spatial NIfTI axis as time is rejected.
    Other fields are preserved as strings for provenance and subgroup summaries.
    """

    raw: Dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Malformed Info.cfg line {line_number}: {line!r}")
        key, value = stripped.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError(f"Empty Info.cfg key/value on line {line_number}")
        if key in raw:
            raise ValueError(f"Duplicate Info.cfg field: {key}")
        raw[key] = value

    required = {"ED", "ES", "NbFrame"}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError("Info.cfg is missing: " + ", ".join(missing))

    parsed: Dict[str, object] = dict(raw)
    for key in ("ED", "ES", "NbFrame"):
        try:
            parsed[key] = int(raw[key])
        except ValueError as exc:
            raise ValueError(f"Info.cfg {key} is not an integer: {raw[key]!r}") from exc

    ed = int(parsed["ED"])
    es = int(parsed["ES"])
    num_frames = int(parsed["NbFrame"])
    if num_frames < 3:
        raise ValueError("A cine sequence must contain at least three frames")
    if not 1 <= ed <= num_frames or not 1 <= es <= num_frames:
        raise ValueError(
            f"ED/ES must be inside 1..{num_frames}; received ED={ed}, ES={es}"
        )
    if ed == es:
        raise ValueError("ED and ES must identify different frames")
    return parsed


def circular_moving_average(values: Sequence[float], window: int = 3) -> np.ndarray:
    """Return an odd-width circular moving average for a periodic cine curve."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 3:
        raise ValueError("values must be a one-dimensional sequence of length >= 3")
    if not np.isfinite(array).all():
        raise ValueError("values contain NaN or infinity")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window > array.size:
        raise ValueError("window cannot exceed the number of frames")
    radius = window // 2
    result = np.zeros_like(array)
    for offset in range(-radius, radius + 1):
        result += np.roll(array, offset)
    return result / float(window)


def circular_frame_distance(first: int, second: int, num_frames: int) -> int:
    """Return the shortest distance between one-based frames on a cine cycle."""

    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if not 1 <= first <= num_frames or not 1 <= second <= num_frames:
        raise ValueError("frame numbers must be inside the cine sequence")
    direct = abs(first - second)
    return int(min(direct, num_frames - direct))


def dice_by_class(
    prediction: np.ndarray,
    reference: np.ndarray,
    classes: Iterable[int] = FOREGROUND_CLASSES,
) -> Dict[str, float]:
    """Compute hard-mask Dice with the historical empty/empty score of one."""

    pred = np.asarray(prediction)
    true = np.asarray(reference)
    if pred.shape != true.shape:
        raise ValueError(f"Dice shape mismatch: {pred.shape} versus {true.shape}")
    rows: Dict[str, float] = {}
    scores = []
    for class_id in classes:
        pred_class = pred == class_id
        true_class = true == class_id
        denominator = int(pred_class.sum()) + int(true_class.sum())
        score = (
            1.0
            if denominator == 0
            else float(2.0 * np.logical_and(pred_class, true_class).sum() / denominator)
        )
        rows[f"dice_{CLASS_NAMES.get(class_id, str(class_id))}"] = score
        scores.append(score)
    rows["mean_dice"] = float(np.mean(scores))
    return rows


def _world_centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return np.full(3, np.nan, dtype=np.float64)
    index_centroid = coordinates.mean(axis=0, dtype=np.float64)
    homogeneous = np.append(index_centroid, 1.0)
    return np.asarray(affine, dtype=np.float64).dot(homogeneous)[:3]


def segmentation_frame_metrics(
    mask: np.ndarray,
    affine: np.ndarray,
    voxel_volume_ml: float,
) -> Dict[str, float]:
    """Measure physical class volumes and coarse global motion surrogates.

    ``mean_myo_radius_mm`` is the mean Euclidean distance of myocardial voxel
    centres from the LV-cavity centroid.  It is a global segmentation-derived
    radial surrogate, not wall thickness and not a dense displacement field.
    """

    labels = np.asarray(mask)
    matrix = np.asarray(affine, dtype=np.float64)
    if labels.ndim != 3:
        raise ValueError("mask must be three-dimensional")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not np.isfinite(voxel_volume_ml) or voxel_volume_ml <= 0.0:
        raise ValueError("voxel_volume_ml must be finite and positive")

    result: Dict[str, float] = {}
    for class_id, class_name in CLASS_NAMES.items():
        class_mask = labels == class_id
        result[f"{class_name.lower()}_volume_ml"] = float(
            class_mask.sum() * voxel_volume_ml
        )
        centroid = _world_centroid(class_mask, matrix)
        for axis, value in zip("xyz", centroid):
            result[f"{class_name.lower()}_centroid_{axis}_mm"] = float(value)

    lv_centroid = np.array(
        [result[f"lv_centroid_{axis}_mm"] for axis in "xyz"], dtype=np.float64
    )
    myo_indices = np.argwhere(labels == 2)
    if myo_indices.size == 0 or not np.isfinite(lv_centroid).all():
        result["mean_myo_radius_mm"] = float("nan")
    else:
        homogeneous = np.concatenate(
            [myo_indices.astype(np.float64), np.ones((myo_indices.shape[0], 1))],
            axis=1,
        )
        world = homogeneous.dot(matrix.T)[:, :3]
        result["mean_myo_radius_mm"] = float(
            np.linalg.norm(world - lv_centroid[None, :], axis=1).mean()
        )
    return result


def analyse_lv_curve(
    lv_volumes_ml: Sequence[float],
    reference_ed_frame: int,
    reference_es_frame: int,
    smoothing_window: int = 3,
) -> Dict[str, object]:
    """Derive phase, functional, and smoothness metrics from a full LV curve."""

    raw = np.asarray(lv_volumes_ml, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 3:
        raise ValueError("LV curve must be one-dimensional with at least 3 frames")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("LV curve must contain finite non-negative volumes")
    if np.ptp(raw) <= 0.0:
        raise ValueError("LV curve has no measurable temporal variation")

    smoothed = circular_moving_average(raw, smoothing_window)
    predicted_ed = int(np.argmax(smoothed)) + 1
    predicted_es = int(np.argmin(smoothed)) + 1
    edv = float(smoothed[predicted_ed - 1])
    esv = float(smoothed[predicted_es - 1])
    if edv <= 0.0 or esv > edv:
        raise ValueError("Derived EDV/ESV are not physiologically ordered")
    stroke_volume = edv - esv
    ejection_fraction = 100.0 * stroke_volume / edv

    amplitude = float(np.ptp(smoothed))
    second_difference = np.roll(smoothed, -1) - 2.0 * smoothed + np.roll(smoothed, 1)
    smoothness = float(np.mean(np.abs(second_difference)) / max(amplitude, 1e-12))
    circular_tv = float(
        np.mean(np.abs(np.roll(smoothed, -1) - smoothed))
        / max(amplitude, 1e-12)
    )

    return {
        "predicted_ed_frame": predicted_ed,
        "predicted_es_frame": predicted_es,
        "ed_frame_error": circular_frame_distance(
            predicted_ed, reference_ed_frame, raw.size
        ),
        "es_frame_error": circular_frame_distance(
            predicted_es, reference_es_frame, raw.size
        ),
        "curve_edv_ml": edv,
        "curve_esv_ml": esv,
        "curve_sv_ml": stroke_volume,
        "curve_ef_percent": ejection_fraction,
        "curve_peak_to_peak_ml": amplitude,
        "normalized_second_difference": smoothness,
        "normalized_circular_total_variation": circular_tv,
        "smoothed_lv_volumes_ml": [float(value) for value in smoothed],
    }


def functional_indices(
    edv_ml: float,
    esv_ml: float,
    require_physiological_order: bool = True,
) -> Dict[str, float]:
    """Calculate stroke volume and ejection fraction from endpoint volumes.

    Reference masks must use the default strict ordering.  For model outputs,
    callers can retain an observed ``ESV > EDV`` failure by setting
    ``require_physiological_order=False`` rather than crashing or silently
    swapping the annotated phases.
    """

    for name, value in (("EDV", edv_ml), ("ESV", esv_ml)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if edv_ml <= 0.0:
        raise ValueError("EDV must be positive")
    if require_physiological_order and esv_ml > edv_ml:
        raise ValueError("ESV cannot exceed EDV")
    stroke_volume = float(edv_ml - esv_ml)
    return {
        "sv_ml": stroke_volume,
        "ef_percent": float(100.0 * stroke_volume / edv_ml),
    }


def finite_mean(values: Sequence[float]) -> float:
    """Return the mean of finite values and reject an entirely missing metric."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("metric contains no finite values")
    return float(finite.mean())


def pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Pearson r for finite paired observations, rejecting constants."""

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Pearson inputs must be paired one-dimensional arrays")
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        raise ValueError("Pearson correlation needs at least two finite pairs")
    x, y = x[finite], y[finite]
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant vector")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_mean_ci(
    values: Sequence[float],
    replicates: int = 10_000,
    seed: int = 20_260_821,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Return a deterministic patient-level percentile bootstrap mean CI."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("bootstrap CI needs at least two finite observations")
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence": float(confidence),
        "replicates": int(replicates),
        "seed": int(seed),
        "n": int(array.size),
    }


def require_columns(rows: Sequence[Mapping[str, object]], columns: Iterable[str]) -> None:
    """Fail closed when an aggregate is built from incomplete patient rows."""

    required = set(columns)
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"patient row {index} is missing: {', '.join(missing)}")
