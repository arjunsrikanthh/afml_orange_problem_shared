#!/usr/bin/env python3
"""Evaluate per-parameter-tensor affine weight restoration on development only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Keep both `python scripts/...py` and `python -m scripts...` entry points valid.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.translation_noisy_baseline import EXPECTED_PARAMETER_COUNT, TranslationTransformer
from scripts.weight_baselines import fit_coefficients, make_assignment


DATA = ROOT / "data" / "23_Team_Toxic"
DEFAULT_OUTPUT = ROOT / "results" / "baselines" / "weight_family_v1.json"
GLOBAL_AFFINE_REFERENCE = 0.4982864015
DEFAULT_MIN_FITTING = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_map() -> list[dict[str, Any]]:
    """Return the fixed named-parameter offset map from the supplied class."""
    model = TranslationTransformer()
    result: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        result.append({
            "name": name,
            "shape": list(parameter.shape),
            "count": count,
            "start": offset,
            "stop": offset + count,
        })
        offset += count
    if offset != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"parameter map has {offset} values, expected {EXPECTED_PARAMETER_COUNT}")
    if result[-1]["stop"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("parameter map is not contiguous")
    return result


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered = x - x_mean
    variance = float(np.mean(centered * centered))
    slope = float(np.mean(centered * (y - y_mean)) / variance) if variance else 1.0
    return slope, float(y_mean - slope * x_mean)


def _rmse(prediction: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        raise ValueError("development slice is empty")
    error = np.asarray(prediction[mask], dtype=np.float64) - np.asarray(clean[mask], dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def calibrate(
    noisy: np.ndarray,
    clean: np.ndarray,
    labels: np.ndarray,
    families: list[dict[str, Any]],
    *,
    min_fitting: int = DEFAULT_MIN_FITTING,
) -> dict[str, Any]:
    """Fit family coefficients on fitting positions and score only development."""
    global_coefficients = fit_coefficients(noisy, clean, labels)
    global_slope = global_coefficients["affine_prediction_slope"]
    global_intercept = global_coefficients["affine_prediction_intercept"]
    global_prediction = global_slope * noisy + global_intercept
    development = labels == 1
    family_prediction = np.asarray(global_prediction, dtype=np.float64).copy()
    reports: list[dict[str, Any]] = []
    half = noisy.size // 2

    for family in families:
        start, stop = int(family["start"]), int(family["stop"])
        offsets = np.concatenate((np.arange(start, stop), np.arange(half + start, half + stop)))
        fitting = labels[offsets] == 0
        development_family = labels[offsets] == 1
        fitting_count = int(np.count_nonzero(fitting))
        development_count = int(np.count_nonzero(development_family))
        reason = None
        if fitting_count < min_fitting:
            reason = "insufficient_fitting_count"
            slope, intercept = global_slope, global_intercept
        else:
            slope, intercept = _fit_affine(noisy[offsets][fitting], clean[offsets][fitting])
            if not np.isfinite(slope) or not np.isfinite(intercept):
                reason = "nonfinite_family_coefficients"
                slope, intercept = global_slope, global_intercept
            elif float(np.var(noisy[offsets][fitting], dtype=np.float64)) == 0.0:
                reason = "zero_fitting_variance"
                slope, intercept = global_slope, global_intercept

        family_prediction[offsets] = slope * noisy[offsets] + intercept
        family_development_mask = np.isin(np.arange(noisy.size), offsets) & development
        reports.append({
            "name": family["name"],
            "shape": family["shape"],
            "parameter_count": family["count"],
            "offset_start": start,
            "offset_stop": stop,
            "fitting_count": fitting_count,
            "development_count": development_count,
            "fallback": reason is not None,
            "fallback_reason": reason,
            "slope": float(slope),
            "intercept": float(intercept),
            "development_rmse_original_units": (
                _rmse(family_prediction, clean, family_development_mask)
                if development_count else None
            ),
        })

    return {
        "global_coefficients": global_coefficients,
        "global_development_count": int(np.count_nonzero(development)),
        "global_affine_development_rmse_original_units": _rmse(global_prediction, clean, development),
        "family_affine_development_rmse_original_units": _rmse(family_prediction, clean, development),
        "families": reports,
        "family_prediction": family_prediction,
    }


def run(
    *,
    data_dir: Path = DATA,
    output: Path = DEFAULT_OUTPUT,
    seed: int = 2301,
    block_size: int = 4096,
    guard: int = 32,
    min_fitting: int = DEFAULT_MIN_FITTING,
) -> dict[str, Any]:
    clean_path = data_dir / "train_weights_clean.npy"
    noisy_path = data_dir / "train_weights_noisy.npy"
    clean = np.load(clean_path, mmap_mode="r")
    noisy = np.load(noisy_path, mmap_mode="r")
    if clean.shape != noisy.shape or clean.ndim != 1 or clean.size != 2 * EXPECTED_PARAMETER_COUNT:
        raise ValueError("training arrays must be aligned one-dimensional paired halves")
    if clean.dtype != np.float32 or noisy.dtype != np.float32 or not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError("training arrays must be finite float32 values")
    labels = make_assignment(clean.size, half_length=EXPECTED_PARAMETER_COUNT, seed=seed, block_size=block_size, guard=guard)
    families = parameter_map()
    report = calibrate(noisy, clean, labels, families, min_fitting=min_fitting)
    split_spec = {
        "rule": "sha256(weight-v1|seed|group_id) byte thresholds 0.60/0.80; same offset group in both halves",
        "seed": seed,
        "block_size": block_size,
        "guard": guard,
        "half_length": EXPECTED_PARAMETER_COUNT,
        "total_length": int(clean.size),
    }
    split_hash = hashlib.sha256(json.dumps(split_spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    global_rmse = report["global_affine_development_rmse_original_units"]
    family_rmse = report["family_affine_development_rmse_original_units"]
    result = {
        "schema_version": 1,
        "experiment": "BASE-FAMILY-001",
        "team": 23,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "model": {
            "class": "TranslationTransformer",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "parameter_family_count": len(families),
            "map_derivation": "supplied TranslationTransformer.named_parameters() order",
            "equal_half_layout": "same parameter offsets in both equal-length halves; explicit unverified-layout hypothesis",
        },
        "split": {
            **split_spec,
            "spec_sha256": split_hash,
            "counts": {
                "fitting": int(np.count_nonzero(labels == 0)),
                "development": int(np.count_nonzero(labels == 1)),
                "audit": int(np.count_nonzero(labels == 2)),
                "guard": int(np.count_nonzero(labels == -1)),
            },
        },
        "input_sha256": {"train_weights_clean.npy": sha256_file(clean_path), "train_weights_noisy.npy": sha256_file(noisy_path)},
        "minimum_fitting_per_family": min_fitting,
        "global_affine_reference_rmse": GLOBAL_AFFINE_REFERENCE,
        "global_affine_development_rmse_original_units": global_rmse,
        "family_affine_development_rmse_original_units": family_rmse,
        "delta_family_minus_global": family_rmse - global_rmse,
        "failure_criteria": {
            "criterion": "family aggregate development RMSE must be strictly below the frozen global-affine incumbent 0.4982864015",
            "reference": GLOBAL_AFFINE_REFERENCE,
            "observed_family_rmse_below_reference": family_rmse < GLOBAL_AFFINE_REFERENCE,
            "promotion_status": "pass" if family_rmse < GLOBAL_AFFINE_REFERENCE else "fail",
        },
        "families": report["families"],
        "audit_metrics": None,
        "limitations": [
            "Only fitting positions determine family and global coefficients; development is evaluation-only.",
            "Audit clean values and hidden test labels are never indexed for metrics.",
            "Equal-half correspondence and tensor boundaries are an explicit unverified-layout hypothesis.",
            "A lower scalar RMSE does not establish improved translation quality or test performance.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-fitting", type=int, default=DEFAULT_MIN_FITTING)
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output=args.output, min_fitting=args.min_fitting)
    print(json.dumps({key: result[key] for key in ("global_affine_development_rmse_original_units", "family_affine_development_rmse_original_units", "failure_criteria")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
