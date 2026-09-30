#!/usr/bin/env python3
"""Leakage-aware scalar baselines for the paired Team 23 weight arrays.

The two equal-length halves are treated as corresponding offset blocks only for
the purpose of conservative split assignment. This script does not infer that
they are checkpoints or otherwise equivalent tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Literal

import numpy as np


SplitName = Literal["fitting", "development", "audit", "guard"]
SPLIT_NAMES: tuple[str, ...] = ("fitting", "development", "audit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_name(seed: int, group_id: int) -> str:
    value = hashlib.sha256(f"weight-v1|{seed}|{group_id}".encode("ascii")).digest()[0] / 256.0
    if value < 0.60:
        return "fitting"
    if value < 0.80:
        return "development"
    return "audit"


def make_assignment(
    total_length: int,
    *,
    half_length: int,
    seed: int = 2301,
    block_size: int = 4096,
    guard: int = 32,
) -> np.ndarray:
    """Return one label per scalar, with -1 for guard values.

    Assignment is based only on positions, lengths, and the fixed seed. The
    same group ID is used at the same offset in both halves.
    """
    if total_length != 2 * half_length or total_length <= 0:
        raise ValueError("expected two equal-length halves")
    if block_size <= 0 or guard < 0:
        raise ValueError("block_size must be positive and guard must be nonnegative")
    labels = np.full(total_length, -1, dtype=np.int8)
    codes = {"fitting": 0, "development": 1, "audit": 2}
    group_count = (half_length + block_size - 1) // block_size
    group_labels = [_group_name(seed, group_id) for group_id in range(group_count)]
    for half_start in (0, half_length):
        for group_id, name in enumerate(group_labels):
            start = half_start + group_id * block_size
            stop = min(half_start + (group_id + 1) * block_size, half_start + half_length)
            labels[start:stop] = codes[name]
    if guard:
        for group_id in range(1, group_count):
            if group_labels[group_id] != group_labels[group_id - 1]:
                boundary_offsets = (group_id * block_size, half_length + group_id * block_size)
                for boundary in boundary_offsets:
                    labels[max(0, boundary - guard) : min(total_length, boundary + guard)] = -1
    return labels


def validate_assignment(labels: np.ndarray) -> None:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [-1, 0, 1, 2]).all():
        raise AssertionError("assignment contains invalid or overlapping labels")
    if not np.all((labels >= -1) & (labels <= 2)):
        raise AssertionError("assignment has an invalid label")
    if np.any(labels == -1):
        # Guards are the only permitted unassigned positions.
        return


def fit_coefficients(noisy: np.ndarray, clean: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    fit = labels == 0
    if not np.any(fit):
        raise ValueError("fitting split is empty")
    x = np.asarray(noisy[fit], dtype=np.float64)
    y = np.asarray(clean[fit], dtype=np.float64)
    residual = y - x
    bias = float(np.mean(residual))
    x_centered = x - np.mean(x)
    variance = float(np.mean(x_centered * x_centered))
    slope = float(np.mean(x_centered * (y - np.mean(y))) / variance) if variance else 1.0
    intercept = float(np.mean(y) - slope * np.mean(x))
    return {
        "identity_slope": 1.0,
        "identity_intercept": 0.0,
        "bias_residual_mean": bias,
        "bias_prediction_slope": 1.0,
        "bias_prediction_intercept": bias,
        "affine_prediction_slope": slope,
        "affine_prediction_intercept": intercept,
    }


def rmse(prediction: np.ndarray, clean: np.ndarray, labels: np.ndarray, code: int) -> float:
    dev = labels == code
    if not np.any(dev):
        raise ValueError("development split is empty")
    error = np.asarray(prediction[dev], dtype=np.float64) - np.asarray(clean[dev], dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def evaluate(noisy: np.ndarray, clean: np.ndarray, labels: np.ndarray, coefficients: dict[str, float]) -> dict[str, float]:
    x = np.asarray(noisy)
    identity = x
    bias = x + coefficients["bias_prediction_intercept"]
    affine = coefficients["affine_prediction_slope"] * x + coefficients["affine_prediction_intercept"]
    return {
        "identity": rmse(identity, clean, labels, 1),
        "bias_correction": rmse(bias, clean, labels, 1),
        "global_affine": rmse(affine, clean, labels, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/23_Team_Toxic"))
    parser.add_argument("--output", type=Path, default=Path("results/baselines/weight_v1.json"))
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--guard", type=int, default=32)
    args = parser.parse_args()
    data_dir = args.data_dir
    clean_path = data_dir / "train_weights_clean.npy"
    noisy_path = data_dir / "train_weights_noisy.npy"
    clean = np.load(clean_path, mmap_mode="r")
    noisy = np.load(noisy_path, mmap_mode="r")
    if clean.shape != noisy.shape or clean.ndim != 1 or not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError("clean/noisy arrays must be finite, one-dimensional, and shape-aligned")
    if clean.size % 2:
        raise ValueError("training array length must have two equal halves")
    # Freeze position-only assignment before any clean labels enter a fit.
    labels = make_assignment(
        int(clean.size), half_length=int(clean.size // 2), seed=args.seed,
        block_size=args.block_size, guard=args.guard,
    )
    validate_assignment(labels)
    coefficients = fit_coefficients(noisy, clean, labels)
    metrics = evaluate(noisy, clean, labels, coefficients)
    split_counts = {
        "fitting": int(np.count_nonzero(labels == 0)),
        "development": int(np.count_nonzero(labels == 1)),
        "audit": int(np.count_nonzero(labels == 2)),
        "guard": int(np.count_nonzero(labels == -1)),
    }
    split_spec = {
        "rule": "sha256(weight-v1|seed|group_id) byte thresholds 0.60/0.80; same offset group in both halves",
        "seed": args.seed,
        "block_size": args.block_size,
        "guard": args.guard,
        "half_length": int(clean.size // 2),
        "total_length": int(clean.size),
    }
    split_hash = hashlib.sha256(json.dumps(split_spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {
        "schema_version": 1,
        "experiment": "BASE-WEIGHT-001",
        "team": 23,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "split": {**split_spec, "spec_sha256": split_hash, "counts": split_counts},
        "input_sha256": {name: sha256(data_dir / name) for name in ("train_weights_clean.npy", "train_weights_noisy.npy")},
        "fit_coefficients": coefficients,
        "development_rmse_original_units": metrics,
        "audit_metrics": None,
        "limitations": [
            "The clean audit labels are reserved and no audit-label metric is computed.",
            "The equal-length halves are grouped conservatively by offset only; their checkpoint meaning is unverified.",
            "Scalar baselines do not establish performance on the clean test weights, whose labels are unavailable.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "development_rmse_original_units": metrics, "counts": split_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
