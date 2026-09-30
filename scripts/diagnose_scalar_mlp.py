#!/usr/bin/env python3
"""Read-only aggregate diagnosis for the rejected scalar MLP checkpoint.

This script intentionally evaluates fitting and development positions only. It
does not train, write outputs, or index clean audit values.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_scalar_mlp import ScalarResidualMLP, sha256_file, split_labels
from scripts.weight_baselines import fit_coefficients


DATA = ROOT / "data" / "23_Team_Toxic"
CHECKPOINT = ROOT / "results" / "checkpoints" / "scalar_mlp_v1.pt"


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def linear_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    slope = float(np.mean(xc * yc) / np.mean(xc * xc))
    intercept = float(y.mean() - slope * x.mean())
    corr = float(np.mean(xc * yc) / np.sqrt(np.mean(xc * xc) * np.mean(yc * yc)))
    return {"slope": slope, "intercept": intercept, "correlation": corr}


def _validate_checkpoint_metadata(
    payload: dict[str, object],
    *,
    clean_path: Path,
    noisy_path: Path,
    split: dict[str, object],
    labels: np.ndarray,
) -> None:
    """Reject a diagnosis if its checkpoint is for different inputs or splits."""
    if payload.get("schema_version") != 1:
        raise ValueError("diagnosis expects the rejected scalar_mlp_v1 checkpoint schema")
    expected_hashes = {"clean": sha256_file(clean_path), "noisy": sha256_file(noisy_path)}
    if payload.get("input_sha256") != expected_hashes:
        raise ValueError("checkpoint input hashes do not match the current Team 23 files")
    expected_split = {
        **split,
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
    }
    if payload.get("split") != expected_split:
        raise ValueError("checkpoint split metadata does not match the current fixed split")


def _predict_in_chunks(
    model: ScalarResidualMLP,
    noisy: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Run inference on selected positions without materializing a full input tensor."""
    if indices.size == 0:
        raise ValueError("diagnostic split is empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = np.empty(indices.size, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, indices.size, chunk_size):
            batch_indices = indices[start : start + chunk_size]
            batch = torch.from_numpy(np.asarray(noisy[batch_indices], dtype=np.float32))
            output[start : start + batch_indices.size] = model(batch).cpu().numpy().reshape(-1)
    return output


def main() -> None:
    noisy_path = DATA / "train_weights_noisy.npy"
    clean_path = DATA / "train_weights_clean.npy"
    noisy = np.load(noisy_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    clean = np.load(clean_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    labels, split = split_labels(int(noisy.size))
    fit = labels == 0
    dev = labels == 1

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    _validate_checkpoint_metadata(
        payload,
        clean_path=clean_path,
        noisy_path=noisy_path,
        split=split,
        labels=labels,
    )
    model = ScalarResidualMLP(int(payload["config"]["hidden"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    fit_indices = np.flatnonzero(fit)
    dev_indices = np.flatnonzero(dev)
    prediction_fit = _predict_in_chunks(model, noisy, fit_indices)
    prediction_dev = _predict_in_chunks(model, noisy, dev_indices)

    coefficients = fit_coefficients(noisy, clean, labels)
    affine = coefficients["affine_prediction_slope"] * noisy + coefficients["affine_prediction_intercept"]

    report: dict[str, object] = {
        "checkpoint": {
            "epoch": int(payload["epoch"]),
            "step": int(payload["step"]),
            "config": payload["config"],
            "input_sha256": {"clean": sha256_file(clean_path), "noisy": sha256_file(noisy_path)},
            "split_spec_sha256": split["spec_sha256"],
        },
        "counts": {"fitting": int(fit.sum()), "development": int(dev.sum())},
        "rmse": {
            "fitting": {
                "identity": rmse(noisy[fit], clean[fit]),
                "global_affine": rmse(affine[fit], clean[fit]),
                "scalar_mlp": rmse(prediction_fit, clean[fit]),
            },
            "development": {
                "identity": rmse(noisy[dev], clean[dev]),
                "global_affine": rmse(affine[dev], clean[dev]),
                "scalar_mlp": rmse(prediction_dev, clean[dev]),
            },
        },
        "mapping": {
            "clean_on_noisy": {"fitting": linear_summary(noisy[fit], clean[fit]), "development": linear_summary(noisy[dev], clean[dev])},
            "mlp_on_noisy": {"fitting": linear_summary(noisy[fit], prediction_fit), "development": linear_summary(noisy[dev], prediction_dev)},
            "prediction_summary": {
                "fitting_mean": float(np.mean(prediction_fit)),
                "fitting_std": float(np.std(prediction_fit)),
                "development_mean": float(np.mean(prediction_dev)),
                "development_std": float(np.std(prediction_dev)),
                "fitting_min": float(np.min(prediction_fit)),
                "fitting_max": float(np.max(prediction_fit)),
                "development_min": float(np.min(prediction_dev)),
                "development_max": float(np.max(prediction_dev)),
            },
        },
    }

    # A nonparametric scalar lookup is a cheap capacity/optimization control:
    # fit conditional means in noisy-value bins on fitting positions only.
    edges = np.quantile(np.asarray(noisy[fit], dtype=np.float64), np.linspace(0.0, 1.0, 33))
    edges = np.unique(edges)
    fit_bins = np.clip(np.searchsorted(edges, noisy[fit], side="right") - 1, 0, len(edges) - 2)
    dev_bins = np.clip(np.searchsorted(edges, noisy[dev], side="right") - 1, 0, len(edges) - 2)
    lookup = np.array([np.mean(clean[fit][fit_bins == i]) if np.any(fit_bins == i) else 0.0 for i in range(len(edges) - 1)])
    lookup_dev = lookup[dev_bins]
    lookup_fit = lookup[fit_bins]
    report["scalar_lookup_32_bins"] = {
        "bin_count": int(len(lookup)),
        "fitting_rmse": rmse(lookup_fit, clean[fit]),
        "development_rmse": rmse(lookup_dev, clean[dev]),
        "note": "conditional means fitted on fitting positions only; aggregate diagnostic, not a promoted candidate",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
