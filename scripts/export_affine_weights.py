#!/usr/bin/env python3
"""Export a reproducible global-affine restoration of Team 23 test weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.weight_baselines import sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "results" / "baselines" / "weight_v1.json"
DEFAULT_DATA = ROOT / "data" / "23_Team_Toxic"
DEFAULT_OUTPUT = ROOT / "results" / "candidates" / "affine_v1.npy"


def export(baseline_path: Path, data_dir: Path, output: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("experiment") != "BASE-WEIGHT-001" or baseline.get("audit_metrics") is not None:
        raise ValueError("expected an un-audited Team 23 weight baseline")
    for name, expected in baseline["input_sha256"].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f"training file hash changed: {name}")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    test_path = data_dir / "test_weights_noisy.npy"
    test_hash = sha256(test_path)
    if test_hash != manifest["files"][test_path.name]["sha256"]:
        raise ValueError("test weight hash differs from the preflight manifest")
    noisy = np.load(test_path, allow_pickle=False)
    if noisy.dtype != np.float32 or noisy.shape != (2_961_630,) or not np.isfinite(noisy).all():
        raise ValueError("test noisy vector has invalid dtype, shape, or values")
    coeff = baseline["fit_coefficients"]
    slope = float(coeff["affine_prediction_slope"])
    intercept = float(coeff["affine_prediction_intercept"])
    restored = np.asarray(slope * noisy.astype(np.float64) + intercept, dtype=np.float32)
    if not np.isfinite(restored).all():
        raise ValueError("restored vector contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, restored, allow_pickle=False)
    if not np.array_equal(np.load(output, allow_pickle=False), restored):
        raise ValueError("saved restored vector failed round-trip verification")
    provenance = {
        "method": "fitting_split_global_affine",
        "team": 23,
        "shape": list(restored.shape),
        "dtype": str(restored.dtype),
        "baseline_sha256": sha256(baseline_path),
        "test_noisy_sha256": test_hash,
        "restored_sha256": sha256(output),
        "affine_prediction_slope": slope,
        "affine_prediction_intercept": intercept,
        "development_rmse_original_units": baseline["development_rmse_original_units"]["global_affine"],
        "audit_metrics": None,
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.baseline, args.data_dir, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
