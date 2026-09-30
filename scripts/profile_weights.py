#!/usr/bin/env python3
"""Descriptive, model-free profile of the assigned Team 23 weight arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_count(a: np.ndarray) -> int:
    return int(np.isfinite(a).sum())


def qstats(a: np.ndarray) -> dict[str, float]:
    q = np.percentile(a, np.array(QUANTILES) * 100.0)
    return {f"q{int(p * 100):02d}": float(v) for p, v in zip(QUANTILES, q)}


def summary(a: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite": finite_count(a),
        "count": int(a.size),
        "mean": float(np.mean(a, dtype=np.float64)),
        "std": float(np.std(a, dtype=np.float64)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "quantiles": qstats(a),
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    xc = x64 - np.mean(x64)
    yc = y64 - np.mean(y64)
    denom = np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
    return float(np.dot(xc, yc) / denom) if denom else float("nan")


def residual_profile(clean: np.ndarray, noisy: np.ndarray) -> dict[str, object]:
    r = np.asarray(noisy, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    r_std = float(np.std(r))
    abs_r = np.abs(r)
    return {
        "count": int(r.size),
        "bias_mean": float(np.mean(r)),
        "scale_std": r_std,
        "mae": float(np.mean(abs_r)),
        "identity_rmse": float(np.sqrt(np.mean(r * r))),
        "residual_quantiles": qstats(r),
        "absolute_residual_quantiles": qstats(abs_r),
        "tail_fraction_abs_gt_2std": float(np.mean(abs_r > 2.0 * r_std)) if r_std else 0.0,
        "tail_fraction_abs_gt_3std": float(np.mean(abs_r > 3.0 * r_std)) if r_std else 0.0,
        "max_abs_residual": float(np.max(abs_r)),
        "global_lag_correlation": {
            str(lag): correlation(r[:-lag], r[lag:])
            for lag in (1, 2, 4, 8, 16)
            if lag < r.size
        },
        "noisy_clean_correlation": correlation(clean, noisy),
        "affine_diagnostic": affine_diagnostic(clean, noisy),
    }


def affine_diagnostic(clean: np.ndarray, noisy: np.ndarray) -> dict[str, float]:
    x = np.asarray(clean, dtype=np.float64)
    y = np.asarray(noisy, dtype=np.float64)
    xc = x - np.mean(x)
    var_x = float(np.mean(xc * xc))
    slope = float(np.mean(xc * (y - np.mean(y))) / var_x) if var_x else float("nan")
    intercept = float(np.mean(y) - slope * np.mean(x)) if np.isfinite(slope) else float("nan")
    return {"least_squares_slope": slope, "least_squares_intercept": intercept}


def distribution_comparison(train_noisy: np.ndarray, test_noisy: np.ndarray) -> dict[str, object]:
    train64 = np.asarray(train_noisy, dtype=np.float64)
    test64 = np.asarray(test_noisy, dtype=np.float64)
    tq = qstats(train64)
    xq = qstats(test64)
    return {
        "train_noisy": summary(train_noisy),
        "test_noisy": summary(test_noisy),
        "test_minus_train_mean": float(np.mean(test64) - np.mean(train64)),
        "test_over_train_std": float(np.std(test64) / np.std(train64)),
        "test_minus_train_quantiles": {k: xq[k] - tq[k] for k in tq},
        "test_fraction_below_train_q01": float(np.mean(test64 < tq["q01"])),
        "test_fraction_above_train_q99": float(np.mean(test64 > tq["q99"])),
    }


def fmt_num(x: float) -> str:
    return f"{x:.6g}"


def table_summary(rows: dict[str, dict[str, object]]) -> str:
    out = [
        "| Array | Shape | Dtype | Finite / count | Mean | Std | Min | Max |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in rows.items():
        out.append(
            f"| `{name}` | `{s['shape']}` | `{s['dtype']}` | "
            f"{s['finite']} / {s['count']} | {fmt_num(s['mean'])} | {fmt_num(s['std'])} | "
            f"{fmt_num(s['min'])} | {fmt_num(s['max'])} |"
        )
    return "\n".join(out)


def quantile_table(rows: dict[str, dict[str, object]], key: str = "quantiles") -> str:
    labels = [f"q{int(p * 100):02d}" for p in QUANTILES]
    out = ["| Array | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for name, s in rows.items():
        q = s[key]
        out.append("| `" + name + "` | " + " | ".join(fmt_num(q[k]) for k in labels) + " |")
    return "\n".join(out)


def build_report(root: Path, manifest: dict[str, object], clean: np.ndarray, noisy: np.ndarray, test: np.ndarray, hashes: dict[str, str], command: str) -> str:
    clean_s = summary(clean)
    noisy_s = summary(noisy)
    test_s = summary(test)
    residual = residual_profile(clean, noisy)
    comparison = distribution_comparison(noisy, test)
    expected = manifest["files"]
    hash_lines = []
    for name in ("train_weights_clean.npy", "train_weights_noisy.npy", "test_weights_noisy.npy"):
        exp = expected[name]["sha256"]
        got = hashes[name]
        hash_lines.append(f"| `{name}` | `{exp}` | `{got}` | {'PASS' if exp == got else 'FAIL'} |")
    rq = residual["residual_quantiles"]
    arq = residual["absolute_residual_quantiles"]
    lag = residual["global_lag_correlation"]
    cmp_q = comparison["test_minus_train_quantiles"]
    return f"""# Team 23 weight profile

Descriptive audit only. No model, fit, split, validation target, checkpoint, or submission was created.

## Scope and reproducibility

- Team: `{manifest['team']}`; source directory: `data/23_Team_Toxic/`.
- Generated: `{datetime.now(timezone.utc).isoformat()}`.
- Python: `{platform.python_version()}`; NumPy: `{np.__version__}`.
- Exact command: `{command}`
- Inputs were loaded with `numpy.load(..., mmap_mode="r")`; residual calculations were performed in float64 for numerical stability.

## Manifest hash and array checks

| File | Manifest SHA-256 | Observed SHA-256 | Result |
|---|---|---|---|
{chr(10).join(hash_lines)}

{table_summary({'train_weights_clean.npy': clean_s, 'train_weights_noisy.npy': noisy_s, 'test_weights_noisy.npy': test_s})}

All three arrays have the manifest-declared one-dimensional shapes and float32 dtypes. The finite counts equal the element counts: no NaN or infinity was observed.

### Value quantiles

{quantile_table({'train clean': clean_s, 'train noisy': noisy_s, 'test noisy': test_s})}

## Training clean/noisy residual diagnostics

Residual definition: `r = train_weights_noisy - train_weights_clean`, aligned by array position.

| Metric | Value |
|---|---:|
| Count | {residual['count']} |
| Residual mean (bias) | {fmt_num(residual['bias_mean'])} |
| Residual standard deviation (scale) | {fmt_num(residual['scale_std'])} |
| Mean absolute residual | {fmt_num(residual['mae'])} |
| Identity RMSE | {fmt_num(residual['identity_rmse'])} |
| Noisy/clean Pearson correlation | {fmt_num(residual['noisy_clean_correlation'])} |
| Noisy-on-clean affine diagnostic slope | {fmt_num(residual['affine_diagnostic']['least_squares_slope'])} |
| Noisy-on-clean affine diagnostic intercept | {fmt_num(residual['affine_diagnostic']['least_squares_intercept'])} |

Residual quantiles:

| Statistic | q00 | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual | {' | '.join(fmt_num(rq[k]) for k in ['q00','q01','q05','q25','q50','q75','q95','q99','q100'])} |
| Absolute residual | {' | '.join(fmt_num(arq[k]) for k in ['q00','q01','q05','q25','q50','q75','q95','q99','q100'])} |

- Fraction with `abs(r) > 2 * residual_std`: `{fmt_num(residual['tail_fraction_abs_gt_2std'])}`.
- Fraction with `abs(r) > 3 * residual_std`: `{fmt_num(residual['tail_fraction_abs_gt_3std'])}`.
- Maximum absolute residual: `{fmt_num(residual['max_abs_residual'])}`.

### Global lag correlation

Residual Pearson correlations between `r[:-lag]` and `r[lag:]` over the entire flattened vector:

| Lag | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Correlation | {' | '.join(fmt_num(lag[str(k)]) for k in (1,2,4,8,16))} |

These are global flattened-array diagnostics. They can be driven by concatenation boundaries, repeated tensor layouts, or long-range position effects; they do not establish local stochastic dependence within a model tensor and do not justify a denoiser or generalization claim.

## Train/test noisy distribution comparison

The test clean target is unavailable, so this section compares only the observed noisy training and noisy test arrays.

| Metric | Value |
|---|---:|
| Test noisy mean minus train noisy mean | {fmt_num(comparison['test_minus_train_mean'])} |
| Test noisy std / train noisy std | {fmt_num(comparison['test_over_train_std'])} |
| Test fraction below train noisy q01 | {fmt_num(comparison['test_fraction_below_train_q01'])} |
| Test fraction above train noisy q99 | {fmt_num(comparison['test_fraction_above_train_q99'])} |

Differences in corresponding test-minus-train noisy quantiles:

| Quantile | q00 | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Difference | {' | '.join(fmt_num(cmp_q[k]) for k in ['q00','q01','q05','q25','q50','q75','q95','q99','q100'])} |

## Limits and interpretation

- The identity RMSE is a descriptive training-pair diagnostic, not a validation or test score.
- No validation split was created, and no clean test target exists. No generalization, leaderboard, or denoiser claim is supported.
- Flattened lag correlations ignore verified tensor boundaries and parameter semantics.
- Aggregate distributions can hide tensor-family or magnitude-dependent behavior; this audit does not infer the model layout.
- Raw array values were not written to the report; only aggregate statistics and hashes are recorded.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "data" / "23_Team_Toxic"
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = ("train_weights_clean.npy", "train_weights_noisy.npy", "test_weights_noisy.npy")
    paths = {name: data_dir / name for name in names}
    hashes = {name: sha256(path) for name, path in paths.items()}
    for name, observed in hashes.items():
        if observed != manifest["files"][name]["sha256"]:
            raise SystemExit(f"manifest hash mismatch for {name}")
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
    clean, noisy, test = (arrays[name] for name in names)
    for name, a in arrays.items():
        expected = manifest["files"][name]
        if list(a.shape) != expected["shape"] or str(a.dtype) != expected["dtype"]:
            raise SystemExit(f"schema mismatch for {name}: observed {a.shape}/{a.dtype}, expected {expected['shape']}/{expected['dtype']}")
        if finite_count(a) != a.size:
            raise SystemExit(f"non-finite values in {name}")
    if clean.shape != noisy.shape:
        raise SystemExit("clean/noisy training arrays are not aligned in shape")
    report = args.report or (root / "docs" / "WEIGHT_PROFILE.md")
    command = ".venv/bin/python scripts/profile_weights.py"
    report.write_text(build_report(root, manifest, clean, noisy, test, hashes, command))
    print(f"Wrote {report}")
    for name in names:
        print(f"{name}: sha256={hashes[name]} shape={arrays[name].shape} dtype={arrays[name].dtype} finite={finite_count(arrays[name])}/{arrays[name].size}")
    print(f"identity_rmse={residual_profile(clean, noisy)['identity_rmse']:.9g}")


if __name__ == "__main__":
    main()
