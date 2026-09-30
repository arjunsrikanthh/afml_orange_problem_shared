#!/usr/bin/env python3
"""Compare the fitting-only affine baseline with validated scalar MLP checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from scripts.train_scalar_mlp import SCHEMA_VERSION, ScalarResidualMLP, split_labels
from scripts.weight_baselines import fit_coefficients


REQUIRED_CHECKPOINT_KEYS = {
    "schema_version", "model", "optimizer", "scheduler", "epoch", "step",
    "config", "input_sha256", "split", "rng",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_arrays(noisy: np.ndarray, clean: np.ndarray, labels: np.ndarray) -> None:
    if noisy.ndim != 1 or clean.ndim != 1 or labels.ndim != 1:
        raise ValueError("noisy, clean, and labels must be one-dimensional")
    if noisy.shape != clean.shape or noisy.shape != labels.shape:
        raise ValueError("noisy, clean, and labels must have identical shapes")
    if not np.issubdtype(noisy.dtype, np.floating) or not np.issubdtype(clean.dtype, np.floating):
        raise ValueError("noisy and clean arrays must be floating point")
    if not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError("split labels contain an unsupported code")
    # Do not call isfinite on clean[labels == 2]. Audit targets are reserved.
    _finite(np.asarray(noisy), "noisy")
    allowed = (labels == 0) | (labels == 1)
    _finite(np.asarray(clean[allowed]), "fitting/development clean values")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("fitting and development splits must be non-empty")


def contiguous_development_blocks(labels: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return half-open contiguous runs whose label is development (1)."""
    labels = np.asarray(labels)
    blocks: list[list[tuple[int, int]]] = []
    start: int | None = None
    for position, is_development in enumerate(labels == 1):
        if is_development and start is None:
            start = position
        elif not is_development and start is not None:
            blocks.append([(start, position)])
            start = None
    if start is not None:
        blocks.append([(start, labels.size)])
    if not blocks:
        raise ValueError("development split has no contiguous blocks")
    return blocks


def paired_development_blocks(labels: np.ndarray) -> list[list[tuple[int, int]]]:
    """Group same-offset contiguous development runs across equal halves."""
    labels = np.asarray(labels)
    if labels.size % 2 == 0 and np.array_equal(labels[: labels.size // 2], labels[labels.size // 2 :]):
        half = labels.size // 2
        first = contiguous_development_blocks(labels[:half])
        return [segments + [(start + half, stop + half) for start, stop in segments] for segments in first]
    return contiguous_development_blocks(labels)


def _validate_prediction(prediction: np.ndarray, name: str, expected_shape: tuple[int, ...]) -> None:
    if prediction.shape != expected_shape:
        raise ValueError(f"{name} prediction shape does not match inputs")
    _finite(prediction, f"{name} prediction")


def _reject_nonfinite(value: Any, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint {path} contains non-finite tensor values")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"checkpoint {path} contains non-finite array values")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"checkpoint {path} contains a non-finite value")
    elif isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]")


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _block_statistics(affine_prediction: np.ndarray, mlp_prediction: np.ndarray, clean: np.ndarray, blocks: Iterable[list[tuple[int, int]]]) -> tuple[list[dict[str, Any]], int, float, float]:
    details: list[dict[str, Any]] = []
    affine_total = 0.0
    mlp_total = 0.0
    total_count = 0
    for block_id, segments in enumerate(blocks):
        affine_sse = 0.0
        mlp_sse = 0.0
        count = 0
        for start, stop in segments:
            target = np.asarray(clean[start:stop], dtype=np.float64)
            affine_error = np.asarray(affine_prediction[start:stop], dtype=np.float64) - target
            mlp_error = np.asarray(mlp_prediction[start:stop], dtype=np.float64) - target
            affine_sse += float(np.dot(affine_error, affine_error))
            mlp_sse += float(np.dot(mlp_error, mlp_error))
            count += int(stop - start)
        affine_mse = affine_sse / count
        mlp_mse = mlp_sse / count
        details.append({
            "block": block_id, "segments": [[start, stop] for start, stop in segments], "count": count,
            "affine_sse": affine_sse, "mlp_sse": mlp_sse,
            "affine_mse": affine_mse, "mlp_mse": mlp_mse,
            "mlp_minus_affine_mse": mlp_mse - affine_mse,
            "winner": "mlp" if mlp_mse < affine_mse else "affine" if affine_mse < mlp_mse else "tie",
        })
        affine_total += affine_sse
        mlp_total += mlp_sse
        total_count += count
    return details, total_count, affine_total, mlp_total


def _bootstrap_rmse_difference(blocks: list[dict[str, Any]], seed: int, repetitions: int) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for draw in range(repetitions):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        total_count = sum(blocks[index]["count"] for index in selected)
        affine_sse = sum(blocks[index]["affine_sse"] for index in selected)
        mlp_sse = sum(blocks[index]["mlp_sse"] for index in selected)
        samples[draw] = np.sqrt(mlp_sse / total_count) - np.sqrt(affine_sse / total_count)
    return {
        "seed": int(seed), "repetitions": int(repetitions),
        "statistic": "count_weighted_mlp_rmse_minus_affine_rmse",
        "percentile": [2.5, 97.5],
        "resampling_unit": "linked same-offset contiguous development runs",
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "dependence_caveat": "Linked same-offset half-blocks are resampled together; positions within blocks and across halves may still be dependent.",
    }


def compare_predictions(noisy: np.ndarray, clean: np.ndarray, labels: np.ndarray, mlp_prediction: np.ndarray, *, bootstrap_seed: int = 2301, bootstrap_repetitions: int = 2000) -> dict[str, Any]:
    """Fit affine on fitting positions and compare it with supplied MLP outputs."""
    noisy, clean, labels, mlp_prediction = map(np.asarray, (noisy, clean, labels, mlp_prediction))
    _validate_arrays(noisy, clean, labels)
    _validate_prediction(mlp_prediction, "MLP", noisy.shape)
    coefficients = fit_coefficients(noisy, clean, labels)
    noisy64 = np.asarray(noisy, dtype=np.float64)
    affine_prediction = coefficients["affine_prediction_slope"] * noisy64 + coefficients["affine_prediction_intercept"]
    _validate_prediction(affine_prediction, "affine", noisy.shape)
    dev = labels == 1
    dev_clean = np.full(clean.shape, np.nan, dtype=np.float64)
    dev_clean[dev] = clean[dev]
    details, count, affine_sse, mlp_sse = _block_statistics(affine_prediction, mlp_prediction, dev_clean, paired_development_blocks(labels))
    affine_rmse = float(np.sqrt(affine_sse / count))
    mlp_rmse = float(np.sqrt(mlp_sse / count))
    mlp_wins = sum(item["winner"] == "mlp" for item in details)
    affine_wins = sum(item["winner"] == "affine" for item in details)
    return {
        "development_count": count, "block_count": len(details),
        "affine": {"slope": coefficients["affine_prediction_slope"], "intercept": coefficients["affine_prediction_intercept"], "rmse": affine_rmse, "sse": affine_sse},
        "mlp": {"rmse": mlp_rmse, "sse": mlp_sse},
        "mlp_minus_affine": {"rmse": mlp_rmse - affine_rmse, "sse": mlp_sse - affine_sse, "mse": (mlp_sse - affine_sse) / count},
        "block_wins_losses": {"mlp_wins": mlp_wins, "affine_wins": affine_wins, "ties": len(details) - mlp_wins - affine_wins},
        "blocks": details,
        "bootstrap": _bootstrap_rmse_difference(details, bootstrap_seed, bootstrap_repetitions),
    }


def _expected_split(labels: np.ndarray, split: dict[str, Any]) -> dict[str, Any]:
    expected = dict(split)
    expected["labels_sha256"] = hashlib.sha256(np.asarray(labels).tobytes()).hexdigest()
    return expected


def _validate_checkpoint(path: Path, expected_input: dict[str, str], expected_split: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"unable to read checkpoint: {path}") from exc
    if not isinstance(payload, dict) or not REQUIRED_CHECKPOINT_KEYS.issubset(payload):
        raise ValueError("checkpoint schema is incomplete")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("only schema2 scalar MLP checkpoints are accepted")
    if payload["input_sha256"] != expected_input:
        raise ValueError("checkpoint input hashes do not match current arrays")
    if payload["split"] != expected_split:
        raise ValueError("checkpoint split metadata does not match current split")
    if not isinstance(payload["config"], dict) or not _strict_int(payload["config"].get("hidden", 0)) or payload["config"].get("hidden", 0) <= 0:
        raise ValueError("checkpoint config has no positive hidden width")
    required_config = {"hidden", "learning_rate", "weight_decay", "batch_size", "epochs", "chunk_size", "seed"}
    if set(payload["config"]) != required_config:
        raise ValueError("checkpoint config is incomplete")
    if any(not _strict_int(payload["config"][key]) or payload["config"][key] <= 0 for key in ("hidden", "batch_size", "epochs", "chunk_size", "seed")) or isinstance(payload["config"]["learning_rate"], bool) or isinstance(payload["config"]["weight_decay"], bool) or float(payload["config"]["learning_rate"]) <= 0 or float(payload["config"]["weight_decay"]) < 0:
        raise ValueError("checkpoint config has invalid training values")
    if not isinstance(payload["optimizer"], dict) or not {"state", "param_groups"}.issubset(payload["optimizer"]):
        raise ValueError("checkpoint optimizer state is malformed")
    if not payload["optimizer"]["param_groups"] or int(payload["epoch"]) > 0 and not payload["optimizer"]["state"]:
        raise ValueError("trained checkpoint optimizer state is empty")
    if not isinstance(payload["scheduler"], dict) or not {"last_epoch", "_step_count", "base_lrs"}.issubset(payload["scheduler"]):
        raise ValueError("checkpoint scheduler state is malformed")
    if not isinstance(payload["rng"], dict) or not {"python", "numpy", "torch"}.issubset(payload["rng"]):
        raise ValueError("checkpoint optimizer, scheduler, and RNG state are malformed")
    if not isinstance(payload["rng"]["python"], tuple) or len(payload["rng"]["python"]) != 3 or not isinstance(payload["rng"]["numpy"], tuple) or len(payload["rng"]["numpy"]) != 5 or not isinstance(payload["rng"]["torch"], torch.Tensor) or payload["rng"]["torch"].dtype != torch.uint8:
        raise ValueError("checkpoint RNG state is incomplete")
    try:
        random_state = random.Random()
        random_state.setstate(payload["rng"]["python"])
        numpy_state = np.random.RandomState()
        numpy_state.set_state(payload["rng"]["numpy"])
        torch_state = torch.Generator(device="cpu")
        torch_state.set_state(payload["rng"]["torch"])
    except Exception as exc:
        raise ValueError("checkpoint RNG state is malformed") from exc
    for key in ("model", "optimizer", "scheduler", "rng", "config"):
        _reject_nonfinite(payload[key], key)
    if not all(_strict_int(payload[key]) and payload[key] >= 0 for key in ("epoch", "step")):
        raise ValueError("checkpoint progress counters are invalid")
    expected_steps = int(payload["epoch"]) * ((int(expected_split["counts"]["fitting"]) + int(payload["config"]["batch_size"]) - 1) // int(payload["config"]["batch_size"]))
    if int(payload["epoch"]) > int(payload["config"]["epochs"]) or int(payload["step"]) != expected_steps:
        raise ValueError("checkpoint progress is inconsistent with config and fitting count")
    model = ScalarResidualMLP(int(payload["config"]["hidden"]))
    if not isinstance(payload["model"], dict) or set(payload["model"]) != set(model.state_dict()):
        raise ValueError("checkpoint model state keys are incompatible")
    for key, expected_tensor in model.state_dict().items():
        actual_tensor = payload["model"][key]
        if not isinstance(actual_tensor, torch.Tensor) or actual_tensor.shape != expected_tensor.shape or actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"checkpoint model tensor {key} has an incompatible shape or dtype")
    try:
        model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:
        raise ValueError("checkpoint model state is incompatible with ScalarResidualMLP") from exc
    for tensor in model.state_dict().values():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("checkpoint model contains non-finite parameters")
    return model, payload


def evaluate_checkpoint(noisy: np.ndarray, clean: np.ndarray, labels: np.ndarray, checkpoint_paths: Iterable[Path], *, input_hashes: dict[str, str], split: dict[str, Any], bootstrap_seed: int = 2301, bootstrap_repetitions: int = 2000, chunk_size: int = 65536) -> dict[str, Any]:
    noisy, clean, labels = map(np.asarray, (noisy, clean, labels))
    _validate_arrays(noisy, clean, labels)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    expected_split = _expected_split(labels, split)
    results = []
    dev_indices = np.flatnonzero(labels == 1)
    for path in checkpoint_paths:
        path = Path(path)
        model, payload = _validate_checkpoint(path, input_hashes, expected_split)
        # Values outside development are deliberately inert; compare_predictions
        # only consumes the paired development blocks.
        prediction = np.zeros(noisy.shape, dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for start in range(0, dev_indices.size, chunk_size):
                indices = dev_indices[start : start + chunk_size]
                values = torch.from_numpy(np.asarray(noisy[indices])).to(dtype=torch.float32)
                output = model(values)
                if output.shape != values.shape or output.ndim != 1:
                    raise ValueError("checkpoint model returned an incompatible prediction shape")
                output_array = output.cpu().numpy()
                if output_array.shape != (indices.size,):
                    raise ValueError("checkpoint model returned an incompatible prediction shape")
                prediction[indices] = output_array
        report = compare_predictions(noisy, clean, labels, prediction, bootstrap_seed=bootstrap_seed, bootstrap_repetitions=bootstrap_repetitions)
        report["checkpoint"] = {"path": str(path), "sha256": sha256_file(path), "schema_version": payload["schema_version"], "epoch": payload["epoch"], "step": payload["step"]}
        results.append(report)
    if not results:
        raise ValueError("at least one schema2 checkpoint is required")
    return {"schema_version": 1, "input_sha256": input_hashes, "split": expected_split, "candidates": results, "audit_metrics": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/23_Team_Toxic"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--guard", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=2301)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    clean_path = args.data_dir / "train_weights_clean.npy"
    noisy_path = args.data_dir / "train_weights_noisy.npy"
    clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
    noisy = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
    if clean.ndim != 1 or noisy.ndim != 1:
        raise ValueError("clean and noisy arrays must be one-dimensional")
    labels, split = split_labels(int(noisy.size), seed=args.seed, block_size=args.block_size, guard=args.guard)
    hashes = {"clean": sha256_file(clean_path), "noisy": sha256_file(noisy_path)}
    report = evaluate_checkpoint(noisy, clean, labels, args.checkpoint, input_hashes=hashes, split=split, bootstrap_seed=args.bootstrap_seed, bootstrap_repetitions=args.bootstrap_repetitions, chunk_size=args.chunk_size)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
