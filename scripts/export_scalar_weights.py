#!/usr/bin/env python3
"""Export a schema2 scalar MLP checkpoint as verified Part 1 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.train_scalar_mlp import SCHEMA_VERSION, ScalarResidualMLP, _validate_checkpoint, sha256_file, split_labels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "23_Team_Toxic"
DEFAULT_CHECKPOINT = ROOT / "results" / "checkpoints" / "scalar_mlp.pt"
DEFAULT_NPY = ROOT / "results" / "candidates" / "scalar_mlp_restored.npy"
DEFAULT_CSV = ROOT / "results" / "candidates" / "scalar_mlp_submission.csv"
EXPECTED_TEST_COUNT = 2_961_630
HASH_LENGTH = 64
FIXED_SEED = 2301
FIXED_BLOCK_SIZE = 4096
FIXED_GUARD = 32


def _finite_tree(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint contains non-finite value in {name}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"checkpoint contains non-finite value in {name}")
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"checkpoint contains non-finite value in {name}")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")


def _hash(path: Path) -> str:
    return sha256_file(path)


def _validate_hashes(payload: dict[str, Any], data_dir: Path) -> dict[str, str]:
    hashes = payload.get("input_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"clean", "noisy"}:
        raise ValueError("checkpoint input hashes are missing or incomplete")
    for key, digest in hashes.items():
        if not isinstance(digest, str) or len(digest) != HASH_LENGTH or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"checkpoint input hash is invalid: {key}")
        path = data_dir / ("train_weights_clean.npy" if key == "clean" else "train_weights_noisy.npy")
        if not path.is_file() or _hash(path) != digest:
            raise ValueError(f"checkpoint input hash does not match local file: {key}")
    return {key: str(value) for key, value in hashes.items()}


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("checkpoint config is missing or incompatible")
    required = {"hidden", "learning_rate", "weight_decay", "batch_size", "epochs", "chunk_size", "seed"}
    if set(config) != required:
        raise ValueError("checkpoint config is missing or incompatible")
    integer_keys = {"hidden", "batch_size", "epochs", "chunk_size", "seed"}
    for key in integer_keys:
        if type(config[key]) is not int:
            raise ValueError("checkpoint config is missing or incompatible")
    if any(config[key] <= 0 for key in ("hidden", "batch_size", "epochs", "chunk_size")):
        raise ValueError("checkpoint config is missing or incompatible")
    if type(config["learning_rate"]) not in (int, float) or not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("checkpoint config is missing or incompatible")
    if type(config["weight_decay"]) not in (int, float) or not np.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("checkpoint config is missing or incompatible")
    return dict(config)


def _validate_split(payload: dict[str, Any], total_length: int) -> dict[str, Any]:
    split = payload.get("split")
    if not isinstance(split, dict):
        raise ValueError("checkpoint split is missing or incompatible")
    if (split.get("seed"), split.get("block_size"), split.get("guard")) != (FIXED_SEED, FIXED_BLOCK_SIZE, FIXED_GUARD):
        raise ValueError("checkpoint split is not the fixed Team 23 scalar split")
    labels, expected = split_labels(
        total_length,
        seed=FIXED_SEED,
        block_size=FIXED_BLOCK_SIZE,
        guard=FIXED_GUARD,
    )
    expected["labels_sha256"] = hashlib.sha256(labels.tobytes()).hexdigest()
    if split != expected:
        raise ValueError("checkpoint split hash or specification does not match")
    return expected


def _load_validated_checkpoint(checkpoint_path: Path, data_dir: Path, total_length: int) -> tuple[dict[str, Any], ScalarResidualMLP, dict[str, str], dict[str, Any]]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("unable to read scalar checkpoint") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema is incomplete or unsupported")
    config = _validate_config(payload.get("config"))
    input_hashes = _validate_hashes(payload, data_dir)
    clean = np.load(data_dir / "train_weights_clean.npy", mmap_mode="r", allow_pickle=False)
    noisy = np.load(data_dir / "train_weights_noisy.npy", mmap_mode="r", allow_pickle=False)
    if clean.shape != (total_length,) or noisy.shape != (total_length,) or clean.dtype != np.float32 or noisy.dtype != np.float32:
        raise ValueError("training arrays are incompatible with the checkpoint")
    split = _validate_split(payload, total_length)
    expected = {"config": config, "input_sha256": input_hashes, "split": split}
    _validate_checkpoint(payload, expected)
    if not isinstance(payload["optimizer"], dict) or not payload["optimizer"].get("state") or not payload["optimizer"].get("param_groups"):
        raise ValueError("checkpoint optimizer state is missing or incomplete")
    if not isinstance(payload["scheduler"], dict) or type(payload["scheduler"].get("last_epoch")) is not int or type(payload["scheduler"].get("_step_count")) is not int or payload["scheduler"]["last_epoch"] < -1 or payload["scheduler"]["_step_count"] < 0:
        raise ValueError("checkpoint scheduler state is missing or incomplete")
    if not (0 <= payload["epoch"] <= config["epochs"] and payload["step"] >= 0):
        raise ValueError("checkpoint progress counters are invalid")
    rng = payload["rng"]
    if not isinstance(rng, dict) or set(rng) - {"python", "numpy", "torch", "mps"} or not {"python", "numpy", "torch"}.issubset(rng):
        raise ValueError("checkpoint RNG state is missing or incomplete")
    if not isinstance(rng["python"], tuple) or not isinstance(rng["numpy"], tuple) or not isinstance(rng["torch"], torch.Tensor):
        raise ValueError("checkpoint RNG state is missing or incomplete")
    for key in ("model", "optimizer", "scheduler", "rng"):
        _finite_tree(payload[key], key)
    model = ScalarResidualMLP(config["hidden"])
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        optimizer.load_state_dict(payload["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
        scheduler.load_state_dict(payload["scheduler"])
    except Exception as exc:
        raise ValueError("checkpoint model state is incompatible") from exc
    model.eval()
    return payload, model, input_hashes, split


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, values, allow_pickle=False)
        generated = temporary.with_suffix(temporary.suffix + ".npy")
        os.replace(generated, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        generated = temporary.with_suffix(temporary.suffix + ".npy")
        if generated.exists():
            generated.unlink()


def _write_csv(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", mode="w", newline="", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "weight"))
        for index, value in enumerate(values):
            writer.writerow((index, repr(float(value))))
    os.replace(temporary, path)


def _verify_outputs(npy_path: Path, csv_path: Path, values: np.ndarray) -> None:
    saved = np.load(npy_path, allow_pickle=False)
    if saved.dtype != np.float32 or saved.shape != values.shape or not np.array_equal(saved, values):
        raise ValueError("saved NPY does not match the restored vector")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["id", "weight"]:
            raise ValueError("CSV header must be exactly id,weight")
        count = 0
        for row in reader:
            if count >= values.size:
                raise ValueError("CSV contains more rows than the restored vector")
            if len(row) != 2 or row[0] != str(count):
                raise ValueError("CSV IDs are missing, reordered, or duplicated")
            try:
                parsed = np.float32(row[1])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("CSV contains an invalid weight") from exc
            if not np.isfinite(parsed) or parsed != values[count]:
                raise ValueError("CSV weights do not match the restored NPY")
            count += 1
        if count != values.size:
            raise ValueError("CSV row count does not match the restored vector")


def export(checkpoint: Path, data_dir: Path, npy_output: Path, csv_output: Path, *, chunk_size: int = 65_536, expected_count: int = EXPECTED_TEST_COUNT) -> dict[str, Any]:
    if chunk_size <= 0 or expected_count <= 0:
        raise ValueError("chunk_size and expected_count must be positive")
    protected = {checkpoint.resolve(), data_dir.resolve() / "manifest.json", data_dir.resolve() / "test_weights_noisy.npy", data_dir.resolve() / "train_weights_clean.npy", data_dir.resolve() / "train_weights_noisy.npy"}
    output_paths = {npy_output.resolve(), csv_output.resolve(), npy_output.with_suffix(".json").resolve()}
    if len(output_paths) != 3 or protected & output_paths or any(path.is_relative_to(data_dir.resolve()) for path in output_paths):
        raise ValueError("output paths must be distinct and cannot overwrite inputs or checkpoint")
    test_path = data_dir / "test_weights_noisy.npy"
    if not test_path.is_file():
        raise ValueError("test noisy weights are missing")
    test = np.load(test_path, mmap_mode="r", allow_pickle=False)
    if test.ndim != 1 or test.dtype != np.float32 or test.size != expected_count:
        raise ValueError("test noisy vector has invalid dtype, shape, or exact count")
    if not np.isfinite(test).all():
        raise ValueError("test noisy vector contains non-finite values")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Team 23 manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_hash = _hash(test_path)
    if manifest.get("files", {}).get(test_path.name, {}).get("sha256") != test_hash:
        raise ValueError("test weight hash differs from the preflight manifest")
    clean_header = np.load(data_dir / "train_weights_clean.npy", mmap_mode="r", allow_pickle=False)
    payload, model, input_hashes, split = _load_validated_checkpoint(checkpoint, data_dir, int(clean_header.size))
    restored = np.empty(expected_count, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, expected_count, chunk_size):
            stop = min(start + chunk_size, expected_count)
            batch = torch.from_numpy(np.array(test[start:stop], copy=True)).to(dtype=torch.float32)
            prediction = model(batch).numpy()
            if prediction.shape != (stop - start,) or not np.isfinite(prediction).all():
                raise ValueError("checkpoint inference produced an invalid chunk")
            restored[start:stop] = prediction.astype(np.float32, copy=False)
    if restored.size != expected_count or not np.isfinite(restored).all():
        raise ValueError("restored vector has invalid values or exact count")
    _atomic_save_npy(npy_output, restored)
    _write_csv(csv_output, restored)
    _verify_outputs(npy_output, csv_output, restored)
    metadata = {
        "schema_version": 2,
        "method": "schema2_scalar_residual_mlp",
        "shape": [int(restored.size)],
        "dtype": "float32",
        "checkpoint_sha256": _hash(checkpoint),
        "input_sha256": input_hashes,
        "split_sha256": split["labels_sha256"],
        "split_spec_sha256": split["spec_sha256"],
        "test_noisy_sha256": test_hash,
        "restored_npy_sha256": _hash(npy_output),
        "submission_csv_sha256": _hash(csv_output),
        "submission_rows": int(restored.size),
        "audit_metrics": None,
        "config": payload["config"],
    }
    metadata_path = npy_output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--npy-output", type=Path, default=DEFAULT_NPY)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.data_dir, args.npy_output, args.csv_output, chunk_size=args.chunk_size), sort_keys=True))


if __name__ == "__main__":
    main()
