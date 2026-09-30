#!/usr/bin/env python3
"""Train a bounded scalar residual MLP on the fixed weight split.

The CLI is intentionally conservative: it reads only fitting and development
rows from the clean array, never reads clean audit rows, and writes checkpoints
atomically.  The module API is used by synthetic tests and by future bounded
jobs; this file does not launch a job when imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.weight_baselines import make_assignment


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "23_Team_Toxic"
DEFAULT_CLEAN = DATA / "train_weights_clean.npy"
DEFAULT_NOISY = DATA / "train_weights_noisy.npy"
DEFAULT_CHECKPOINT = ROOT / "results" / "checkpoints" / "scalar_mlp.pt"
# v1 optimized a broadcast B-by-B loss rather than aligned scalar pairs.
# Its optimizer trajectory cannot be resumed under the corrected objective.
SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_spec(total_length: int, *, seed: int = 2301, block_size: int = 4096, guard: int = 32) -> dict[str, Any]:
    if total_length <= 0 or total_length % 2:
        raise ValueError("weight length must be positive and evenly divisible into two halves")
    spec = {
        "rule": "sha256(weight-v1|seed|group_id) byte thresholds 0.60/0.80; same offset group in both halves",
        "seed": int(seed),
        "block_size": int(block_size),
        "guard": int(guard),
        "half_length": int(total_length // 2),
        "total_length": int(total_length),
    }
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    spec["spec_sha256"] = hashlib.sha256(encoded).hexdigest()
    return spec


def split_labels(total_length: int, *, seed: int = 2301, block_size: int = 4096, guard: int = 32) -> tuple[np.ndarray, dict[str, Any]]:
    spec = split_spec(total_length, seed=seed, block_size=block_size, guard=guard)
    labels = make_assignment(total_length, half_length=total_length // 2, seed=seed, block_size=block_size, guard=guard)
    counts = {name: int(np.count_nonzero(labels == code)) for name, code in (("fitting", 0), ("development", 1), ("audit", 2), ("guard", -1))}
    return labels, {**spec, "counts": counts}


class ScalarResidualMLP(torch.nn.Module):
    """Small scalar-to-scalar residual network."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        values = noisy.reshape(-1, 1)
        return (values + self.network(values)).reshape_as(noisy)


def paired_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Refuse silent pairwise broadcasting in the scalar regression loss."""
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("scalar prediction and target shapes must match exactly and be nonempty")
    loss = torch.mean((prediction - target) ** 2)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite scalar regression loss")
    return loss


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch"}
    if not required.issubset(state):
        raise ValueError("checkpoint is missing a required RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state:
        if not (torch.backends.mps.is_available() and hasattr(torch.mps, "set_rng_state")):
            raise ValueError("checkpoint contains MPS RNG state but MPS is unavailable")
        torch.mps.set_rng_state(state["mps"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_arrays(noisy: np.ndarray, clean: np.ndarray, labels: np.ndarray) -> None:
    if noisy.ndim != 1 or clean.ndim != 1 or labels.ndim != 1 or noisy.shape != clean.shape or noisy.shape != labels.shape:
        raise ValueError("noisy, clean, and labels must be aligned one-dimensional arrays")
    if noisy.dtype != np.float32 or clean.dtype != np.float32:
        raise ValueError("noisy and clean arrays must be float32")
    if not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError("split labels must be fitting, development, audit, or guard")
    # Preflight validates the complete files. This trainer must not inspect
    # clean audit values, even for its own finite-value gate.
    allowed = (labels == 0) | (labels == 1)
    if not np.isfinite(noisy).all() or not np.isfinite(clean[allowed]).all():
        raise ValueError("noisy and fitting/development clean values must be finite")


def _checkpoint_payload(
    model: ScalarResidualMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    step: int,
    config: dict[str, Any],
    input_hashes: dict[str, str],
    split: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "config": config,
        "input_sha256": input_hashes,
        "split": split,
        "rng": _rng_state(),
    }


def _validate_checkpoint(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    required = {"schema_version", "model", "optimizer", "scheduler", "epoch", "step", "config", "input_sha256", "split", "rng"}
    if not required.issubset(payload) or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("checkpoint schema is incomplete or unsupported")
    for key in ("config", "input_sha256", "split"):
        if payload[key] != expected[key]:
            raise ValueError(f"checkpoint {key} does not match the current run")
    if not isinstance(payload["epoch"], int) or not isinstance(payload["step"], int) or payload["epoch"] < 0 or payload["step"] < 0:
        raise ValueError("checkpoint progress counters are invalid")
    _restore_rng(payload["rng"])


def _load_checkpoint(path: Path, expected: dict[str, Any], model: ScalarResidualMLP, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> tuple[int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"unable to read checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    _validate_checkpoint(payload, expected)
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
    except Exception as exc:
        raise ValueError("checkpoint model or optimizer state is incompatible") from exc
    return int(payload["epoch"]), int(payload["step"])


def _device(name: str | None) -> torch.device:
    selected = name or ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(selected)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def _rmse(model: ScalarResidualMLP, noisy: np.ndarray, clean: np.ndarray, indices: np.ndarray, device: torch.device, chunk_size: int) -> float:
    if indices.size == 0:
        raise ValueError("development split is empty")
    model.eval()
    squared = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, indices.size, chunk_size):
            batch = indices[start : start + chunk_size]
            x = torch.from_numpy(np.asarray(noisy[batch])).to(device=device, dtype=torch.float32)
            prediction = model(x).detach().cpu().numpy().reshape(-1).astype(np.float64)
            error = prediction - np.asarray(clean[batch], dtype=np.float64)
            squared += float(np.dot(error, error))
            count += error.size
    return float(np.sqrt(squared / count))


def train(
    noisy: np.ndarray,
    clean: np.ndarray,
    labels: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
    checkpoint: Path | None = None,
    input_hashes: dict[str, str] | None = None,
    resume: bool = False,
    device_name: str | None = None,
    max_epochs_this_run: int | None = None,
) -> dict[str, Any]:
    """Train for the bounded config and return development-only metrics.

    ``clean`` is indexed only with fitting/development masks.  A caller may
    provide file hashes without loading complete arrays into memory.
    """
    _validate_arrays(noisy, clean, labels)
    settings = {
        "hidden": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 4096,
        "epochs": 2,
        "chunk_size": 65536,
        "seed": 2301,
    }
    if config:
        settings.update(config)
    if any(int(settings[key]) <= 0 for key in ("hidden", "batch_size", "epochs", "chunk_size")) or float(settings["learning_rate"]) <= 0:
        raise ValueError("training sizes and learning_rate must be positive")
    if max_epochs_this_run is not None and max_epochs_this_run <= 0:
        raise ValueError("max_epochs_this_run must be positive")
    labels = np.asarray(labels)
    fit_indices = np.flatnonzero(labels == 0)
    dev_indices = np.flatnonzero(labels == 1)
    if fit_indices.size == 0 or dev_indices.size == 0:
        raise ValueError("fitting and development splits must be non-empty")
    split = {
        **split_spec(int(labels.size)),
        "counts": {"fitting": int(fit_indices.size), "development": int(dev_indices.size), "audit": int(np.count_nonzero(labels == 2)), "guard": int(np.count_nonzero(labels == -1))},
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
    }
    expected = {"config": settings, "input_sha256": dict(input_hashes or {}), "split": split}
    device = _device(device_name)
    random.seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    model = ScalarResidualMLP(int(settings["hidden"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(settings["epochs"]))
    start_epoch = 0
    step = 0
    if resume:
        if checkpoint is None or not checkpoint.is_file():
            raise ValueError("resume requires an existing checkpoint")
        start_epoch, step = _load_checkpoint(checkpoint, expected, model, optimizer, scheduler)
    stop_epoch = min(int(settings["epochs"]), start_epoch + max_epochs_this_run) if max_epochs_this_run else int(settings["epochs"])
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        # Epoch-specific permutations make interrupted and uninterrupted runs identical.
        order = np.random.default_rng(int(settings["seed"]) + epoch).permutation(fit_indices.size)
        for start in range(0, fit_indices.size, int(settings["batch_size"])):
            positions = order[start : start + int(settings["batch_size"])]
            indices = fit_indices[positions]
            x = torch.from_numpy(np.asarray(noisy[indices])).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(np.asarray(clean[indices])).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = paired_mse(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1
        scheduler.step()
        if checkpoint is not None:
            _atomic_torch_save(_checkpoint_payload(model, optimizer, scheduler, epoch=epoch + 1, step=step, config=settings, input_hashes=dict(input_hashes or {}), split=split), checkpoint)
    return {"development_rmse_original_units": _rmse(model, noisy, clean, dev_indices, device, int(settings["chunk_size"])), "epoch": stop_epoch, "step": step, "device": str(device), "audit_metrics": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--noisy", type=Path, default=DEFAULT_NOISY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--max-epochs-this-run", type=int, default=1)
    args = parser.parse_args()
    noisy = np.load(args.noisy, mmap_mode="r", allow_pickle=False)
    clean = np.load(args.clean, mmap_mode="r", allow_pickle=False)
    labels, split = split_labels(int(noisy.size))
    hashes = {"clean": sha256_file(args.clean), "noisy": sha256_file(args.noisy)}
    config = {"epochs": args.epochs, "batch_size": args.batch_size, "hidden": args.hidden}
    result = train(noisy.reshape(-1), clean.reshape(-1), labels, config=config, checkpoint=args.checkpoint, input_hashes=hashes, resume=args.resume, device_name=args.device, max_epochs_this_run=args.max_epochs_this_run)
    print(json.dumps({"checkpoint": str(args.checkpoint), "split": split, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
