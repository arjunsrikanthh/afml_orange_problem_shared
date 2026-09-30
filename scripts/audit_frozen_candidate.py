#!/usr/bin/env python3
"""Run the one-time audit for an already frozen Team 23 candidate.

The freeze document and every referenced input/artifact/source hash are checked
before either held-out target is indexed.  This module deliberately has no
training or selection path; the only production entry point is ``main``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = {
    "boilerplate.ipynb", "test_queries.csv", "test_weights_noisy.npy",
    "train_parallel_corpus.csv", "train_weights_clean.npy", "train_weights_noisy.npy",
}
RECIPE = {
    "scalar": {"seed": 2301, "epochs": 2, "hidden": 32, "batch_size": 4096,
               "chunk_size": 65536, "learning_rate": 0.001, "weight_decay": 0.0},
    "translation": {"seed": 2301, "epochs": 20, "batch_size": 8,
                    "learning_rate": 0.0001, "device": "mps"},
    "decoding": {"method": "greedy_argmax", "max_len": 25},
}
EXPECTED_AUDIT_FILES = {
    "scalar_checkpoint", "restored_weights", "translation_checkpoint", "translation_split"
}
EXPECTED_PARAMETER_COUNT = 2_961_630
SUPPORTED_RUNTIME_MODULES = {
    "scripts/audit_frozen_candidate.py", "scripts/compare_weight_candidates.py",
    "scripts/export_scalar_weights.py", "scripts/make_translation_split.py",
    "scripts/team23_tokenizer.py", "scripts/train_scalar_mlp.py",
    "scripts/train_translation.py", "scripts/translation_checkpoint.py",
    "scripts/translation_noisy_baseline.py", "scripts/validate_translation_split.py",
    "scripts/weight_baselines.py",
}
TRANSLATION_TRAINING_RUNTIME = {
    "scripts/train_translation.py", "scripts/translation_noisy_baseline.py",
    "scripts/team23_tokenizer.py", "scripts/validate_translation_split.py",
    "scripts/translation_checkpoint.py", "scripts/make_translation_split.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("freeze paths must be nonempty repository-relative paths")
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("freeze path escapes the repository")
    return result


def validate_freeze_document(payload: Any) -> dict[str, Any]:
    """Validate the immutable, exact contract portion of a freeze document."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("status") != "frozen":
        raise ValueError("freeze must have schema_version 1 and status 'frozen'")
    if set(payload.get("input_sha256", {})) != RAW_FILES:
        raise ValueError("freeze input_sha256 must name exactly the six raw Team 23 files")
    for name, digest in payload["input_sha256"].items():
        _hash(digest, f"input_sha256[{name}]")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != EXPECTED_AUDIT_FILES:
        raise ValueError("freeze artifacts are incomplete")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"artifact contract is invalid: {name}")
        _hash(record["sha256"], f"artifacts[{name}].sha256")
    runtime = payload.get("runtime_sha256")
    if not isinstance(runtime, dict) or set(runtime) != SUPPORTED_RUNTIME_MODULES:
        raise ValueError("runtime_sha256 must contain exactly the supported runtime module set")
    for name, digest in runtime.items():
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError("runtime source paths must be repository-relative")
        _hash(digest, f"runtime_sha256[{name}]")
    recipe = payload.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("freeze recipe is missing")
    for section, required in RECIPE.items():
        if not isinstance(recipe.get(section), dict) or set(recipe[section]) != set(required):
            raise ValueError(f"freeze recipe section is missing or incomplete: {section}")
        for key, expected in required.items():
            if type(recipe[section].get(key)) is not type(expected) or recipe[section].get(key) != expected:
                raise ValueError(f"freeze recipe value is invalid: {section}.{key}")
    return payload


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"this frozen recipe has already been consumed: {path}") from exc


def start_audit(output_dir: Path, freeze_path: Path, *, root: Path = ROOT) -> tuple[Path, dict[str, Any]]:
    """Atomically claim a freeze globally before held-out target access."""
    output_dir = output_dir.resolve()
    root = root.resolve()
    data_dir = root / "data" / "23_Team_Toxic"
    if output_dir == data_dir or output_dir.is_relative_to(data_dir):
        raise ValueError("audit output cannot be inside the raw data directory")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit directory: {output_dir}")
    freeze_hash = sha256_file(freeze_path)
    ledger_path = root / "results" / "audit_ledger" / f"{freeze_hash}.json"
    ledger = {
        "schema_version": 1, "status": "started", "freeze_sha256": freeze_hash,
        "freeze": str(freeze_path.resolve()),
        "output_dir": str(output_dir), "started_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    _exclusive_json(ledger_path, ledger)
    return ledger_path, ledger


def _record_failure(ledger_path: Path, ledger: dict[str, Any], exc: BaseException) -> None:
    ledger = dict(ledger)
    ledger.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    _write_json(ledger_path, ledger)


def validate_referenced_files(payload: dict[str, Any], root: Path) -> dict[str, Path]:
    """Hash-check all freeze references before any audit target is indexed."""
    paths: dict[str, Path] = {}
    for name in RAW_FILES:
        path = root / "data" / "23_Team_Toxic" / name
        if not path.is_file() or sha256_file(path) != payload["input_sha256"][name]:
            raise ValueError(f"raw input hash mismatch or missing file: {name}")
        paths[name] = path
    for name, record in payload["artifacts"].items():
        path = _path(root, record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch or missing file: {name}")
        paths[name] = path
    for name, digest in payload["runtime_sha256"].items():
        path = _path(root, name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"runtime source hash mismatch or missing file: {name}")
    return paths


def audit_model_rmse(model: torch.nn.Module, noisy: np.ndarray, clean: np.ndarray, labels: np.ndarray, *, device: torch.device | None = None, chunk_size: int = 65_536) -> dict[str, Any]:
    """Score a scalar model on full-array label-2 positions without other targets."""
    noisy = np.asarray(noisy)
    labels = np.asarray(labels)
    if noisy.ndim != 1 or clean.ndim != 1 or labels.ndim != 1 or noisy.shape != clean.shape or noisy.shape != labels.shape:
        raise ValueError("scalar audit arrays must be aligned one-dimensional arrays")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if hasattr(model, "eval"):
        model.eval()
    audit = labels == 2
    total_sse = 0.0
    count = 0
    with torch.no_grad():
      for start in range(0, noisy.size, chunk_size):
        stop = min(start + chunk_size, noisy.size)
        indices = np.flatnonzero(audit[start:stop]) + start
        if indices.size == 0:
            continue
        x = np.asarray(noisy[indices], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("scalar audit noisy inputs are non-finite")
        prediction = model(torch.from_numpy(np.array(x, copy=True)).to(device=device, dtype=torch.float32))
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().to(device="cpu").numpy().reshape(-1)
        prediction = np.asarray(prediction)
        if prediction.shape != (indices.size,) or not np.isfinite(prediction).all():
            raise ValueError("scalar audit model output is invalid")
        error = prediction.astype(np.float64) - np.asarray(clean[indices], dtype=np.float64)
        if not np.isfinite(error).all():
            raise ValueError("scalar audit error is non-finite")
        total_sse += float(np.dot(error, error))
        count += int(error.size)
    return {"rmse": (float(np.sqrt(total_sse / count)) if count else None), "sse": total_sse, "count": count, "empty": count == 0}


def validate_translation_init(payload: dict[str, Any], *, expected_hashes: dict[str, str], expected_split_ids: dict[str, list[str]], special_ids: dict[str, int], expected_init_hash: str) -> None:
    """Validate a complete frozen translation checkpoint and its initialization lineage."""
    from scripts.translation_checkpoint import validate_checkpoint
    validate_checkpoint(payload, input_hashes=expected_hashes, split_ids=expected_split_ids, parameter_count=EXPECTED_PARAMETER_COUNT, special_ids=special_ids)
    if payload.get("input_hashes", {}).get("weights") != expected_init_hash:
        raise ValueError("translation checkpoint initialization hash does not match restored scalar artifact")
    if payload.get("epoch") != 20 or payload.get("step") != 540:
        raise ValueError("translation checkpoint must be complete through epoch 20 at step 540")
    history = payload.get("history")
    if not isinstance(history, list) or [item.get("epoch") for item in history] != list(range(1, 21)) or [item.get("step") for item in history] != [27 * epoch for epoch in range(1, 21)]:
        raise ValueError("translation checkpoint history must contain epochs 1 through 20 at steps 27 through 540")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("history", {}).get("complete") is not True or lineage.get("history", {}).get("gaps") != []:
        raise ValueError("translation checkpoint lineage does not prove complete history")


def _production_translation_audit(paths: dict[str, Path], split_payload: dict[str, Any], device_name: str, output_dir: Path) -> dict[str, Any]:
    from scripts.translation_noisy_baseline import EXPECTED_PARAMETER_COUNT, evaluate, greedy_decode, prepare_model
    from scripts.team23_tokenizer import load_tokenizers
    checkpoint = torch.load(paths["translation_checkpoint"], map_location="cpu", weights_only=False)
    source_tok, target_tok = load_tokenizers(paths["boilerplate.ipynb"])
    model = prepare_model(np.load(paths["restored_weights"], allow_pickle=False), torch.device(device_name))
    model.load_state_dict(checkpoint["model"], strict=True)
    rows_by_id: dict[str, dict[str, str]] = {}
    with paths["train_parallel_corpus.csv"].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_id[row["id"]] = row
    ids = list(split_payload["splits"]["audit"])
    if len(ids) != len(set(ids)) or any(identifier not in rows_by_id for identifier in ids):
        raise ValueError("translation audit IDs are invalid or missing")
    predictions: list[str] = []
    references: list[str] = []
    diagnostics = {"empty_predictions": 0, "cap_terminated": 0, "eos_terminated": 0, "repetitive": 0}
    for identifier in ids:
        row = rows_by_id[identifier]
        sequence = greedy_decode(model, source_tok.encode(row["asdfghjkl"]), bos_id=target_tok.bos_id, eos_id=target_tok.eos_id, pad_id=target_tok.pad_id, max_len=25, device=model.src_embedding.weight.device)
        prediction = target_tok.decode(sequence, skip_special_tokens=True)
        predictions.append(prediction)
        references.append(row["english"])
        if not prediction.strip(): diagnostics["empty_predictions"] += 1
        if target_tok.eos_id in sequence[1:]: diagnostics["eos_terminated"] += 1
        else: diagnostics["cap_terminated"] += 1
        tokens = prediction.split()
        if tokens and max(tokens.count(token) for token in set(tokens)) / len(tokens) >= 0.5: diagnostics["repetitive"] += 1
    metrics = evaluate(predictions, references)
    csv_path = output_dir / "translation_audit.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "source", "reference", "prediction"))
        writer.writerows((i, rows_by_id[i]["asdfghjkl"], rows_by_id[i]["english"], p) for i, p in zip(ids, predictions))
    return {"ids": ids, "metrics": metrics, "diagnostics": diagnostics, "csv_sha256": sha256_file(csv_path)}


def _validate_requested_device(name: str) -> torch.device:
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid translation device: {name}") from exc
    available = {
        "cpu": True,
        "mps": bool(torch.backends.mps.is_available()),
        "cuda": bool(torch.cuda.is_available()),
    }
    if device.type not in available or not available[device.type]:
        raise ValueError(f"requested translation device is unavailable: {name}")
    return device


def _preflight(freeze: Path, *, root: Path, translation_device: str) -> tuple[dict[str, Any], dict[str, Path], torch.nn.Module, dict[str, Any], torch.device]:
    """Validate metadata, hashes, contracts, split IDs, and recovery state."""
    payload = validate_freeze_document(json.loads(freeze.read_text(encoding="utf-8")))
    paths = validate_referenced_files(payload, root)
    requested_device = _validate_requested_device(translation_device)
    from scripts.export_scalar_weights import _load_validated_checkpoint
    clean = np.load(paths["train_weights_clean.npy"], mmap_mode="r", allow_pickle=False)
    scalar_payload, scalar_model, _, _ = _load_validated_checkpoint(paths["scalar_checkpoint"], paths["train_weights_clean.npy"].parent, int(clean.size))
    if scalar_payload.get("epoch") != 2 or scalar_payload.get("config", {}).get("epochs") != 2:
        raise ValueError("scalar checkpoint must be complete through epoch 2")
    for key, expected in payload["recipe"]["scalar"].items():
        if key not in scalar_payload.get("config", {}) or scalar_payload["config"][key] != expected:
            raise ValueError(f"scalar checkpoint config differs from frozen recipe: {key}")
    restored = np.load(paths["restored_weights"], mmap_mode="r", allow_pickle=False)
    if restored.dtype != np.float32 or restored.shape != (EXPECTED_PARAMETER_COUNT,) or not np.isfinite(restored).all():
        raise ValueError("restored scalar artifact has an invalid shape, dtype, or value")
    split_payload = json.loads(paths["translation_split"].read_text(encoding="utf-8"))
    from scripts.validate_translation_split import validate_frozen_split
    validate_frozen_split(paths["train_parallel_corpus.csv"], paths["translation_split"])
    split_ids = split_payload.get("splits")
    if not isinstance(split_ids, dict) or set(split_ids) != {"fitting", "development", "audit"} or any(not isinstance(v, list) or len(v) != len(set(v)) for v in split_ids.values()):
        raise ValueError("translation split IDs are incomplete or duplicated")
    groups = [set(split_ids[name]) for name in ("fitting", "development", "audit")]
    if any(groups[left] & groups[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("translation split IDs overlap")
    from scripts.team23_tokenizer import load_tokenizers
    source_tok, target_tok = load_tokenizers(paths["boilerplate.ipynb"])
    checkpoint_payload = torch.load(paths["translation_checkpoint"], map_location="cpu", weights_only=False)
    expected_hashes = {"notebook": payload["input_sha256"]["boilerplate.ipynb"], "corpus": payload["input_sha256"]["train_parallel_corpus.csv"], "split": payload["artifacts"]["translation_split"]["sha256"], "weights": payload["artifacts"]["restored_weights"]["sha256"]}
    validate_translation_init(checkpoint_payload, expected_hashes=expected_hashes, expected_split_ids=split_ids, special_ids={"source_pad": source_tok.pad_id, "target_pad": target_tok.pad_id, "target_bos": target_tok.bos_id, "target_eos": target_tok.eos_id}, expected_init_hash=payload["artifacts"]["restored_weights"]["sha256"])
    for key, expected in payload["recipe"]["translation"].items():
        if key == "epochs":
            continue  # Proven by payload.epoch, step, and complete history above.
        if key == "device":
            if expected != str(requested_device):
                raise ValueError("translation recipe device differs from requested device")
            continue
        if key in {"seed", "batch_size", "learning_rate"} and (key not in checkpoint_payload.get("config", {}) or checkpoint_payload["config"][key] != expected):
            raise ValueError(f"translation checkpoint config differs from frozen recipe: {key}")
        if key in checkpoint_payload.get("config", {}) and checkpoint_payload["config"][key] != expected:
            raise ValueError(f"translation checkpoint config differs from frozen recipe: {key}")
    source_hashes = checkpoint_payload.get("lineage", {}).get("source", {}).get("file_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != TRANSLATION_TRAINING_RUNTIME:
        raise ValueError("translation checkpoint source lineage is missing or incomplete")
    for name, digest in source_hashes.items():
        if payload["runtime_sha256"].get(name) != digest:
            raise ValueError(f"translation checkpoint source hash differs from frozen runtime: {name}")
    # Strictly load the restored architecture and frozen checkpoint before the
    # one-time ledger claim; this performs no forward pass or target access.
    from scripts.translation_noisy_baseline import prepare_model
    restored_for_model = np.load(paths["restored_weights"], allow_pickle=False)
    translation_model = prepare_model(restored_for_model, requested_device)
    translation_model.load_state_dict(checkpoint_payload["model"], strict=True)
    return payload, paths, scalar_model, split_payload, requested_device


def run_audit(freeze: Path, output_dir: Path, *, root: Path = ROOT, device: str = "cpu", translation_runner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    # Bad metadata/hash/recovery input is rejected before a claim or target access.
    payload, paths, scalar_model, split_payload, translation_device = _preflight(freeze, root=root, translation_device=device)
    ledger_path, ledger = start_audit(output_dir, freeze, root=root)
    output_dir = output_dir.resolve()
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
        _write_json(ledger_path, {**ledger, "output_created": str(output_dir)})
        clean = np.load(paths["train_weights_clean.npy"], mmap_mode="r", allow_pickle=False)
        noisy = np.load(paths["train_weights_noisy.npy"], mmap_mode="r", allow_pickle=False)
        labels = np.asarray(__import__("scripts.train_scalar_mlp", fromlist=["split_labels"]).split_labels(int(clean.size))[0])
        # This is the first point at which clean audit positions are indexed.
        scalar_model.to(torch.device("cpu"))
        scalar_metrics = audit_model_rmse(scalar_model, noisy, clean, labels, device=torch.device("cpu"), chunk_size=65_536)
        translation = (translation_runner or _production_translation_audit)(paths, split_payload, str(translation_device), output_dir)
        report = {"schema_version": 1, "status": "completed", "freeze": payload, "hashes": {"raw": payload["input_sha256"], "artifacts": {k: v["sha256"] for k, v in payload["artifacts"].items()}, "runtime": payload["runtime_sha256"]}, "scalar": scalar_metrics, "translation": translation, "environment": {"python": platform.python_version(), "torch": torch.__version__, "scalar_device": "cpu", "translation_device": str(translation_device)}}
        _write_json(output_dir / "audit_report.json", report)
        ledger.update({"status": "completed", "report_sha256": sha256_file(output_dir / "audit_report.json")})
        _write_json(ledger_path, ledger)
        return report
    except BaseException as exc:
        _record_failure(ledger_path, ledger, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_audit(args.freeze, args.output_dir, device=args.device)
    print(json.dumps({"status": report["status"], "scalar": report["scalar"], "translation": report["translation"]["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
