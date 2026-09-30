#!/usr/bin/env python3
"""Bounded fine-tuning for the supplied Team 23 translation Transformer.

The trainer reads fitting IDs for optimization and development IDs for loss
validation.  It never loads audit rows and does not perform generation or
compute translation metrics.  Use ``scripts/run_job.py`` to bound a real run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.translation_noisy_baseline import (
    DEFAULT_CORPUS,
    DEFAULT_NOTEBOOK,
    DEFAULT_SPLIT,
    DEFAULT_WEIGHTS,
    EXPECTED_PARAMETER_COUNT,
    TranslationTransformer,
    prepare_model,
    sha256_file,
)
from scripts.team23_tokenizer import Team23Tokenizer, load_tokenizers
from scripts.validate_translation_split import validate_frozen_split


DEFAULT_OUTPUT = ROOT / "results" / "training" / "translation"
TRAINER_VERSION = "translation-trainer-v1"


def source_provenance() -> dict[str, Any]:
    """Return reproducible source identity without requiring Git to be present."""
    revision: str | None = None
    git_status = "unavailable"
    dirty: bool | None = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        revision = completed.stdout.strip() or None
        if revision is not None:
            git_status = "available"
            dirty = (
                subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode != 0
                or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode != 0
            )
    except (OSError, subprocess.CalledProcessError):
        pass
    source_files = {}
    for relative in (
        "scripts/train_translation.py",
        "scripts/translation_noisy_baseline.py",
        "scripts/team23_tokenizer.py",
        "scripts/validate_translation_split.py",
        "scripts/translation_checkpoint.py",
        "scripts/make_translation_split.py",
    ):
        path = ROOT / relative
        if path.is_file():
            source_files[relative] = sha256_file(path)
    return {
        "git_revision": revision,
        "git_status": git_status,
        "dirty": dirty,
        "file_sha256": source_files,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_batch(sequences: Sequence[Sequence[int]], pad_id: int, device: torch.device) -> torch.Tensor:
    if not sequences:
        raise ValueError("cannot pad an empty batch")
    width = max(len(sequence) for sequence in sequences)
    if width < 1:
        raise ValueError("sequences must contain at least one token")
    output = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        output[row, : len(sequence)] = torch.as_tensor(sequence, dtype=torch.long, device=device)
    return output


def shifted_targets(target_ids: Sequence[int], pad_id: int) -> tuple[list[int], list[int]]:
    """Return decoder input and next-token labels for one BOS/EOS sequence."""
    if len(target_ids) < 2:
        raise ValueError("target sequence must contain at least BOS and EOS")
    decoder_input = list(target_ids[:-1])
    labels = list(target_ids[1:])
    if not labels or all(token == pad_id for token in labels):
        raise ValueError("target labels contain no non-pad token")
    return decoder_input, labels


def make_batch(
    examples: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    src_pad_id: int,
    tgt_pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot make an empty batch")
    sources = [list(source) for source, _ in examples]
    decoder_inputs: list[list[int]] = []
    labels: list[list[int]] = []
    for _, target in examples:
        decoder_input, label = shifted_targets(target, tgt_pad_id)
        decoder_inputs.append(decoder_input)
        labels.append(label)
    return (
        pad_batch(sources, src_pad_id, device),
        pad_batch(decoder_inputs, tgt_pad_id, device),
        pad_batch(labels, tgt_pad_id, device),
    )


def teacher_forced_loss(
    model: torch.nn.Module,
    examples: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    src_pad_id: int,
    tgt_pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    src, decoder_input, labels = make_batch(
        examples, src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id, device=device
    )
    logits = model(src, decoder_input, src_pad_idx=src_pad_id, tgt_pad_idx=tgt_pad_id)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=tgt_pad_id)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite teacher-forced loss")
    return loss


def iter_batches(
    examples: Sequence[tuple[Sequence[int], Sequence[int]]],
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
    max_batches: int | None = None,
) -> Iterable[list[tuple[Sequence[int], Sequence[int]]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    order = list(range(len(examples)))
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(order), generator=generator).tolist()
    emitted = 0
    for start in range(0, len(order), batch_size):
        if max_batches is not None and emitted >= max_batches:
            break
        indices = order[start : start + batch_size]
        yield [examples[index] for index in indices]
        emitted += 1


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _validate_finite_tree(value: Any, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint contains non-finite tensor: {path}")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"checkpoint contains non-finite value: {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{path}[{index}]")


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch"}
    if not required.issubset(state):
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    elif "cuda" in state:
        raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
    if "mps" in state:
        if not (torch.backends.mps.is_available() and hasattr(torch.mps, "set_rng_state")):
            raise ValueError("checkpoint contains MPS RNG state but MPS is unavailable")
        torch.mps.set_rng_state(state["mps"])


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    step: int,
    config: dict[str, Any],
    input_hashes: dict[str, str],
    split_ids: dict[str, list[str]],
    lineage: dict[str, Any] | None = None,
    history: list[dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    payload = {
        "contract": TRAINER_VERSION,
        "model": _cpu_tree(model.state_dict()),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": _cpu_tree(scheduler.state_dict()),
        "epoch": epoch,
        "step": step,
        "rng": _cpu_tree(capture_rng_state()),
        "config": config,
        "input_hashes": input_hashes,
        "split_ids": split_ids,
    }
    if lineage is not None:
        payload["lineage"] = lineage
    if history is not None:
        payload["history"] = history
    return payload


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    required = {"contract", "model", "optimizer", "scheduler", "epoch", "step", "rng", "config", "input_hashes", "split_ids"}
    if not required.issubset(payload) or payload["contract"] != TRAINER_VERSION:
        raise ValueError("refusing to save an incomplete checkpoint contract")
    _atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    input_hashes: dict[str, str],
    split_ids: dict[str, list[str]],
    device: torch.device,
) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"contract", "model", "optimizer", "scheduler", "epoch", "step", "rng", "config", "input_hashes", "split_ids"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("checkpoint is missing required contract fields")
    if checkpoint["contract"] != TRAINER_VERSION:
        raise ValueError("checkpoint trainer contract version mismatch")
    if checkpoint["config"] != config or checkpoint["input_hashes"] != input_hashes or checkpoint["split_ids"] != split_ids:
        raise ValueError("checkpoint inputs, split IDs, or configuration do not match current run")
    if (type(checkpoint["epoch"]) is not int or type(checkpoint["step"]) is not int
            or checkpoint["epoch"] < 0 or checkpoint["step"] < 0):
        raise ValueError("checkpoint epoch and step must be nonnegative integers")
    _validate_finite_tree(checkpoint["model"], "model")
    _validate_finite_tree(checkpoint["optimizer"], "optimizer")
    _validate_finite_tree(checkpoint["scheduler"], "scheduler")
    scheduler_state = checkpoint["scheduler"]
    if (not isinstance(scheduler_state, dict)
            or type(scheduler_state.get("last_epoch")) is not int
            or ("_step_count" in scheduler_state and type(scheduler_state["_step_count"]) is not int)
            or ("_step_count" in scheduler_state and scheduler_state["_step_count"] < 0)):
        raise ValueError("checkpoint scheduler progress is invalid")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    restore_rng_state(checkpoint["rng"])
    return checkpoint["epoch"], checkpoint["step"]


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    return checkpoint


def _assess_history(
    candidate: Any,
    *,
    parent_epoch: int,
    parent_step: int,
    source: str,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Keep only ordered records proven to belong to the parent checkpoint."""
    records: list[dict[str, float | int]] = []
    issues: list[str] = []
    if not isinstance(candidate, list):
        candidate = []
        issues.append("history is absent")
    previous_epoch = 0
    previous_step = 0
    for index, record in enumerate(candidate):
        if not isinstance(record, dict):
            issues.append(f"record {index} is not an object")
            continue
        epoch, step = record.get("epoch"), record.get("step")
        losses = (record.get("train_loss"), record.get("development_loss"))
        if type(epoch) is not int or type(step) is not int or epoch < 1 or step < 1:
            issues.append(f"record {index} has invalid progress")
            continue
        if any(not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(float(loss)) for loss in losses):
            issues.append(f"record {index} has invalid loss")
            continue
        if epoch > parent_epoch or step > parent_step:
            continue
        if epoch <= previous_epoch or step <= previous_step:
            issues.append(f"record {index} is out of order")
            continue
        records.append(dict(record))
        previous_epoch, previous_step = epoch, step
    observed = {int(record["epoch"]) for record in records}
    missing = [epoch for epoch in range(1, parent_epoch + 1) if epoch not in observed]
    if missing:
        issues.append("missing epochs: " + ",".join(str(epoch) for epoch in missing))
    return records, {"source": source, "complete": not issues, "gaps": issues}


def _parent_history(resume: Path, *, parent_epoch: int, parent_step: int) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Recover bounded history from a checkpoint, or a legacy sibling summary."""
    checkpoint = _read_checkpoint(resume)
    history = checkpoint.get("history")
    if history is not None:
        return _assess_history(history, parent_epoch=parent_epoch, parent_step=parent_step, source="checkpoint")
    summary_path = resume.parent / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [], {"source": "summary", "complete": False, "gaps": ["summary is unreadable"]}
        summary_history = summary.get("history")
        return _assess_history(summary_history, parent_epoch=parent_epoch, parent_step=parent_step, source="summary")
    return [], {"source": "none", "complete": parent_epoch == 0, "gaps": [] if parent_epoch == 0 else ["history source is absent"]}


def _parent_summary(resume: Path) -> dict[str, str] | None:
    summary_path = resume.parent / "summary.json"
    if not summary_path.is_file():
        return None
    return {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)}


def prepare_output_dir(output_dir: Path, resume: Path | None) -> None:
    """Reject ambiguous output reuse while allowing documented continuation."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    entries = {entry.name for entry in output_dir.iterdir()}
    if not entries:
        return
    if resume is not None:
        try:
            same_directory = resume.resolve().parent == output_dir.resolve()
        except OSError:
            same_directory = False
        if same_directory and resume.name == "checkpoint_last.pt" and entries <= {"checkpoint_last.pt", "summary.json"}:
            return
    raise FileExistsError(
        f"refusing to overwrite non-empty output directory: {output_dir}; use a fresh directory"
    )


def validate_resume_target(epochs: int, saved_epoch: int) -> None:
    if type(saved_epoch) is not int or saved_epoch < 0:
        raise ValueError("saved checkpoint epoch must be a nonnegative integer")
    if epochs <= saved_epoch:
        raise ValueError("epochs must be greater than the saved checkpoint epoch for a continuation")


def _load_data(
    corpus: Path, split_path: Path, notebook: Path, weights: Path
) -> tuple[pd.DataFrame, dict[str, Any], Team23Tokenizer, Team23Tokenizer, dict[str, str]]:
    split = validate_frozen_split(corpus, split_path)
    if sha256_file(corpus) != split["source_corpus_sha256"]:
        raise ValueError("split corpus hash does not match corpus")
    split_ids = {name: list(split["splits"][name]) for name in ("fitting", "development", "audit")}
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    if any(len(split_sets[name]) != len(split_ids[name]) for name in split_ids):
        raise ValueError("split contains duplicate IDs")
    if any(split_sets[left] & split_sets[right] for left, right in (("fitting", "development"), ("fitting", "audit"), ("development", "audit"))):
        raise ValueError("split IDs overlap")
    selected_ids = split_sets["fitting"] | split_sets["development"]
    selected_rows: list[dict[str, str]] = []
    corpus_ids: set[str] = set()
    with corpus.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "asdfghjkl", "english"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("corpus is missing required columns")
        for row in reader:
            identifier = row["id"]
            if identifier in corpus_ids:
                raise ValueError("corpus contains duplicate IDs")
            corpus_ids.add(identifier)
            if identifier in selected_ids:
                if not row["asdfghjkl"] or not row["english"]:
                    raise ValueError("selected corpus row has an empty source or target")
                selected_rows.append({key: row[key] for key in required})
    if set().union(*split_sets.values()) != corpus_ids or len(selected_rows) != len(selected_ids):
        raise ValueError("split IDs are not exhaustive or a selected row is missing")
    by_id = pd.DataFrame(selected_rows).set_index("id")
    source_tokenizer, target_tokenizer = load_tokenizers(notebook)
    hashes = {
        "notebook": sha256_file(notebook),
        "corpus": sha256_file(corpus),
        "split": sha256_file(split_path),
        "weights": sha256_file(weights),
    }
    return by_id, split, source_tokenizer, target_tokenizer, hashes


def _examples(rows: pd.DataFrame, ids: Sequence[str], source_tokenizer: Team23Tokenizer, target_tokenizer: Team23Tokenizer) -> list[tuple[list[int], list[int]]]:
    output = []
    for identifier in ids:
        row = rows.loc[identifier]
        output.append((source_tokenizer.encode(str(row["asdfghjkl"])), target_tokenizer.encode(str(row["english"]))))
    return output


def load_checked_weights(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (EXPECTED_PARAMETER_COUNT,) or not np.isfinite(values).all():
        raise ValueError("translation initialization must be a finite float32 vector of the exact model length")
    if path.resolve() == DEFAULT_WEIGHTS.resolve():
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if sha256_file(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError("supplied noisy test weights differ from the preflight manifest")
    return values


def run_training(
    *,
    notebook: Path = DEFAULT_NOTEBOOK,
    corpus: Path = DEFAULT_CORPUS,
    weights: Path = DEFAULT_WEIGHTS,
    split: Path = DEFAULT_SPLIT,
    output_dir: Path = DEFAULT_OUTPUT,
    device_name: str | None = None,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 2301,
    smoke: bool = False,
    smoke_batches: int = 2,
    resume: Path | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or smoke_batches < 1:
        raise ValueError("epochs, batch size, smoke batches, and learning rate must be positive")
    prepare_output_dir(output_dir, resume)
    rows, split_payload, source_tokenizer, target_tokenizer, hashes = _load_data(corpus, split, notebook, weights)
    split_ids = {name: list(split_payload["splits"][name]) for name in ("fitting", "development", "audit")}
    fitting = _examples(rows, split_ids["fitting"], source_tokenizer, target_tokenizer)
    development = _examples(rows, split_ids["development"], source_tokenizer, target_tokenizer)
    if smoke:
        fitting = fitting[: batch_size * smoke_batches]
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))
    seed_everything(seed)
    model = prepare_model(load_checked_weights(weights), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = {
        "trainer": TRAINER_VERSION,
        "architecture": "supplied_TranslationTransformer_v1",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "smoke": smoke,
        "smoke_batches": smoke_batches,
        "device": str(device),
        "special_ids": {"source_pad": source_tokenizer.pad_id, "target_pad": target_tokenizer.pad_id, "target_bos": target_tokenizer.bos_id, "target_eos": target_tokenizer.eos_id},
        "nested_tensor_disabled_on_mps": device.type == "mps",
    }
    start_epoch, step = 0, 0
    checkpoint_path = output_dir / "checkpoint_last.pt"
    source = source_provenance()
    parent_checkpoint: dict[str, Any] | None = None
    parent_summary: dict[str, str] | None = None
    history: list[dict[str, float | int]] = []
    history_status: dict[str, Any] = {"source": "new-run", "complete": True, "gaps": []}
    if resume is not None:
        parent_payload = _read_checkpoint(resume)
        if type(parent_payload.get("step")) is not int or parent_payload["step"] < 0:
            raise ValueError("parent checkpoint epoch and step must be valid integers")
        validate_resume_target(epochs, parent_payload.get("epoch"))
        parent_checkpoint = {
            "path": str(resume.resolve()),
            "sha256": sha256_file(resume),
            "epoch": parent_payload["epoch"],
            "step": parent_payload["step"],
        }
        parent_summary = _parent_summary(resume)
        history, history_status = _parent_history(
            resume,
            parent_epoch=parent_payload["epoch"],
            parent_step=parent_payload["step"],
        )
        start_epoch, step = load_checkpoint(resume, model=model, optimizer=optimizer, scheduler=scheduler, config=config, input_hashes=hashes, split_ids=split_ids, device=device)
    model.train()
    run_history: list[dict[str, float | int]] = []
    lineage = {
        "source": source,
        "parent_checkpoint": parent_checkpoint,
        "parent_summary": parent_summary,
        "history": history_status,
    }
    for epoch in range(start_epoch, epochs):
        train_loss = 0.0
        train_batches = 0
        for batch in iter_batches(fitting, batch_size, seed=seed, epoch=epoch, shuffle=True, max_batches=smoke_batches if smoke else None):
            optimizer.zero_grad(set_to_none=True)
            loss = teacher_forced_loss(model, batch, src_pad_id=source_tokenizer.pad_id, tgt_pad_id=target_tokenizer.pad_id, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1
            step += 1
        if not train_batches:
            raise ValueError("fitting split produced no training batches")
        model.eval()
        with torch.no_grad():
            validation_loss = float(teacher_forced_loss(model, development, src_pad_id=source_tokenizer.pad_id, tgt_pad_id=target_tokenizer.pad_id, device=device).cpu())
        if not np.isfinite(validation_loss):
            raise FloatingPointError("non-finite development validation loss")
        model.train()
        record = {"epoch": epoch + 1, "step": step, "train_loss": train_loss / train_batches, "development_loss": validation_loss}
        history.append(record)
        run_history.append(record)
        save_checkpoint(checkpoint_payload(model, optimizer, scheduler, epoch=epoch + 1, step=step, config=config, input_hashes=hashes, split_ids=split_ids, lineage=lineage, history=history), checkpoint_path)
    result = {"trainer": TRAINER_VERSION, "epochs_completed": epochs, "step": step, "history": history, "run_history": run_history, "lineage": lineage, "checkpoint": str(checkpoint_path), "scope": {"fitting_ids": len(fitting), "development_ids": len(development), "audit_ids_loaded": 0}}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", dest="device_name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_training(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
