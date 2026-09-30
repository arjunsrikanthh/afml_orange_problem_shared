#!/usr/bin/env python3
"""Build the two standalone Team 23 submission notebooks.

The notebooks carry reviewed repository source as a small JSON source bundle.  At
runtime that bundle is unpacked into a fresh temporary ``scripts`` package, so the
notebooks have no import dependency on this checkout while retaining one canonical
implementation of the model and trainers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks"
SOURCE_MODULES = (
    "weight_baselines",
    "train_scalar_mlp",
    "export_scalar_weights",
    "preflight_team23",
    "make_translation_split",
    "validate_translation_split",
    "team23_tokenizer",
    "translation_checkpoint",
    "translation_noisy_baseline",
    "train_translation",
)
REQUIRED_FILES = (
    "boilerplate.ipynb",
    "train_weights_clean.npy",
    "train_weights_noisy.npy",
    "test_weights_noisy.npy",
    "train_parallel_corpus.csv",
    "test_queries.csv",
)
PINNED_INPUT_HASHES = {
    "boilerplate.ipynb": "d99a021e8747bb9b69248ed586c664eda02fb56aa1fa5472e287b770c5a73d6c",
    "train_weights_clean.npy": "aa8c6c8b743fc68cc0fa6dd7bb7a9690bbeace57894beec81d8f1d56a044edc4",
    "train_weights_noisy.npy": "e5292a4b2f0c43e789f7bb2558d98fc2f11b7e9734eed28e77a78efddeae395a",
    "test_weights_noisy.npy": "b751ffee729b96418148a7d7c684d3e8cc08ba46f0e225991f7a3811eb754397",
    "train_parallel_corpus.csv": "39f3fd2d9bfb2804e4643072d515af054f1f4c34dbc05f05fea04e9d5e167171",
    "test_queries.csv": "30ec797c5769b6a4e233a26aeaa3c38453e8645cc5db3045fcd08dc2a690e32f",
}
ROSTER_TEXT = (
    "Team 23 submission metadata is supplied privately at delivery time. "
    "Completed contributions and official membership must be confirmed by the "
    "team; no personal names, SRNs, or contribution assignments are stored in "
    "this shared repository."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle(root: Path = ROOT) -> tuple[dict[str, str], str]:
    bundle: dict[str, str] = {}
    for module in SOURCE_MODULES:
        path = root / "scripts" / f"{module}.py"
        if not path.is_file():
            raise FileNotFoundError(f"required source module is missing: {path}")
        bundle[f"scripts/{module}.py"] = path.read_text(encoding="utf-8")
    encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return bundle, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def discover_team23(
    *,
    search_roots: Iterable[Path],
    expected_hashes: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one Team 23 directory using pinned hashes and optional manifest."""
    expected = expected_hashes or PINNED_INPUT_HASHES
    candidates: set[Path] = set()
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_dir() and root.name == "23_Team_Toxic":
            candidates.add(root.resolve())
            continue
        if root.is_dir():
            for current, dirs, _files in os.walk(root):
                current_path = Path(current)
                if current_path.name == "23_Team_Toxic":
                    candidates.add(current_path.resolve())
                    dirs[:] = []
                else:
                    for name in dirs:
                        if name == "23_Team_Toxic":
                            candidates.add((current_path / name).resolve())
                    dirs[:] = [name for name in dirs if name != "23_Team_Toxic"]
    if len(candidates) != 1:
        raise ValueError(f"Team 23 dataset resolution expected exactly one candidate; found: {sorted(map(str, candidates))}")
    data_dir = next(iter(candidates))
    manifest_path = data_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    manifest_status = "manifest_unavailable"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError("Team 23 manifest is not valid JSON") from exc
        if manifest.get("team") != 23:
            raise ValueError("manifest team identity is not integer Team 23")
        if set(manifest.get("required_files", ())) != set(REQUIRED_FILES) or len(manifest.get("required_files", ())) != len(REQUIRED_FILES):
            raise ValueError("manifest required_files does not match the six-file Team 23 contract")
        manifest_status = "verified"
    observed: dict[str, str] = {}
    for name in REQUIRED_FILES:
        path = data_dir / name
        if not path.is_file():
            raise ValueError(f"Team 23 required file is missing: {name}")
        observed[name] = sha256_file(path)
        manifest_hash = manifest.get("files", {}).get(name, {}).get("sha256") if manifest else None
        if (manifest and manifest_hash != observed[name]) or expected.get(name) != observed[name]:
            raise ValueError(f"Team 23 input hash mismatch: {name}")
    return data_dir, {"team": 23, "manifest": str(manifest_path) if manifest else manifest_status, "manifest_status": manifest_status, "input_sha256": observed}


def _json_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _bootstrap(bundle: dict[str, str], bundle_hash: str) -> str:
    literal = repr(bundle)
    return f'''# Team 23 standalone runtime bootstrap; source bundle SHA256={bundle_hash}
import base64, csv, hashlib, json, os, platform, shutil, sys, tempfile
from pathlib import Path
SOURCE_BUNDLE = {literal}
SOURCE_BUNDLE_SHA256 = {bundle_hash!r}
REQUIRED_FILES = {list(REQUIRED_FILES)!r}
PINNED_INPUT_HASHES = {PINNED_INPUT_HASHES!r}
ROSTER_TEXT = {ROSTER_TEXT!r}
EXPECTED_WEIGHT_COUNT = 2_961_630
EXPECTED_TRAIN_COUNT = 5_923_260

def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _install_source_bundle():
    runtime = Path(tempfile.mkdtemp(prefix="team23_runtime_"))
    package = runtime / "scripts"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for relative, code in SOURCE_BUNDLE.items():
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(code, encoding="utf-8")
    sys.path.insert(0, str(runtime))
    return runtime

def _candidate_dirs():
    explicit = os.environ.get("TEAM23_DATA_DIR")
    if explicit:
        path = Path(explicit)
        if path.name != "23_Team_Toxic":
            raise RuntimeError("TEAM23_DATA_DIR must name a 23_Team_Toxic directory")
        return [path]
    roots = [Path.cwd(), Path("/kaggle/input")]
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        if root.name == "23_Team_Toxic":
            candidates.append(root)
            continue
        for current, dirs, _files in os.walk(root):
            current_path = Path(current)
            if current_path.name == "23_Team_Toxic":
                candidates.append(current_path)
                dirs[:] = []
            else:
                candidates.extend(current_path / name for name in dirs if name == "23_Team_Toxic")
                dirs[:] = [name for name in dirs if name != "23_Team_Toxic"]
    return candidates

def discover_team23_runtime():
    candidates = {{p.resolve() for p in _candidate_dirs() if p.exists()}}
    if len(candidates) != 1:
        raise RuntimeError("Team 23 discovery expected exactly one 23_Team_Toxic directory; candidates=" + repr(sorted(map(str, candidates))))
    data_dir = next(iter(candidates))
    manifest_path = data_dir / "manifest.json"
    manifest = None
    manifest_status = "manifest_unavailable"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("team") != 23 or set(manifest.get("required_files", ())) != set(REQUIRED_FILES) or len(manifest.get("required_files", ())) != len(REQUIRED_FILES):
            raise RuntimeError("Team 23 manifest identity or required file list is invalid")
        manifest_status = "verified"
    hashes = {{}}
    for name in REQUIRED_FILES:
        path = data_dir / name
        if not path.is_file():
            raise RuntimeError("missing Team 23 input: " + name)
        hashes[name] = _sha256(path)
        if manifest is not None and hashes[name] != manifest.get("files", {{}}).get(name, {{}}).get("sha256"):
            raise RuntimeError("manifest hash mismatch: " + name)
        if hashes[name] != PINNED_INPUT_HASHES[name]:
            raise RuntimeError("pinned Team 23 hash mismatch: " + name)
    runtime_manifest = runtime_root / "private_runtime_manifest.json"
    runtime_manifest.write_text(json.dumps({{"team": 23, "required_files": REQUIRED_FILES, "files": {{name: {{"sha256": digest}} for name, digest in hashes.items()}}}}, sort_keys=True) + "\\n", encoding="utf-8")
    return data_dir, {{"team": 23, "manifest": str(manifest_path) if manifest is not None else manifest_status, "manifest_status": manifest_status, "runtime_manifest": str(runtime_manifest), "input_sha256": hashes}}

def _stage_runtime_data(data_dir):
    staged = runtime_root / "data" / "23_Team_Toxic"
    staged.mkdir(parents=True, exist_ok=True)
    files = {{}}
    for name in REQUIRED_FILES:
        destination = staged / name
        if not destination.exists():
            destination.symlink_to(Path(data_dir) / name)
        files[name] = {{"sha256": _sha256(destination)}}
    (staged / "manifest.json").write_text(json.dumps({{"team": 23, "required_files": REQUIRED_FILES, "files": files}}, sort_keys=True) + "\\n", encoding="utf-8")
    return staged

def _output_dir():
    chosen = os.environ.get("TEAM23_OUTPUT_DIR")
    if chosen:
        return Path(chosen).resolve()
    return (Path("/kaggle/working") if Path("/kaggle").is_dir() else Path.cwd() / "output").resolve()

def _fresh_output(path, protected, *, forbidden_names=(), require_empty=False):
    path = Path(path).resolve()
    if any(path == p or p in path.parents for p in protected):
        raise RuntimeError("output path is inside protected raw input or checkpoint location")
    path.mkdir(parents=True, exist_ok=True)
    collisions = [path / name for name in forbidden_names if (path / name).exists()]
    if collisions:
        raise RuntimeError("refusing to overwrite existing target artifacts: " + repr([str(item) for item in collisions]))
    if require_empty and any(path.iterdir()):
        raise RuntimeError("training output directory must initially be empty: " + str(path))
    return path

def _versions():
    import numpy, pandas, torch, sacrebleu
    return {{"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "torch": torch.__version__, "sacrebleu": sacrebleu.__version__}}

runtime_root = _install_source_bundle()
try:
    import numpy as np
    import pandas as pd
    import torch
    import sacrebleu
except ImportError as exc:
    raise RuntimeError("required Kaggle runtime dependency is unavailable; no installation is attempted") from exc
print("Team 23 | source bundle SHA256:", SOURCE_BUNDLE_SHA256)
print(ROSTER_TEXT)
print("library versions:", json.dumps(_versions(), sort_keys=True))
'''


def _part1_cells(bundle: dict[str, str], bundle_hash: str) -> list[dict[str, Any]]:
    return [
        _markdown_cell("# Team 23 — Part 1\n\nCanonical schema-2 scalar denoising notebook.\n\n" + ROSTER_TEXT),
        _json_cell(_bootstrap(bundle, bundle_hash)),
        _json_cell('''data_dir, identity = discover_team23_runtime()\nprint("selected Team 23 directory:", data_dir)\nprint(json.dumps(identity, indent=2, sort_keys=True))\n\nfrom scripts.train_scalar_mlp import split_labels, train\nfrom scripts.export_scalar_weights import export\n\nclean = np.load(data_dir / "train_weights_clean.npy", mmap_mode="r", allow_pickle=False)\nnoisy = np.load(data_dir / "train_weights_noisy.npy", mmap_mode="r", allow_pickle=False)\ntest = np.load(data_dir / "test_weights_noisy.npy", mmap_mode="r", allow_pickle=False)\nif clean.shape != (EXPECTED_TRAIN_COUNT,) or noisy.shape != clean.shape or test.shape != (EXPECTED_WEIGHT_COUNT,):\n    raise RuntimeError("Team 23 weight shapes do not satisfy the exact contract")\nif any(array.dtype != np.dtype("float32") or not np.isfinite(array).all() for array in (clean, noisy, test)):\n    raise RuntimeError("Team 23 weight arrays must be finite float32")\nlabels, split = split_labels(EXPECTED_TRAIN_COUNT, seed=2301, block_size=4096, guard=32)\nprint("fixed scalar split:", json.dumps(split, sort_keys=True))\n'''),
        _json_cell('''raw_data_dir = data_dir\nstaged_data_dir = _stage_runtime_data(raw_data_dir)\ndata_dir = staged_data_dir\n'''),
        _json_cell('''output_dir = _fresh_output(_output_dir(), {raw_data_dir.resolve(), data_dir.resolve()}, forbidden_names=("denoised_test_weights.npy", "submission_part1.csv", "part1_artifact_metadata.json", "scalar_schema2_checkpoint.pt", "denoised_test_weights.json"))\ncheckpoint = output_dir / "scalar_schema2_checkpoint.pt"\nconfig = {"hidden": 32, "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 4096, "epochs": 2, "chunk_size": 65536, "seed": 2301}\ntraining = train(noisy, clean, labels, config=config, checkpoint=checkpoint, input_hashes={"clean": identity["input_sha256"]["train_weights_clean.npy"], "noisy": identity["input_sha256"]["train_weights_noisy.npy"]}, device_name="cpu")\nprint("development evidence only:", json.dumps(training, sort_keys=True))\nif training.get("audit_metrics") is not None:\n    raise RuntimeError("audit metrics must remain reserved")\n'''),
        _json_cell('''npy_path = output_dir / "denoised_test_weights.npy"\ncsv_path = output_dir / "submission_part1.csv"\nmetadata = export(checkpoint, data_dir, npy_path, csv_path, expected_count=EXPECTED_WEIGHT_COUNT)\nmetadata.update({"team": 23, "title": "AFML Orange Problem — Team 23", "source_bundle_sha256": SOURCE_BUNDLE_SHA256, "software": _versions(), "checkpoint_lineage": {"checkpoint_sha256": _sha256(checkpoint), "schema_version": 2}, "contributions": "PRIVATE_DELIVERY_METADATA_PENDING", "roster": ROSTER_TEXT, "audit_metrics": None})\nsidecar = output_dir / "part1_artifact_metadata.json"\nsidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\n# The exporter's own sidecar is redundant; keep the canonical named handoff authoritative.\n(output_dir / "denoised_test_weights.json").unlink(missing_ok=True)\nprint("Part 1 outputs:", npy_path, csv_path, sidecar)\n'''),
        _json_cell('''saved = np.load(npy_path, allow_pickle=False)\nif saved.dtype != np.dtype("float32") or saved.shape != (EXPECTED_WEIGHT_COUNT,) or not np.isfinite(saved).all():\n    raise RuntimeError("Part 1 NPY round-trip contract failed")\nwith csv_path.open("r", encoding="utf-8", newline="") as handle:\n    reader = csv.reader(handle)\n    if next(reader, None) != ["id", "weight"]:\n        raise RuntimeError("Part 1 CSV header mismatch")\n    for index, row in enumerate(reader):\n        if len(row) != 2 or row[0] != str(index) or np.float32(row[1]) != saved[index]:\n            raise RuntimeError("Part 1 CSV is not exact float32 equality with NPY")\n    if index + 1 != EXPECTED_WEIGHT_COUNT:\n        raise RuntimeError("Part 1 CSV row count mismatch")\nprint("Part 1 schema/hash validation passed; no audit was computed.")\n'''),
    ]


def _part2_input_cell() -> str:
    return '''data_dir, identity = discover_team23_runtime()
part1_dir = Path(os.environ.get("TEAM23_PART1_DIR", str(_output_dir()))).resolve()
if not part1_dir.is_dir():
    raise RuntimeError("TEAM23_PART1_DIR must point to the restored Part 1 handoff directory")
part1_npy = part1_dir / "denoised_test_weights.npy"
part1_sidecar = part1_dir / "part1_artifact_metadata.json"
if not part1_npy.is_file() or not part1_sidecar.is_file():
    raise RuntimeError("Part 2 requires the exact Part 1 NPY and metadata sidecar; noisy fallback is forbidden")
part1 = json.loads(part1_sidecar.read_text(encoding="utf-8"))
queries_path, corpus_path = data_dir / "test_queries.csv", data_dir / "train_parallel_corpus.csv"
queries = pd.read_csv(queries_path, dtype=str, keep_default_na=False)
if list(queries.columns) != ["id", "asdfghjkl"] or len(queries) != 200 or queries["id"].duplicated().any():
    raise RuntimeError("Team 23 query contract failed")
print("Part 1 handoff candidate:", part1_npy)
'''


def _part2_handoff_gate_cell() -> str:
    return '''from scripts.train_scalar_mlp import split_labels
expected_scalar_config = {"hidden": 32, "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 4096, "epochs": 2, "chunk_size": 65536, "seed": 2301}
expected_input_hashes = {"clean": identity["input_sha256"]["train_weights_clean.npy"], "noisy": identity["input_sha256"]["train_weights_noisy.npy"]}
labels_for_gate, split_for_gate = split_labels(EXPECTED_TRAIN_COUNT, seed=2301, block_size=4096, guard=32)
split_for_gate["labels_sha256"] = hashlib.sha256(labels_for_gate.tobytes()).hexdigest()

def validate_part1_handoff(sidecar, npy_path, csv_path):
    if sidecar.get("team") != 23 or sidecar.get("title") != "AFML Orange Problem — Team 23":
        raise RuntimeError("Part 1 sidecar team identity is invalid")
    if sidecar.get("source_bundle_sha256") != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("Part 1 source bundle hash differs from this notebook")
    if sidecar.get("schema_version") != 2 or sidecar.get("method") != "schema2_scalar_residual_mlp":
        raise RuntimeError("Part 1 schema or method is not the fixed scalar recipe")
    if sidecar.get("shape") != [EXPECTED_WEIGHT_COUNT] or sidecar.get("dtype") != "float32" or sidecar.get("submission_rows") != EXPECTED_WEIGHT_COUNT:
        raise RuntimeError("Part 1 shape, dtype, or row-count metadata is invalid")
    if sidecar.get("input_sha256") != expected_input_hashes:
        raise RuntimeError("Part 1 clean/noisy input hashes do not match Team 23")
    if sidecar.get("test_noisy_sha256") != identity["input_sha256"]["test_weights_noisy.npy"]:
        raise RuntimeError("Part 1 test noisy input hash does not match Team 23")
    if sidecar.get("config") != expected_scalar_config:
        raise RuntimeError("Part 1 scalar configuration is not the exact seven-field fixed recipe")
    if sidecar.get("split_sha256") != split_for_gate["labels_sha256"] or sidecar.get("split_spec_sha256") != split_for_gate["spec_sha256"]:
        raise RuntimeError("Part 1 scalar split hash or specification is invalid")
    if sidecar.get("restored_npy_sha256") != _sha256(npy_path):
        raise RuntimeError("Part 1 restored NPY hash does not match its sidecar")
    if not csv_path.is_file() or sidecar.get("submission_csv_sha256") != _sha256(csv_path):
        raise RuntimeError("Part 1 CSV hash is missing or stale")
    return sidecar["restored_npy_sha256"]

restored_hash = validate_part1_handoff(part1, part1_npy, part1_dir / "submission_part1.csv")
weights = np.load(part1_npy, allow_pickle=False)
if weights.dtype != np.dtype("float32") or weights.shape != (EXPECTED_WEIGHT_COUNT,) or not np.isfinite(weights).all():
    raise RuntimeError("restored Part 1 vector fails exact dtype/count/finite contract")
output_dir = _fresh_output(_output_dir(), {data_dir.resolve()}, forbidden_names=("submission_part2.csv", "part2_artifact_metadata.json"))
print("Part 1 initialization hash:", restored_hash)
'''


def _part2_cells(bundle: dict[str, str], bundle_hash: str) -> list[dict[str, Any]]:
    return [
        _markdown_cell("# Team 23 — Part 2\n\nCanonical modified supplied Transformer notebook.\n\n" + ROSTER_TEXT),
        _json_cell(_bootstrap(bundle, bundle_hash)),
        _json_cell(_part2_input_cell()),
        _json_cell(_part2_handoff_gate_cell()),
        _json_cell('''from scripts.make_translation_split import make_split, write_json\nfrom scripts.validate_translation_split import validate_frozen_split\nfrom scripts.team23_tokenizer import load_tokenizers\nfrom scripts.translation_noisy_baseline import EXPECTED_PARAMETER_COUNT, prepare_model, greedy_decode\nfrom scripts.train_translation import run_training\n\nruntime_data = runtime_root / "data" / "23_Team_Toxic"\nruntime_data.mkdir(parents=True, exist_ok=True)\nboilerplate_link = runtime_data / "boilerplate.ipynb"\nboilerplate_link.symlink_to(data_dir / "boilerplate.ipynb")\ncorpus_link = runtime_data / "train_parallel_corpus.csv"\ncorpus_link.symlink_to(corpus_path)\nsplit_path = runtime_data / "translation_split.json"\nwrite_json(split_path, make_split(corpus_path, seed=2301, expected_rows=300))\nsplit = validate_frozen_split(corpus_path, split_path)\nif split["counts"] != {"fitting": 210, "development": 45, "audit": 45}:\n    raise RuntimeError("frozen source-only translation split counts failed")\nsource_tok, target_tok = load_tokenizers(boilerplate_link)\nif {"<pad>": source_tok.pad_id, "<bos>": source_tok.bos_id, "<eos>": source_tok.eos_id, "<unk>": source_tok.unk_id} != {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}:\n    raise RuntimeError("supplied Team 23 token IDs failed")\n'''),
        _json_cell('''training_candidate = Path(os.environ.get("TEAM23_PART2_TRAINING_DIR", str(output_dir / "part2_training"))).resolve()\nif training_candidate in {part1_dir, data_dir}:\n    raise RuntimeError("Part 2 training directory cannot be the Part 1 handoff or raw input directory")\ntraining_dir = _fresh_output(training_candidate, {data_dir.resolve()}, require_empty=True)\ndevice_name = os.environ.get("TEAM23_TRANSLATION_DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))\nresult = run_training(notebook=boilerplate_link, corpus=corpus_path, weights=part1_npy, split=split_path, output_dir=training_dir, device_name=device_name, epochs=20, batch_size=8, learning_rate=1e-4, seed=2301, smoke=False)\ncheckpoint_path = Path(result["checkpoint"])\nif result.get("scope", {}).get("audit_ids_loaded") != 0:\n    raise RuntimeError("audit English must remain reserved")\n'''),
        _json_cell('''checkpoint_hash = _sha256(checkpoint_path)\npayload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)\nmodel = prepare_model(weights, torch.device(device_name))\nmodel.load_state_dict(payload["model"], strict=True)\nmodel.eval()\nrows = []\nwith queries_path.open("r", encoding="utf-8-sig", newline="") as handle:\n    for row in csv.DictReader(handle):\n        source_ids = source_tok.encode(row["asdfghjkl"])\n        decoded = greedy_decode(model, source_ids, bos_id=target_tok.bos_id, eos_id=target_tok.eos_id, pad_id=target_tok.pad_id, max_len=25, device=torch.device(device_name))\n        text_value = target_tok.decode(decoded).strip()\n        if not text_value or any(token in text_value for token in ("<pad>", "<bos>", "<eos>", "<unk>")):\n            raise RuntimeError("decoded query contains empty text or a special-token string")\n        rows.append((row["id"], text_value))\nif [row[0] for row in rows] != queries["id"].astype(str).tolist() or len(rows) != 200:\n    raise RuntimeError("Part 2 query ID order/count failed")\nsubmission_path = output_dir / "submission_part2.csv"\nwith submission_path.open("x", encoding="utf-8", newline="") as handle:\n    writer = csv.writer(handle, lineterminator="\\n")\n    writer.writerow(("id", "translation"))\n    writer.writerows(rows)\nmetadata = {"team": 23, "title": "AFML Orange Problem — Team 23", "method": "supplied_TranslationTransformer_v1_modified_tokenizer_adapter", "source_bundle_sha256": SOURCE_BUNDLE_SHA256, "boilerplate_sha256": _sha256(boilerplate_link), "initialization_sha256": restored_hash, "checkpoint_sha256": checkpoint_hash, "prediction_sha256": _sha256(submission_path), "split_sha256": _sha256(split_path), "split_counts": split["counts"], "config": payload["config"], "epochs_completed": result["epochs_completed"], "audit_metrics": None, "roster": ROSTER_TEXT, "contributions": "PRIVATE_DELIVERY_METADATA_PENDING"}\n(output_dir / "part2_artifact_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\nprint("Part 2 outputs:", submission_path, output_dir / "part2_artifact_metadata.json")\n'''),
        _json_cell('''with (output_dir / "submission_part2.csv").open("r", encoding="utf-8", newline="") as handle:\n    reader = csv.reader(handle)\n    if next(reader, None) != ["id", "translation"]:\n        raise RuntimeError("Part 2 CSV header mismatch")\n    checked = list(reader)\nif len(checked) != 200 or [row[0] for row in checked] != queries["id"].astype(str).tolist() or any(len(row) != 2 or not row[1].strip() for row in checked):\n    raise RuntimeError("Part 2 CSV exact output contract failed")\nprint("Part 2 schema/hash validation passed; no audit was computed or retained.")\n'''),
    ]


def build_notebook(path: Path, part: int, *, root: Path = ROOT) -> str:
    bundle, bundle_hash = source_bundle(root)
    cells = _part1_cells(bundle, bundle_hash) if part == 1 else _part2_cells(bundle, bundle_hash)
    for index, cell in enumerate(cells):
        cell["id"] = f"team23-part{part}-cell{index:02d}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for part, name in ((1, "23_part1.ipynb"), (2, "23_part2.ipynb")):
        print(name, build_notebook(args.output_dir / name, part))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
