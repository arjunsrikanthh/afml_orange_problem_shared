#!/usr/bin/env python3
"""Audit the assigned notebook's weight loader and embedded token IDs without training."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from team23_tokenizer import load_tokenizers


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "23_Team_Toxic"
OUTPUT = ROOT / "reports" / "generated" / "loader_audit.json"
EXPECTED_WEIGHTS = 2_961_630
SPECIAL_NAMES = ("<pad>", "<bos>", "<eos>", "<unk>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_vocab(cell: str, assignment: str) -> dict[str, int]:
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment for target in node.targets
        ):
            encoded = ast.literal_eval(node.value)
            return json.loads(zlib.decompress(base64.b64decode(encoded)).decode("utf-8"))
    raise ValueError(f"missing embedded vocabulary {assignment}")


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name in ("boilerplate.ipynb", "test_weights_noisy.npy"):
        if sha256(DATA / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"input changed since preflight: {name}")

    notebook = json.loads((DATA / "boilerplate.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    if len(cells) != 8 or "class TranslationTransformer" not in cells[4]:
        raise ValueError("unexpected notebook structure; review before executing audit cells")

    namespace = {"find_data_file": lambda filename, fallbacks=None: f"/nonexistent/{filename}"}
    for index in (1, 3, 4):
        exec(compile(cells[index], f"boilerplate.ipynb/cell-{index}", "exec"), namespace)
    model = namespace["model"]
    parameters = list(model.named_parameters())
    state = model.state_dict()
    parameter_keys = [name for name, _ in parameters]
    state_keys = [name for name in state if name != "pos_encoder.pe"]
    total = sum(tensor.numel() for _, tensor in parameters)
    layout_ok = total == EXPECTED_WEIGHTS and parameter_keys == state_keys

    table = []
    offset = 0
    for name, tensor in parameters:
        count = tensor.numel()
        table.append({"name": name, "shape": list(tensor.shape), "offset": offset, "count": count})
        offset += count

    flat = np.load(DATA / "test_weights_noisy.npy", mmap_mode="r", allow_pickle=False)
    if flat.dtype != np.float32 or flat.shape != (EXPECTED_WEIGHTS,):
        raise ValueError("test vector dtype or shape differs from verified contract")
    if not layout_ok:
        raise ValueError("parameter keys/order/count do not match test vector")
    model.load_flattened_weights(flat)
    loaded = torch.cat([tensor.detach().reshape(-1).cpu() for _, tensor in model.named_parameters()])
    real_roundtrip_ok = np.array_equal(loaded.numpy(), flat)

    sentinel = np.linspace(-1.0, 1.0, EXPECTED_WEIGHTS, dtype=np.float32)
    model.load_flattened_weights(sentinel)
    loaded_sentinel = torch.cat([tensor.detach().reshape(-1).cpu() for _, tensor in model.named_parameters()])
    sentinel_roundtrip_ok = np.array_equal(loaded_sentinel.numpy(), sentinel)

    vocab_results = {}
    for language, assignment, tokenizer in (
        ("source", "_EMBEDDED_VOCAB_ASDF_B64", namespace["src_tok"]),
        ("target", "_EMBEDDED_VOCAB_ENG_B64", namespace["tgt_tok"]),
    ):
        original = embedded_vocab(cells[3], assignment)
        original_special = {name: original.get(name) for name in SPECIAL_NAMES}
        active_special = {
            "<pad>": tokenizer.pad_id,
            "<bos>": tokenizer.bos_id,
            "<eos>": tokenizer.eos_id,
            "<unk>": tokenizer.unk_id,
        }
        vocab_results[language] = {
            "embedded_entries": len(original),
            "embedded_id_range": [min(original.values()), max(original.values())],
            "embedded_special_ids": original_special,
            "starter_vocab_size": tokenizer.vocab_size,
            "starter_special_ids": active_special,
            "special_ids_match": original_special == active_special,
        }

    corrected_source, corrected_target = load_tokenizers(DATA / "boilerplate.ipynb")
    source_embedding_rows = model.src_embedding.num_embeddings
    target_embedding_rows = model.tgt_embedding.num_embeddings
    corrected_ids_ok = all(
        vocab_results[language]["embedded_special_ids"] == {
            "<pad>": tokenizer.pad_id,
            "<bos>": tokenizer.bos_id,
            "<eos>": tokenizer.eos_id,
            "<unk>": tokenizer.unk_id,
        }
        for language, tokenizer in (("source", corrected_source), ("target", corrected_target))
    )
    embedding_bounds_ok = (
        max(corrected_source.word2idx.values()) < source_embedding_rows
        and max(corrected_target.word2idx.values()) < target_embedding_rows
    )
    oov_counts = {"train_source": 0, "train_target": 0, "test_source": 0}
    for filename, columns in (
        ("train_parallel_corpus.csv", (("asdfghjkl", corrected_source, "train_source"), ("english", corrected_target, "train_target"))),
        ("test_queries.csv", (("asdfghjkl", corrected_source, "test_source"),)),
    ):
        with (DATA / filename).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for column, tokenizer, label in columns:
                    oov_counts[label] += sum(token not in tokenizer.word2idx for token in tokenizer.tokenize(row[column]))

    result = {
        "schema_version": 1,
        "notebook_sha256": manifest["files"]["boilerplate.ipynb"]["sha256"],
        "test_weights_sha256": manifest["files"]["test_weights_noisy.npy"]["sha256"],
        "torch_version": torch.__version__,
        "parameter_tensors": len(parameters),
        "parameter_count": total,
        "state_dict_tensors": len(state),
        "excluded_buffer": {"name": "pos_encoder.pe", "shape": list(state["pos_encoder.pe"].shape)},
        "weight_layout_passed": layout_ok,
        "real_roundtrip_passed": real_roundtrip_ok,
        "sentinel_roundtrip_passed": sentinel_roundtrip_ok,
        "tokenizers": vocab_results,
        "corrected_tokenizer_special_ids_passed": corrected_ids_ok,
        "embedding_id_bounds_passed": embedding_bounds_ok,
        "corrected_tokenizer_oov_counts": oov_counts,
        "parameter_table": table,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"loader audit: {OUTPUT}")
    print(f"weight layout={layout_ok}, real roundtrip={real_roundtrip_ok}, sentinel={sentinel_roundtrip_ok}")
    print("embedded special IDs preserved=" + str(all(v["special_ids_match"] for v in vocab_results.values())))
    print(f"corrected tokenizer IDs={corrected_ids_ok}, bounds={embedding_bounds_ok}, OOV={oov_counts}")
    return 0 if all((layout_ok, real_roundtrip_ok, sentinel_roundtrip_ok, corrected_ids_ok, embedding_bounds_ok)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
