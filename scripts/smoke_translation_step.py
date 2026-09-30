#!/usr/bin/env python3
"""Measure one synthetic translation training step with assigned noisy weights."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from team23_tokenizer import load_tokenizers


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "23_Team_Toxic"
NOTEBOOK = DATA / "boilerplate.ipynb"
BATCH = 32
SEQUENCE = 17  # observed corpus maximum 15 whitespace tokens, plus BOS/EOS


def main() -> int:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this local smoke requires the Mac GPU")
    torch.manual_seed(23)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    if len(cells) != 8 or "class TranslationTransformer" not in cells[4]:
        raise RuntimeError("unexpected notebook structure")
    namespace = {"find_data_file": lambda filename, fallbacks=None: f"/nonexistent/{filename}"}
    for index in (1, 3, 4):
        exec(compile(cells[index], f"boilerplate.ipynb/cell-{index}", "exec"), namespace)
    model = namespace["model"]
    src_tokenizer, tgt_tokenizer = load_tokenizers(NOTEBOOK)
    if (src_tokenizer.bos_id, src_tokenizer.eos_id, tgt_tokenizer.bos_id, tgt_tokenizer.eos_id) != (1, 2, 1, 2):
        raise RuntimeError("corrected token IDs do not match embedded vocabularies")
    weights = np.load(DATA / "test_weights_noisy.npy", mmap_mode="r", allow_pickle=False)
    if weights.shape != (2_961_630,) or weights.dtype != np.float32:
        raise RuntimeError("unexpected weight vector")
    model.load_flattened_weights(weights)
    model.transformer.encoder.use_nested_tensor = False
    model = model.to("mps").train()

    src = torch.randint(4, src_tokenizer.vocab_size, (BATCH, SEQUENCE), device="mps")
    tgt = torch.randint(4, tgt_tokenizer.vocab_size, (BATCH, SEQUENCE), device="mps")
    src[:, 0] = src_tokenizer.bos_id
    src[:, -1] = src_tokenizer.eos_id
    tgt[:, 0] = tgt_tokenizer.bos_id
    tgt[:, -1] = tgt_tokenizer.eos_id
    src[: BATCH // 2, -2:] = src_tokenizer.pad_id
    tgt[: BATCH // 2, -2:] = tgt_tokenizer.pad_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_id, label_smoothing=0.05)
    torch.mps.synchronize()
    if hasattr(torch.mps, "reset_peak_memory_stats"):
        torch.mps.reset_peak_memory_stats()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    out = model(src, tgt[:, :-1], src_pad_idx=src_tokenizer.pad_id, tgt_pad_idx=tgt_tokenizer.pad_id)
    loss = criterion(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    print(json.dumps({
        "kind": "synthetic_single_step_smoke",
        "batch": BATCH,
        "sequence_length": SEQUENCE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": "mps",
        "torch_version": torch.__version__,
        "elapsed_seconds": elapsed,
        "loss_finite": True,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_peak_allocated_bytes": torch.mps.peak_allocated_memory() if hasattr(torch.mps, "peak_allocated_memory") else None,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
