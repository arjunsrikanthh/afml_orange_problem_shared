#!/usr/bin/env python3
"""Evaluate the supplied noisy translation weights on Team 23 development only.

This is an inference-only baseline.  It deliberately does not read test queries,
fit on any rows, or alter the supplied weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sacrebleu
import torch
from sacrebleu.metrics import BLEU, CHRF

from scripts.team23_tokenizer import Team23Tokenizer, load_tokenizers
from scripts.validate_translation_split import validate_frozen_split
from scripts.translation_checkpoint import validate_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "23_Team_Toxic"
DEFAULT_NOTEBOOK = DATA / "boilerplate.ipynb"
DEFAULT_CORPUS = DATA / "train_parallel_corpus.csv"
DEFAULT_WEIGHTS = DATA / "test_weights_noisy.npy"
DEFAULT_SPLIT = DATA / "translation_split.json"
DEFAULT_JSON = ROOT / "results" / "baselines" / "translation_noisy_dev.json"
DEFAULT_CSV = ROOT / "results" / "baselines" / "translation_noisy_dev.csv"
EXPECTED_PARAMETER_COUNT = 2_961_630


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TranslationTransformer(torch.nn.Module):
    """The supplied notebook architecture, with its parameter order unchanged."""

    def __init__(
        self,
        src_vocab_size: int = 5947,
        tgt_vocab_size: int = 5982,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = torch.nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ) -> torch.Tensor:
        device = src.device
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)
        src_padding_mask = src == src_pad_idx
        tgt_padding_mask = tgt == tgt_pad_idx
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.fc_out(out)

    def load_flattened_weights(self, flat_weights: np.ndarray | torch.Tensor) -> None:
        values = torch.as_tensor(flat_weights, dtype=torch.float32).reshape(-1)
        parameters = list(self.named_parameters())
        expected = sum(parameter.numel() for _, parameter in parameters)
        if values.numel() != expected:
            raise ValueError(f"weight length {values.numel()} does not equal model parameter count {expected}")
        offset = 0
        with torch.no_grad():
            for _, parameter in parameters:
                count = parameter.numel()
                parameter.copy_(values[offset : offset + count].view_as(parameter))
                offset += count


def prepare_model(weights: np.ndarray, device: torch.device) -> TranslationTransformer:
    model = TranslationTransformer()
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("fixed Transformer parameter count changed")
    if device.type == "mps":
        # PyTorch's nested-tensor encoder path is unavailable on MPS for this model.
        model.transformer.encoder.use_nested_tensor = False
    model.load_flattened_weights(weights)
    model.to(device)
    model.eval()
    return model


def greedy_decode(
    model: TranslationTransformer,
    source_ids: list[int] | torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 25,
    device: torch.device | None = None,
) -> list[int]:
    """Decode one source sequence, including BOS and the first EOS when emitted."""
    if max_len < 1:
        raise ValueError("max_len must be positive")
    device = device or next(model.parameters()).device
    src = torch.as_tensor(source_ids, dtype=torch.long, device=device).reshape(1, -1)
    generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src, generated, src_pad_idx=pad_id, tgt_pad_idx=pad_id)[:, -1, :]
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("non-finite logits during translation decoding")
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if int(next_id.item()) == eos_id:
                break
    return generated[0].detach().cpu().tolist()


def validate_inputs(
    corpus: Path, split_path: Path, weights_path: Path
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    split = validate_frozen_split(corpus, split_path)
    observed_corpus_hash = sha256_file(corpus)
    if observed_corpus_hash != split["source_corpus_sha256"]:
        raise ValueError("development split corpus hash does not match the corpus on disk")
    rows = pd.read_csv(corpus)
    required = {"id", "asdfghjkl", "english"}
    if not required.issubset(rows.columns):
        raise ValueError(f"corpus is missing required columns: {sorted(required - set(rows.columns))}")
    development_ids = list(split["splits"]["development"])
    if len(development_ids) != split["counts"]["development"] or len(set(development_ids)) != len(development_ids):
        raise ValueError("development split has invalid IDs")
    if set(development_ids) - set(rows["id"]):
        raise ValueError("development split contains an ID absent from the corpus")
    selected = rows.set_index("id").loc[development_ids].reset_index()
    weights = np.load(weights_path, allow_pickle=False)
    if weights.dtype != np.float32 or weights.size != EXPECTED_PARAMETER_COUNT or not np.isfinite(weights).all():
        raise ValueError("test noisy weights fail dtype, length, or finite-value validation")
    if weights_path.resolve() == DEFAULT_WEIGHTS.resolve():
        manifest = json.loads((weights_path.parent / "manifest.json").read_text(encoding="utf-8"))
        if sha256_file(weights_path) != manifest["files"][weights_path.name]["sha256"]:
            raise ValueError("supplied noisy test weights differ from the preflight manifest")
    return selected, split, weights.reshape(-1)


def evaluate(predictions: list[str], references: list[str]) -> dict[str, Any]:
    if not references or len(predictions) != len(references):
        raise ValueError("metrics require equal nonempty prediction/reference lists")
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])
    chrf = CHRF(word_order=2)
    chrf_score = chrf.corpus_score(predictions, [references])
    return {
        "corpus_bleu4": float(bleu_score.score),
        "chrf_plus_plus": float(chrf_score.score),
        "exact_match": float(sum(pred == ref for pred, ref in zip(predictions, references)) / len(references)),
        "metric_versions": {"sacrebleu": sacrebleu.__version__},
        "metric_signatures": {
            "bleu": bleu.get_signature().format(),
            "chrf_plus_plus": chrf.get_signature().format(),
        },
    }


def run_baseline(
    *,
    notebook: Path = DEFAULT_NOTEBOOK,
    corpus: Path = DEFAULT_CORPUS,
    weights: Path = DEFAULT_WEIGHTS,
    split: Path = DEFAULT_SPLIT,
    output_json: Path = DEFAULT_JSON,
    output_csv: Path = DEFAULT_CSV,
    device_name: str | None = None,
    max_len: int = 25,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    rows, split_payload, flat_weights = validate_inputs(corpus, split, weights)
    source_tokenizer, target_tokenizer = load_tokenizers(notebook)
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = prepare_model(flat_weights, device)
    checkpoint_progress = None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_hashes = {
            "notebook": sha256_file(notebook), "corpus": sha256_file(corpus),
            "split": sha256_file(split), "weights": sha256_file(weights),
        }
        validate_checkpoint(
            payload, input_hashes=expected_hashes, split_ids=split_payload["splits"],
            parameter_count=EXPECTED_PARAMETER_COUNT,
            special_ids={"source_pad": source_tokenizer.pad_id,
                         "target_pad": target_tokenizer.pad_id,
                         "target_bos": target_tokenizer.bos_id,
                         "target_eos": target_tokenizer.eos_id},
        )
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        checkpoint_progress = {"epoch": payload.get("epoch"), "step": payload.get("step"), "sha256": sha256_file(checkpoint)}
    predictions: list[str] = []
    generated_sequences: list[list[int]] = []
    for source in rows["asdfghjkl"].astype(str):
        source_ids = source_tokenizer.encode(source, add_special_tokens=True)
        generated = greedy_decode(
            model,
            source_ids,
            bos_id=target_tokenizer.bos_id,
            eos_id=target_tokenizer.eos_id,
            pad_id=target_tokenizer.pad_id,
            max_len=max_len,
            device=device,
        )
        predictions.append(target_tokenizer.decode(generated, skip_special_tokens=True))
        generated_sequences.append(generated)
    references = rows["english"].astype(str).tolist()
    metrics = evaluate(predictions, references)
    # Raw-reference metrics above remain the fixed selection contract. These
    # diagnostics expose formatting and termination errors without silently
    # redefining the reported baseline or optimizing on audit references.
    normalized_references = [" ".join(target_tokenizer.tokenize(ref)) for ref in references]
    eos_count = sum(target_tokenizer.eos_id in seq[1:] for seq in generated_sequences)
    token_lists = [prediction.split() for prediction in predictions]
    repetitive = sum(bool(tokens) and max(tokens.count(token) for token in set(tokens)) / len(tokens) >= 0.5
                     for tokens in token_lists)
    per_id = rows[["id", "asdfghjkl", "english"]].copy()
    per_id["prediction"] = predictions
    per_id["exact_match"] = per_id["prediction"] == per_id["english"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_id.rename(columns={"asdfghjkl": "source", "english": "reference"}).to_csv(output_csv, index=False)
    result: dict[str, Any] = {
        "baseline": ("translation_noisy_weights_no_finetuning" if weights.resolve() == DEFAULT_WEIGHTS.resolve()
                     else "translation_restored_weights_no_finetuning") if checkpoint is None else "translation_finetuned_weights",
        "scope": {"split": "development", "rows": len(rows), "ids": rows["id"].tolist()},
        "metrics": metrics,
        "diagnostics": {
            "normalized_reference_metrics": evaluate(predictions, normalized_references),
            "reference_normalization": "target_tokenizer.tokenize joined with single spaces; diagnostic only",
            "eos_terminated": eos_count,
            "cap_terminated": len(predictions) - eos_count,
            "empty_predictions": sum(not prediction.strip() for prediction in predictions),
            "unique_predictions": len(set(predictions)),
            "rows_with_token_share_at_least_half": repetitive,
            "mean_prediction_tokens": sum(map(len, token_lists)) / len(token_lists),
            "mean_reference_tokens": sum(len(ref.split()) for ref in normalized_references) / len(references),
        },
        "config": {
            "device": str(device),
            "max_len_generated_after_bos": max_len,
            "decoding": "greedy_argmax",
            "explicit_bos_eos": True,
            "model_eval": True,
            "nested_tensor_disabled_on_mps": device.type == "mps",
            "fine_tuning": checkpoint is not None,
            "initialization": "supplied_noisy" if weights.resolve() == DEFAULT_WEIGHTS.resolve() else "restored_candidate",
        },
        "provenance": {
            "prediction_csv_sha256": sha256_file(output_csv),
            "notebook_sha256": sha256_file(notebook),
            "corpus_sha256": sha256_file(corpus),
            "split_sha256": sha256_file(split),
            "weights_sha256": sha256_file(weights),
            "checkpoint": checkpoint_progress,
            "split_corpus_hash_validated": True,
            "weights_file": str(weights),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "source_vocab_size": source_tokenizer.vocab_size,
            "target_vocab_size": target_tokenizer.vocab_size,
            "special_ids": {
                "source": {"pad": source_tokenizer.pad_id, "bos": source_tokenizer.bos_id, "eos": source_tokenizer.eos_id},
                "target": {"pad": target_tokenizer.pad_id, "bos": target_tokenizer.bos_id, "eos": target_tokenizer.eos_id},
            },
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--device", dest="device_name", default=None)
    parser.add_argument("--max-len", type=int, default=25)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = run_baseline(**vars(args))
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
