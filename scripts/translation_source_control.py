"""Development-only wrong-source control using saved deterministic predictions.

Rotating predictions is equivalent to rotating source inputs for independent,
deterministic per-source decoding. This is a diagnostic, not an audit score or
a permutation significance test. It does not require another model run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.translation_noisy_baseline import DATA, evaluate, sha256_file
from scripts.validate_translation_split import validate_frozen_split


def controls(predictions: list[str], references: list[str]) -> dict:
    if len(predictions) < 2 or len(predictions) != len(references):
        raise ValueError("wrong-source control requires at least two aligned rows")
    aligned = evaluate(predictions, references)
    rotated = [evaluate(predictions[k:] + predictions[:k], references)
               for k in range(1, len(predictions))]
    return {
        "aligned": aligned,
        "wrong_source_rotations": len(rotated),
        "wrong_source": {
            key: {"min": min(m[key] for m in rotated),
                  "median": float(np.median([m[key] for m in rotated])),
                  "max": max(m[key] for m in rotated),
                  "at_least_aligned": sum(m[key] >= aligned[key] for m in rotated)}
            for key in ("corpus_bleu4", "chrf_plus_plus", "exact_match")
        },
        "unique_predictions": len(set(predictions)),
        "empty_predictions": sum(not x.strip() for x in predictions),
    }


def run(result_path: Path, predictions_path: Path, output: Path) -> dict:
    frozen = validate_frozen_split(DATA / "train_parallel_corpus.csv", DATA / "translation_split.json")
    result = json.loads(result_path.read_text())
    frame = pd.read_csv(predictions_path, keep_default_na=False)
    if (result["scope"]["split"] != "development"
            or frame["id"].tolist() != frozen["splits"]["development"]
            or result["scope"]["ids"] != frame["id"].tolist()
            or result["config"]["decoding"] != "greedy_argmax"):
        raise ValueError("requires frozen development IDs and deterministic independent greedy outputs")
    observed_hash = sha256_file(predictions_path)
    if result["provenance"].get("prediction_csv_sha256") != observed_hash:
        raise ValueError("prediction CSV hash is missing or mismatched")
    summary = controls(frame["prediction"].tolist(), frame["reference"].tolist())
    if summary["aligned"] != result["metrics"]:
        raise ValueError("saved metrics do not recompute from CSV")
    summary.update(scope="development_only", rows=len(frame),
                   prediction_csv_sha256=observed_hash, result_json_sha256=sha256_file(result_path),
                   limitation="Cyclic wrong-source diagnostic; not an independent audit or significance test.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.result, args.predictions, args.output), sort_keys=True))
