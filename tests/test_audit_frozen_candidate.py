from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts.audit_frozen_candidate import (
    RAW_FILES,
    RECIPE,
    SUPPORTED_RUNTIME_MODULES,
    audit_model_rmse,
    run_audit,
    start_audit,
    validate_freeze_document,
    validate_translation_init,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FrozenAuditTests(unittest.TestCase):
    def freeze(self) -> dict:
        return {
            "schema_version": 1, "status": "frozen",
            "input_sha256": {name: "a" * 64 for name in RAW_FILES},
            "artifacts": {name: {"path": f"results/{name}", "sha256": "b" * 64} for name in ("scalar_checkpoint", "restored_weights", "translation_checkpoint", "translation_split")},
            "runtime_sha256": {name: "c" * 64 for name in SUPPORTED_RUNTIME_MODULES}, "recipe": copy.deepcopy(RECIPE),
            "selection": {"rationale": "synthetic", "time": "test", "git": "test"},
        }

    def test_tampered_freeze_and_hash_are_rejected(self):
        payload = self.freeze()
        validate_freeze_document(payload)
        payload["recipe"]["decoding"]["max_len"] = 24
        with self.assertRaises(ValueError): validate_freeze_document(payload)

    def test_full_recipe_fields_cannot_be_omitted_or_changed(self):
        for section, key in (("scalar", "hidden"), ("scalar", "weight_decay"), ("translation", "device")):
            payload = self.freeze()
            del payload["recipe"][section][key]
            with self.assertRaises(ValueError):
                validate_freeze_document(payload)
        payload = self.freeze()
        payload["recipe"]["translation"]["device"] = "cpu"
        with self.assertRaises(ValueError):
            validate_freeze_document(payload)
        payload = self.freeze()
        payload["recipe"]["scalar"]["seed"] = True
        with self.assertRaises(ValueError):
            validate_freeze_document(payload)
        payload = self.freeze()
        payload["input_sha256"]["test_queries.csv"] = "z" * 64
        with self.assertRaises(ValueError): validate_freeze_document(payload)

    def test_model_audit_uses_full_unequal_array_and_never_reads_non_audit_values(self):
        class AddOne(torch.nn.Module):
            def forward(self, values):
                return values + 1
        noisy = np.array([np.nan, 1., 2., np.nan, np.nan, np.nan, 4.], dtype=np.float32)
        clean = np.array([np.nan, 1., 2., np.nan, np.nan, np.nan, 2.], dtype=np.float32)
        labels = np.array([0, 2, 2, 0, 1, 0, 2], dtype=np.int8)
        result = audit_model_rmse(AddOne(), noisy, clean, labels, chunk_size=4)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["sse"], 11.0)
        self.assertAlmostEqual(result["rmse"], np.sqrt(11 / 3))

    def test_duplicate_consumption_refusal_even_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = Path(directory) / "audit"
            freeze = Path(directory) / "freeze.json"
            freeze.write_text("freeze", encoding="utf-8")
            ledger_path, ledger = start_audit(output, freeze, root=root)
            self.assertEqual(ledger["status"], "started")
            with self.assertRaises(FileExistsError): start_audit(Path(directory) / "other", freeze, root=root)

    def test_checkpoint_initialization_mismatch_is_rejected_cheaply(self):
        payload = {"input_hashes": {"weights": "d" * 64}}
        with mock.patch("scripts.translation_checkpoint.validate_checkpoint"):
            with self.assertRaisesRegex(ValueError, "initialization hash"):
                validate_translation_init(payload, expected_hashes={}, expected_split_ids={}, special_ids={}, expected_init_hash="e" * 64)

    def test_failed_run_writes_ledger_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps(self.freeze()), encoding="utf-8")
            output = root / "audit"
            with self.assertRaises(ValueError): run_audit(freeze, output, root=root)
            self.assertFalse(output.exists())
            self.assertFalse((root / "results" / "audit_ledger").exists())

    def test_alternate_output_dir_is_refused_after_started_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = root / "freeze.json"
            freeze.write_text("freeze", encoding="utf-8")
            first = root / "one"
            ledger_path, _ = start_audit(first, freeze, root=root)
            ledger_path.write_text(ledger_path.read_text().replace('"started"', '"failed"'), encoding="utf-8")
            with self.assertRaises(FileExistsError): start_audit(root / "two", freeze, root=root)


if __name__ == "__main__":
    unittest.main()
