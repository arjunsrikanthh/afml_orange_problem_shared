from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.train_translation import (
    _parent_history,
    checkpoint_payload,
    iter_batches,
    load_checkpoint,
    make_batch,
    prepare_output_dir,
    restore_rng_state,
    run_training,
    sha256_file,
    source_provenance,
    save_checkpoint,
    seed_everything,
    shifted_targets,
    teacher_forced_loss,
    validate_resume_target,
)


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab: int = 7):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, 5)
        self.output = torch.nn.Linear(5, vocab)

    def forward(self, src, tgt, **_):
        return self.output(self.embedding(tgt))


class _TinyTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def encode(self, _text):
        return [1, 3, 2]


class TrainTranslationTests(unittest.TestCase):
    def test_shifted_targets_and_pad_ignored(self):
        decoder, labels = shifted_targets([1, 4, 2], 0)
        self.assertEqual(decoder, [1, 4])
        self.assertEqual(labels, [4, 2])
        src, tgt, label = make_batch([([1, 3, 2], [1, 4, 2]), ([1, 2], [1, 5, 6, 2])], src_pad_id=0, tgt_pad_id=0, device=torch.device("cpu"))
        self.assertEqual(src.tolist(), [[1, 3, 2], [1, 2, 0]])
        self.assertEqual(tgt.tolist(), [[1, 4, 0], [1, 5, 6]])
        self.assertEqual(label.tolist(), [[4, 2, 0], [5, 6, 2]])
        loss = teacher_forced_loss(_TinyModel(), [([1, 2], [1, 4, 2])], src_pad_id=0, tgt_pad_id=0, device=torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))

    def test_batches_are_deterministic(self):
        examples = [([index], [1, index + 1]) for index in range(8)]
        first = [[example[0][0] for example in batch_group] for batch_group in iter_batches(examples, 3, seed=9, epoch=2, shuffle=True)]
        second = [[example[0][0] for example in batch_group] for batch_group in iter_batches(examples, 3, seed=9, epoch=2, shuffle=True)]
        self.assertEqual(first, second)

    def test_checkpoint_roundtrip_and_fail_closed_inputs(self):
        seed_everything(12)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        payload = checkpoint_payload(model, optimizer, scheduler, epoch=2, step=4, config={"seed": 12}, input_hashes={"corpus": "abc"}, split_ids={"fitting": ["f1"], "development": ["d1"], "audit": ["a1"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(payload, path)
            restored_model = _TinyModel()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
            self.assertEqual(load_checkpoint(path, model=restored_model, optimizer=restored_optimizer, scheduler=restored_scheduler, config={"seed": 12}, input_hashes={"corpus": "abc"}, split_ids={"fitting": ["f1"], "development": ["d1"], "audit": ["a1"]}, device=torch.device("cpu")), (2, 4))
            with self.assertRaises(ValueError):
                load_checkpoint(path, model=restored_model, optimizer=restored_optimizer, scheduler=restored_scheduler, config={"seed": 99}, input_hashes={"corpus": "abc"}, split_ids={"fitting": ["f1"], "development": ["d1"], "audit": ["a1"]}, device=torch.device("cpu"))

    def test_legacy_v1_checkpoint_without_lineage_remains_loadable(self):
        seed_everything(13)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        payload = checkpoint_payload(model, optimizer, scheduler, epoch=1, step=2, config={"seed": 13}, input_hashes={}, split_ids={"fitting": [], "development": [], "audit": []})
        payload.pop("lineage", None)
        payload.pop("history", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(payload, path)
            restored = _TinyModel()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
            self.assertEqual(load_checkpoint(path, model=restored, optimizer=restored_optimizer, scheduler=restored_scheduler, config={"seed": 13}, input_hashes={}, split_ids={"fitting": [], "development": [], "audit": []}, device=torch.device("cpu")), (1, 2))

    def test_lineage_and_history_are_retained_for_fresh_output_resume(self):
        lineage = {
            "source_git_revision": "a" * 40,
            "parent_checkpoint": {"path": "/old/checkpoint_last.pt", "sha256": "b" * 64, "epoch": 3, "step": 12},
            "parent_summary": {"path": "/old/summary.json", "sha256": "c" * 64},
        }
        history = [{"epoch": 1, "step": 4, "train_loss": 1.0, "development_loss": 2.0}]
        seed_everything(14)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        payload = checkpoint_payload(model, optimizer, scheduler, epoch=3, step=12, config={"seed": 14}, input_hashes={}, split_ids={"fitting": [], "development": [], "audit": []}, lineage=lineage, history=history)
        self.assertEqual(payload["lineage"], lineage)
        self.assertEqual(payload["history"], history)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            parent.mkdir()
            parent_checkpoint = parent / "checkpoint_last.pt"
            save_checkpoint(payload, parent_checkpoint)
            recovered, status = _parent_history(parent_checkpoint, parent_epoch=3, parent_step=12)
            self.assertEqual(recovered, history)
            self.assertFalse(status["complete"])
            self.assertIn("missing epochs: 2,3", status["gaps"])
            prepare_output_dir(parent, parent_checkpoint)
            legacy = checkpoint_payload(model, optimizer, scheduler, epoch=5, step=20, config={"seed": 14}, input_hashes={}, split_ids={"fitting": [], "development": [], "audit": []})
            legacy.pop("lineage", None)
            legacy.pop("history", None)
            legacy_checkpoint = parent / "checkpoint_epoch5.pt"
            save_checkpoint(legacy, legacy_checkpoint)
            later_history = [{"epoch": epoch, "step": epoch * 4, "train_loss": 1.0, "development_loss": 2.0} for epoch in range(6, 11)]
            (parent / "summary.json").write_text(json.dumps({"history": later_history}), encoding="utf-8")
            recovered, status = _parent_history(legacy_checkpoint, parent_epoch=5, parent_step=20)
            self.assertEqual(recovered, [])
            self.assertFalse(status["complete"])
            self.assertIn("missing epochs: 1,2,3,4,5", status["gaps"])
            output = Path(directory) / "fresh"
            prepare_output_dir(output, parent_checkpoint)
            self.assertTrue(output.is_dir())
            (output / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_dir(output, Path("/other/checkpoint_last.pt"))

    def test_resume_must_advance_beyond_saved_epoch(self):
        validate_resume_target(4, 3)
        with self.assertRaises(ValueError):
            validate_resume_target(3, 3)
        with self.assertRaises(ValueError):
            validate_resume_target(0, -1)

    def test_source_provenance_is_explicit_when_git_is_unavailable(self):
        with patch("scripts.train_translation.subprocess.run", side_effect=FileNotFoundError("git")):
            provenance = source_provenance()
        self.assertIsNone(provenance["git_revision"])
        self.assertEqual(provenance["git_status"], "unavailable")
        self.assertIsNone(provenance["dirty"])
        self.assertIn("scripts/train_translation.py", provenance["file_sha256"])

    def test_resume_rejects_bool_progress_and_nonfinite_state(self):
        seed_everything(15)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        payload = checkpoint_payload(model, optimizer, scheduler, epoch=1, step=1, config={}, input_hashes={}, split_ids={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            payload["epoch"] = True
            save_checkpoint(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model=_TinyModel(), optimizer=torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), scheduler=torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), lambda _: 1.0), config={}, input_hashes={}, split_ids={}, device=torch.device("cpu"))

            payload["epoch"] = 1
            payload["model"]["embedding.weight"][0, 0] = float("nan")
            save_checkpoint(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model=_TinyModel(), optimizer=torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), scheduler=torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), lambda _: 1.0), config={}, input_hashes={}, split_ids={}, device=torch.device("cpu"))

            payload["model"]["embedding.weight"][0, 0] = 0.0
            payload["scheduler"]["last_epoch"] = True
            save_checkpoint(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model=_TinyModel(), optimizer=torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), scheduler=torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), lambda _: 1.0), config={}, input_hashes={}, split_ids={}, device=torch.device("cpu"))

            payload["scheduler"]["last_epoch"] = 1
            payload["scheduler"]["_step_count"] = float("nan")
            save_checkpoint(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model=_TinyModel(), optimizer=torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), scheduler=torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(_TinyModel().parameters(), lr=1e-3), lambda _: 1.0), config={}, input_hashes={}, split_ids={}, device=torch.device("cpu"))

    def test_fresh_directory_continuation_records_parent_and_cumulative_history(self):
        rows = pd.DataFrame(
            [{"id": "f1", "asdfghjkl": "source", "english": "target"}, {"id": "d1", "asdfghjkl": "source2", "english": "target2"}]
        ).set_index("id")
        split = {"splits": {"fitting": ["f1"], "development": ["d1"], "audit": ["a1"]}}
        hashes = {"notebook": "n", "corpus": "c", "split": "s", "weights": "w"}
        tokenizer = _TinyTokenizer()

        def tiny_model(_weights, device):
            return _TinyModel().to(device)

        with tempfile.TemporaryDirectory() as directory, patch("scripts.train_translation._load_data", return_value=(rows, split, tokenizer, tokenizer, hashes)), patch("scripts.train_translation.load_checked_weights", return_value=np.zeros(1, dtype=np.float32)), patch("scripts.train_translation.prepare_model", side_effect=tiny_model):
            parent_dir = Path(directory) / "parent"
            first = run_training(output_dir=parent_dir, device_name="cpu", epochs=1, batch_size=1, learning_rate=1e-3)
            parent_checkpoint = Path(first["checkpoint"])
            fresh_dir = Path(directory) / "fresh"
            second = run_training(output_dir=fresh_dir, device_name="cpu", epochs=2, batch_size=1, learning_rate=1e-3, resume=parent_checkpoint)
            parent = second["lineage"]["parent_checkpoint"]
            self.assertEqual(parent["sha256"], sha256_file(parent_checkpoint))
            self.assertEqual((parent["epoch"], parent["step"]), (1, 1))
            self.assertEqual([record["epoch"] for record in second["history"]], [1, 2])
            self.assertEqual([record["epoch"] for record in second["run_history"]], [2])
            self.assertTrue(second["lineage"]["history"]["complete"])


if __name__ == "__main__":
    unittest.main()
