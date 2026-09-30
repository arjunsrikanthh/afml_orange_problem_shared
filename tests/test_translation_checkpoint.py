from __future__ import annotations

import copy
import unittest

import torch

from scripts.train_translation import checkpoint_payload
from scripts.translation_checkpoint import validate_checkpoint


class TranslationCheckpointTests(unittest.TestCase):
    def setUp(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
        self.special = {"source_pad": 0, "target_pad": 0, "target_bos": 1, "target_eos": 2}
        self.hashes = {name: name + "-hash" for name in ("notebook", "corpus", "split", "weights")}
        self.splits = {"fitting": ["fit"], "development": ["dev"], "audit": ["audit"]}
        self.payload = checkpoint_payload(
            model, optimizer, scheduler, epoch=1, step=1,
            config={"trainer": "translation-trainer-v1",
                    "architecture": "supplied_TranslationTransformer_v1",
                    "parameter_count": 2, "smoke": False, "special_ids": self.special},
            input_hashes=self.hashes, split_ids=self.splits,
        )

    def validate(self, payload):
        validate_checkpoint(payload, input_hashes=self.hashes, split_ids=self.splits,
                            parameter_count=2, special_ids=self.special)

    def test_legacy_complete_checkpoint_accepted_without_lineage(self):
        self.validate(self.payload)

    def test_missing_recovery_fields_rejected(self):
        for field in ("optimizer", "scheduler", "rng", "epoch", "step"):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.payload)
                del broken[field]
                with self.assertRaises(ValueError):
                    self.validate(broken)

    def test_invalid_progress_rejected(self):
        for field in ("epoch", "step"):
            for value in (True, -1, 1.5, "10", None):
                with self.subTest(field=field, value=value):
                    broken = copy.deepcopy(self.payload)
                    broken[field] = value
                    with self.assertRaises(ValueError):
                        self.validate(broken)

    def test_nonfinite_model_optimizer_and_scheduler_rejected(self):
        for field in ("model", "optimizer", "scheduler"):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.payload)
                if field == "model":
                    broken[field]["weight"].fill_(float("nan"))
                elif field == "optimizer":
                    next(iter(broken[field]["state"].values()))["exp_avg"].fill_(float("inf"))
                else:
                    broken[field]["_last_lr"] = [float("nan")]
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    self.validate(broken)

    def test_changed_hashes_splits_padding_and_smoke_rejected(self):
        for field in ("hash", "split", "padding", "smoke", "rng"):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.payload)
                if field == "hash":
                    broken["input_hashes"]["weights"] = "different"
                elif field == "split":
                    broken["split_ids"]["development"] = ["audit"]
                elif field == "padding":
                    broken["config"]["special_ids"]["source_pad"] = 99
                elif field == "smoke":
                    broken["config"]["smoke"] = True
                else:
                    broken["rng"]["torch"] = torch.ones(3)
                with self.assertRaises(ValueError):
                    self.validate(broken)
