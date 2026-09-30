import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.compare_weight_candidates import compare_predictions, evaluate_checkpoint
from scripts.train_scalar_mlp import ScalarResidualMLP, split_labels, train


class CompareWeightCandidatesTests(unittest.TestCase):
    def test_global_rmse_is_count_weighted_not_mean_block_rmse(self):
        noisy = np.zeros(6, dtype=np.float32)
        clean = np.zeros(6, dtype=np.float32)
        labels = np.array([0, 1, 1, 1, 1, 1], dtype=np.int8)
        mlp = np.zeros(6, dtype=np.float32)
        noisy[1:] = 1
        mlp[1:] = np.array([0, 0, 0, 2, 2], dtype=np.float32)
        report = compare_predictions(noisy, clean, labels, mlp, bootstrap_repetitions=40)
        self.assertEqual(report["development_count"], 5)
        self.assertAlmostEqual(report["mlp"]["rmse"], np.sqrt(8 / 5), places=12)
        self.assertEqual(report["block_count"], 1)
        labels = np.array([0, 1, 1, 1, 0, 1], dtype=np.int8)
        report = compare_predictions(noisy, clean, labels, mlp, bootstrap_repetitions=40)
        self.assertEqual(report["block_count"], 2)
        self.assertAlmostEqual(report["mlp"]["rmse"], 1.0, places=12)
        self.assertNotAlmostEqual(report["mlp"]["rmse"], (np.sqrt(4 / 3) + 2.0) / 2.0, places=6)
        self.assertEqual(report["blocks"][0]["count"], 3)
        self.assertEqual(report["blocks"][1]["count"], 1)

    def test_block_pairing_and_wins_losses_use_same_positions(self):
        noisy = np.zeros(8, dtype=np.float32)
        clean = np.zeros(8, dtype=np.float32)
        labels = np.array([0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int8)
        noisy[labels == 1] = 1
        mlp = np.array([0, 0, 2, 0, 3, 3, 0, 0], dtype=np.float32)
        report = compare_predictions(noisy, clean, labels, mlp, bootstrap_repetitions=30)
        self.assertEqual(report["block_wins_losses"], {"mlp_wins": 1, "affine_wins": 2, "ties": 0})
        self.assertEqual([item["winner"] for item in report["blocks"]], ["affine", "affine", "mlp"])
        self.assertGreater(report["mlp_minus_affine"]["sse"], 0)

    def test_checkpoint_validation_excludes_audit_and_rejects_legacy(self):
        total = 20
        noisy = np.linspace(-1, 1, total, dtype=np.float32)
        clean = noisy.copy()
        labels, split = split_labels(total, block_size=2, guard=0)
        labels[:] = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0], dtype=np.int8)
        split["counts"] = {name: int(np.count_nonzero(labels == code)) for name, code in (("fitting", 0), ("development", 1), ("audit", 2), ("guard", -1))}
        split["labels_sha256"] = hashlib.sha256(labels.tobytes()).hexdigest()
        input_hashes = {"clean": "clean-hash", "noisy": "noisy-hash"}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "schema2.pt"
            train(noisy, clean, labels, config={"hidden": 3, "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 2, "epochs": 1, "chunk_size": 16, "seed": 2301}, checkpoint=checkpoint, input_hashes=input_hashes, device_name="cpu")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            split = payload["split"]
            report = evaluate_checkpoint(noisy, clean, labels, [checkpoint], input_hashes=input_hashes, split=split, bootstrap_repetitions=20)
            self.assertEqual(report["candidates"][0]["development_count"], 6)
            self.assertIsNone(report["audit_metrics"])
            legacy = dict(payload)
            legacy["schema_version"] = 1
            legacy_path = Path(directory) / "legacy.pt"
            torch.save(legacy, legacy_path)
            with self.assertRaisesRegex(ValueError, "schema2"):
                evaluate_checkpoint(noisy, clean, labels, [legacy_path], input_hashes=input_hashes, split=split, bootstrap_repetitions=20)

            empty_state = dict(payload)
            empty_state["optimizer"] = dict(payload["optimizer"])
            empty_state["optimizer"]["state"] = {}
            empty_path = Path(directory) / "empty-state.pt"
            torch.save(empty_state, empty_path)
            with self.assertRaisesRegex(ValueError, "optimizer state is empty"):
                evaluate_checkpoint(noisy, clean, labels, [empty_path], input_hashes=input_hashes, split=split, bootstrap_repetitions=20)

            bad_rng = dict(payload)
            bad_rng["rng"] = dict(payload["rng"])
            bad_rng["rng"]["python"] = (3, (1,), None)
            bad_rng_path = Path(directory) / "bad-rng.pt"
            torch.save(bad_rng, bad_rng_path)
            with self.assertRaisesRegex(ValueError, "RNG state is malformed"):
                evaluate_checkpoint(noisy, clean, labels, [bad_rng_path], input_hashes=input_hashes, split=split, bootstrap_repetitions=20)

    def test_linked_half_units_are_kept_together(self):
        half_labels = np.array([0, 1, 1, 0, 1, 0], dtype=np.int8)
        labels = np.concatenate([half_labels, half_labels])
        noisy = np.zeros(labels.size, dtype=np.float32)
        clean = np.zeros(labels.size, dtype=np.float32)
        mlp = np.zeros(labels.size, dtype=np.float32)
        report = compare_predictions(noisy, clean, labels, mlp, bootstrap_repetitions=20)
        self.assertEqual(report["block_count"], 2)
        self.assertTrue(all(len(item["segments"]) == 2 for item in report["blocks"]))
        self.assertEqual([item["count"] for item in report["blocks"]], [4, 2])
        self.assertEqual(report["bootstrap"]["resampling_unit"], "linked same-offset contiguous development runs")

    def test_nonfinite_audit_is_never_indexed_but_nonfinite_development_fails(self):
        noisy = np.zeros(6, dtype=np.float32)
        clean = np.array([0, 0, 0, np.nan, 0, 0], dtype=np.float32)
        labels = np.array([0, 0, 1, 2, 1, 0], dtype=np.int8)
        compare_predictions(noisy, clean, labels, noisy, bootstrap_repetitions=20)
        clean[2] = np.nan
        with self.assertRaisesRegex(ValueError, "fitting/development"):
            compare_predictions(noisy, clean, labels, noisy, bootstrap_repetitions=20)


if __name__ == "__main__":
    unittest.main()
