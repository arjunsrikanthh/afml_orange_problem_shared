from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train_scalar_mlp import ScalarResidualMLP, paired_mse, train


class ScalarMLPTests(unittest.TestCase):
    def _data(self):
        noisy = np.linspace(-2, 2, 64, dtype=np.float32)
        clean = (0.7 * noisy + 0.2).astype(np.float32)
        labels = np.full(64, -1, dtype=np.int8)
        labels[:32] = 0
        labels[32:48] = 1
        labels[48:] = 2
        return noisy, clean, labels

    def test_model_is_scalar_residual(self):
        model = ScalarResidualMLP(hidden=8)
        values = torch.tensor([[-1.0], [2.0]])
        self.assertEqual(model(values).shape, values.shape)
        self.assertEqual(model(values.reshape(-1)).shape, (2,))

    def test_aligned_loss_does_not_compare_every_prediction_to_every_target(self):
        # The old [B,1] - [B] expression gives 5.0 despite perfect predictions.
        target = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        prediction = target.clone().requires_grad_(True)
        loss = paired_mse(prediction, target)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(target)))
        self.assertGreater(float(torch.mean((target[:, None] - target) ** 2)), 0)
        with self.assertRaises(ValueError):
            paired_mse(target[:, None], target)

    def test_legacy_broadcast_objective_checkpoint_cannot_resume(self):
        noisy, clean, labels = self._data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            config = {"epochs": 2, "hidden": 8, "batch_size": 8}
            train(noisy, clean, labels, config=config, checkpoint=path,
                  device_name="cpu", max_epochs_this_run=1)
            payload = torch.load(path, weights_only=False)
            payload["schema_version"] = 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "schema"):
                train(noisy, clean, labels, config=config, checkpoint=path,
                      device_name="cpu", resume=True, max_epochs_this_run=1)

    def test_checkpoint_roundtrip_and_resume_contract(self):
        noisy, clean, labels = self._data()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            config = {"epochs": 1, "batch_size": 8, "hidden": 8, "chunk_size": 16, "seed": 9}
            first = train(noisy, clean, labels, config=config, checkpoint=checkpoint, input_hashes={"clean": "c", "noisy": "n"}, device_name="cpu")
            self.assertTrue(checkpoint.is_file())
            resumed = train(noisy, clean, labels, config=config, checkpoint=checkpoint, input_hashes={"clean": "c", "noisy": "n"}, resume=True, device_name="cpu")
            self.assertEqual(first["audit_metrics"], None)
            self.assertEqual(resumed["step"], first["step"])
            with self.assertRaises(ValueError):
                train(noisy, clean, labels, config={**config, "hidden": 9}, checkpoint=checkpoint, input_hashes={"clean": "c", "noisy": "n"}, resume=True, device_name="cpu")
            changed_labels = labels.copy()
            changed_labels[0], changed_labels[32] = changed_labels[32], changed_labels[0]
            with self.assertRaises(ValueError):
                train(noisy, clean, changed_labels, config=config, checkpoint=checkpoint, input_hashes={"clean": "c", "noisy": "n"}, resume=True, device_name="cpu")

    def test_audit_labels_cannot_influence_fit_or_development_metric(self):
        noisy, clean, labels = self._data()
        changed = clean.copy()
        changed[labels == 2] += 100000.0
        config = {"epochs": 1, "batch_size": 8, "hidden": 8, "chunk_size": 16, "seed": 9}
        first = train(noisy, clean, labels, config=config, device_name="cpu")
        second = train(noisy, changed, labels, config=config, device_name="cpu")
        self.assertEqual(first["development_rmse_original_units"], second["development_rmse_original_units"])
        self.assertIsNone(first["audit_metrics"])

    def test_chunked_resume_matches_uninterrupted_epochs_and_ignores_audit_nan(self):
        noisy, clean, labels = self._data()
        clean[labels == 2] = np.nan
        config = {"epochs": 2, "batch_size": 8, "hidden": 8, "chunk_size": 16, "seed": 9}
        uninterrupted = train(noisy, clean, labels, config=config, device_name="cpu")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            first = train(noisy, clean, labels, config=config, checkpoint=checkpoint,
                          device_name="cpu", max_epochs_this_run=1)
            resumed = train(noisy, clean, labels, config=config, checkpoint=checkpoint,
                            resume=True, device_name="cpu", max_epochs_this_run=1)
        self.assertEqual(first["epoch"], 1)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["step"], uninterrupted["step"])
        self.assertAlmostEqual(resumed["development_rmse_original_units"], uninterrupted["development_rmse_original_units"], places=7)


if __name__ == "__main__":
    unittest.main()
