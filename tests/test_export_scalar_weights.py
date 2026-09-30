from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.export_scalar_weights import export
from scripts.train_scalar_mlp import split_labels, train


class ScalarExportTests(unittest.TestCase):
    def _fixture(self, directory: str):
        root = Path(directory)
        data = root / "data"
        data.mkdir()
        total = 16_384
        test_count = 8
        clean = np.linspace(-2, 2, total, dtype=np.float32)
        noisy = (clean + 0.25).astype(np.float32)
        test = np.linspace(-1, 1, test_count, dtype=np.float32)
        for name, values in (("train_weights_clean.npy", clean), ("train_weights_noisy.npy", noisy), ("test_weights_noisy.npy", test)):
            np.save(data / name, values, allow_pickle=False)
        manifest = {"files": {"test_weights_noisy.npy": {"sha256": self._sha(data / "test_weights_noisy.npy")}}}
        (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        labels, _ = split_labels(total)
        checkpoint = root / "model.pt"
        train(noisy, clean, labels, config={"hidden": 4, "epochs": 1, "batch_size": 4, "chunk_size": 4, "seed": 9}, checkpoint=checkpoint, input_hashes={"clean": self._sha(data / "train_weights_clean.npy"), "noisy": self._sha(data / "train_weights_noisy.npy")}, device_name="cpu")
        return data, checkpoint

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_export_has_chunked_npy_csv_and_hash_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            data, checkpoint = self._fixture(directory)
            npy, csv_path = Path(directory) / "restored.npy", Path(directory) / "submission.csv"
            metadata = export(checkpoint, data, npy, csv_path, chunk_size=3, expected_count=8)
            restored = np.load(npy, allow_pickle=False)
            self.assertEqual(restored.dtype, np.float32)
            self.assertEqual(restored.shape, (8,))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["id", "weight"])
            self.assertEqual(len(rows) - 1, 8)
            self.assertEqual(metadata["restored_npy_sha256"], self._sha(npy))
            self.assertEqual(metadata["submission_csv_sha256"], self._sha(csv_path))
            self.assertEqual(metadata["split_sha256"], metadata["split_sha256"].lower())

    def test_rejects_schema_config_hash_split_state_and_nonfinite_failures(self):
        mutations = (
            ("schema_version", 1),
            ("config", {"hidden": 4}),
            ("input_sha256", {"clean": "0" * 64, "noisy": "0" * 64}),
            ("split", {"bad": True}),
            ("model", {}),
            ("optimizer", {}),
            ("scheduler", {}),
            ("rng", {}),
        )
        with tempfile.TemporaryDirectory() as directory:
            data, checkpoint = self._fixture(directory)
            for key, value in mutations:
                payload = torch.load(checkpoint, weights_only=False)
                payload[key] = value
                broken = Path(directory) / f"broken_{key}.pt"
                torch.save(payload, broken)
                with self.assertRaises(ValueError):
                    export(broken, data, Path(directory) / f"{key}.npy", Path(directory) / f"{key}.csv", expected_count=8)
            payload = torch.load(checkpoint, weights_only=False)
            payload["model"]["network.0.weight"][0, 0] = float("nan")
            broken = Path(directory) / "nonfinite.pt"
            torch.save(payload, broken)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                export(broken, data, Path(directory) / "nf.npy", Path(directory) / "nf.csv", expected_count=8)

    def test_rejects_exact_count_and_output_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            data, checkpoint = self._fixture(directory)
            with self.assertRaisesRegex(ValueError, "exact count"):
                export(checkpoint, data, Path(directory) / "wrong.npy", Path(directory) / "wrong.csv", expected_count=7)
            npy, csv_path = Path(directory) / "restored.npy", Path(directory) / "submission.csv"
            export(checkpoint, data, npy, csv_path, expected_count=8)
            csv_path.write_text("id,weight\n1,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                from scripts.export_scalar_weights import _verify_outputs
                _verify_outputs(npy, csv_path, np.load(npy, allow_pickle=False))
            restored = np.load(npy, allow_pickle=False)
            csv_path.write_text("id,weight\n" + "\n".join(f"{i},{float(restored[i])!r}" for i in range(8)) + "\n8,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "more rows"):
                _verify_outputs(npy, csv_path, np.load(npy, allow_pickle=False))

    def test_rejects_aliasing_protected_or_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            data, checkpoint = self._fixture(directory)
            with self.assertRaisesRegex(ValueError, "output paths"):
                export(checkpoint, data, data / "test_weights_noisy.npy", Path(directory) / "submission.csv", expected_count=8)
            with self.assertRaisesRegex(ValueError, "output paths"):
                export(checkpoint, data, data / "boilerplate.ipynb", Path(directory) / "submission.csv", expected_count=8)
            with self.assertRaisesRegex(ValueError, "output paths"):
                export(checkpoint, data, Path(directory) / "same.npy", Path(directory) / "same.json", expected_count=8)


if __name__ == "__main__":
    unittest.main()
