from __future__ import annotations

import csv
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_team23.py"
REQUIRED = (
    "boilerplate.ipynb",
    "test_weights_noisy.npy",
    "train_weights_clean.npy",
    "train_weights_noisy.npy",
    "test_queries.csv",
    "train_parallel_corpus.csv",
)
LENGTHS = {
    "test_weights_noisy.npy": 2_961_630,
    "train_weights_clean.npy": 5_923_260,
    "train_weights_noisy.npy": 5_923_260,
}


def invoke(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--data-dir", str(directory)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


def write_sparse_npy(path: Path, length: int, *, value: float | None = None) -> None:
    with path.open("wb") as handle:
        header = repr({"descr": "<f4", "fortran_order": False, "shape": (length,)})
        prefix_length = 6 + 2 + 2
        padding = 64 - ((prefix_length + len(header) + 1) % 64)
        encoded = (header + " " * (padding - 1) + "\n").encode("latin1")
        handle.write(b"\x93NUMPY\x01\x00")
        handle.write(struct.pack("<H", len(encoded)))
        handle.write(encoded)
        offset = handle.tell()
        if value is None:
            handle.truncate(offset + length * 4)
        else:
            handle.write(struct.pack("<f", value))
            handle.truncate(offset + length * 4)


def write_fixture(directory: Path) -> None:
    (directory / "boilerplate.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code"}, {"cell_type": "markdown"}]}), encoding="utf-8"
    )
    for name, length in LENGTHS.items():
        write_sparse_npy(directory / name, length)
    for name, count in (("test_queries.csv", 200), ("train_parallel_corpus.csv", 300)):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "text"])
            for index in range(count):
                writer.writerow([f"{name}-{index}", "synthetic"])


class Team23PreflightTests(unittest.TestCase):
    def test_complete_synthetic_fixture_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory)
            result = invoke(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["team"], 23)
            self.assertEqual(manifest["files"]["train_weights_clean.npy"]["shape"], [5_923_260])
            self.assertEqual(manifest["files"]["test_queries.csv"]["row_count"], 200)
            self.assertEqual(manifest["files"]["boilerplate.ipynb"]["cell_count"], 2)
            expected_hash = hashlib.sha256((directory / "boilerplate.ipynb").read_bytes()).hexdigest()
            self.assertEqual(manifest["files"]["boilerplate.ipynb"]["sha256"], expected_hash)

    def test_missing_input_is_nonzero_and_does_not_write_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory)
            (directory / "test_queries.csv").unlink()
            result = invoke(directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required file", result.stderr)
            self.assertFalse((directory / "manifest.json").exists())

    def test_duplicate_csv_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory)
            with (directory / "test_queries.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "text"])
                for index in range(200):
                    writer.writerow(["same" if index in (0, 199) else str(index), "synthetic"])
            result = invoke(directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate id", result.stderr)
            self.assertFalse((directory / "manifest.json").exists())

    def test_nonfinite_npy_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_fixture(directory)
            write_sparse_npy(directory / "test_weights_noisy.npy", LENGTHS["test_weights_noisy.npy"], value=float("nan"))
            result = invoke(directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-finite", result.stderr)
            self.assertFalse((directory / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
