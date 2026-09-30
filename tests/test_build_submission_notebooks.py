from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.build_submission_notebooks import (
    REQUIRED_FILES,
    build_notebook,
    discover_team23,
    source_bundle,
)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    data = tmp_path / "23_Team_Toxic"
    data.mkdir()
    contents = {
        "boilerplate.ipynb": b"{}",
        "train_weights_clean.npy": b"clean",
        "train_weights_noisy.npy": b"noisy",
        "test_weights_noisy.npy": b"test",
        "train_parallel_corpus.csv": b"id,asdfghjkl,english\n1,a,b\n",
        "test_queries.csv": b"id,asdfghjkl\n1,a\n",
    }
    files = {}
    for name in REQUIRED_FILES:
        path = data / name
        path.write_bytes(contents[name])
        files[name] = hashlib.sha256(contents[name]).hexdigest()
    manifest = {"team": 23, "required_files": list(REQUIRED_FILES), "files": {name: {"sha256": digest} for name, digest in files.items()}}
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data, files


class NotebookBuilderTests(unittest.TestCase):
    def test_shared_output_preserves_part1_and_refuses_raw_or_collisions(self) -> None:
        import ast
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "23_part1.ipynb"
            build_notebook(source, 1)
            document = json.loads(source.read_text())
            bootstrap = ast.parse("".join(document["cells"][1]["source"]))
            helper = next(node for node in bootstrap.body if isinstance(node, ast.FunctionDef) and node.name == "_fresh_output")
            namespace = {"Path": Path}
            exec(compile(ast.Module(body=[helper], type_ignores=[]), "fresh-output", "exec"), namespace)
            working = root / "working"
            raw = root / "23_Team_Toxic"
            working.mkdir()
            raw.mkdir()
            raw = raw.resolve()
            part1 = working / "part1_artifact_metadata.json"
            part1.write_text("preserve")
            guard = namespace["_fresh_output"]
            guard(working, {raw}, forbidden_names=("submission_part2.csv",))
            self.assertEqual(part1.read_text(), "preserve")
            guard(working / "part2_training", {raw}, require_empty=True)
            (working / "submission_part2.csv").write_text("existing")
            with self.assertRaises(RuntimeError):
                guard(working, {raw}, forbidden_names=("submission_part2.csv",))
            with self.assertRaises(RuntimeError):
                guard(raw / "outputs", {raw})

    def test_discovery_accepts_manifestless_and_unordered_manifest(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data, hashes = _fixture(tmp_path)
            (data / "manifest.json").unlink()
            found, evidence = discover_team23(search_roots=[tmp_path], expected_hashes=hashes)
            self.assertEqual(found, data.resolve())
            self.assertEqual(evidence["manifest_status"], "manifest_unavailable")
            manifest = {"team": 23, "required_files": list(reversed(REQUIRED_FILES)), "files": {name: {"sha256": digest} for name, digest in hashes.items()}}
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            found, evidence = discover_team23(search_roots=[tmp_path], expected_hashes=hashes)
            self.assertEqual(found, data.resolve())
            self.assertEqual(evidence["manifest_status"], "verified")

            manifest["team"] = "23"
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integer Team 23"):
                discover_team23(search_roots=[tmp_path], expected_hashes=hashes)

    def test_discovery_accepts_exact_manifest_and_rejects_missing_identity(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data, hashes = _fixture(tmp_path)
            found, evidence = discover_team23(search_roots=[tmp_path], expected_hashes=hashes)
            self.assertEqual(found, data.resolve())
            self.assertEqual(evidence["team"], 23)
            manifest = json.loads((data / "manifest.json").read_text())
            manifest["team"] = "23"
            (data / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "integer Team 23"):
                discover_team23(search_roots=[tmp_path], expected_hashes=hashes)

    def test_discovery_rejects_duplicate_and_hash_tamper(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            first, hashes = _fixture(tmp_path)
            second = tmp_path / "nested" / "23_Team_Toxic"
            second.parent.mkdir()
            second.mkdir()
            for name in REQUIRED_FILES:
                (second / name).write_bytes((first / name).read_bytes())
            manifest = json.loads((first / "manifest.json").read_text())
            (second / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                discover_team23(search_roots=[tmp_path], expected_hashes=hashes)
            (second / "manifest.json").unlink()
            (first / "test_queries.csv").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                discover_team23(search_roots=[first], expected_hashes=hashes)

    def test_builder_emits_standalone_source_bundle_not_repo_imports(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            part1 = tmp_path / "23_part1.ipynb"
            part2 = tmp_path / "23_part2.ipynb"
            build_notebook(part1, 1)
            build_notebook(part2, 2)
            for path in (part1, part2):
                payload = json.loads(path.read_text(encoding="utf-8"))
                cell_ids = [cell["id"] for cell in payload["cells"]]
                self.assertEqual(len(cell_ids), len(set(cell_ids)))
                for cell_id in cell_ids:
                    self.assertRegex(cell_id, r"^[a-zA-Z0-9_-]{1,64}$")
                source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
                self.assertIn("SOURCE_BUNDLE_SHA256", source)
                self.assertIn("from scripts.", source)
                self.assertIn("sys.path.insert(0, str(runtime))", source)
                self.assertNotIn("git rev-parse", source)
                self.assertIn("Team 23 submission metadata is supplied privately", source)
                self.assertIn("no personal names, SRNs, or contribution assignments", source)
                self.assertNotIn("PERSONAL_DELIVERY_RECORD", source)
                self.assertNotIn("<raw data>", source)
                self.assertNotIn('part1.get("input_sha256", {}).get("train_weights_noisy.npy")', source)
                if path.name == "23_part2.ipynb":
                    self.assertIn("queries = pd.read_csv(queries_path, dtype=str, keep_default_na=False)", source)
                    self.assertIn("training_candidate in {part1_dir, data_dir}", source)
                    self.assertIn("_fresh_output(training_candidate, {data_dir.resolve()}, require_empty=True)", source)
            bundle, digest = source_bundle()
            self.assertEqual(hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), digest)

    def test_notebook_cells_compile_without_executing_training(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "23_part1.ipynb"
            build_notebook(path, 1)
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")

    def test_part2_handoff_gate_accepts_valid_sidecar_and_rejects_wrong_hash(self) -> None:
        import tempfile
        import types
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            notebook_path = tmp_path / "23_part2.ipynb"
            build_notebook(notebook_path, 2)
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            gate_source = next(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "def validate_part1_handoff" in "".join(cell["source"])
            )
            prefix = gate_source.split("restored_hash = validate_part1_handoff", 1)[0]
            npy_path = tmp_path / "denoised_test_weights.npy"
            csv_path = tmp_path / "submission_part1.csv"
            npy_path.write_bytes(b"synthetic-restored")
            csv_path.write_bytes(b"id,weight\n0,0.0\n")
            hashes = {
                "train_weights_clean.npy": "clean-hash",
                "train_weights_noisy.npy": "noisy-hash",
                "test_weights_noisy.npy": "test-hash",
            }
            namespace = {
                "hashlib": hashlib,
                "Path": Path,
                "EXPECTED_TRAIN_COUNT": 5_923_260,
                "EXPECTED_WEIGHT_COUNT": 2_961_630,
                "SOURCE_BUNDLE_SHA256": "bundle-hash",
                "identity": {"input_sha256": hashes},
                "_sha256": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            fake_module = types.ModuleType("scripts.train_scalar_mlp")
            fake_module.split_labels = lambda *args, **kwargs: (types.SimpleNamespace(tobytes=lambda: b"synthetic-split"), {"spec_sha256": "synthetic-spec"})
            previous_module = sys.modules.get("scripts.train_scalar_mlp")
            sys.modules["scripts.train_scalar_mlp"] = fake_module
            self.addCleanup(lambda: sys.modules.__setitem__("scripts.train_scalar_mlp", previous_module) if previous_module is not None else sys.modules.pop("scripts.train_scalar_mlp", None))
            exec(prefix, namespace)
            config = {"hidden": 32, "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 4096, "epochs": 2, "chunk_size": 65536, "seed": 2301}
            split = namespace["split_for_gate"]
            sidecar = {
                "team": 23,
                "title": "AFML Orange Problem — Team 23",
                "source_bundle_sha256": "bundle-hash",
                "schema_version": 2,
                "method": "schema2_scalar_residual_mlp",
                "shape": [2_961_630],
                "dtype": "float32",
                "submission_rows": 2_961_630,
                "input_sha256": {"clean": "clean-hash", "noisy": "noisy-hash"},
                "test_noisy_sha256": "test-hash",
                "config": config,
                "split_sha256": split["labels_sha256"],
                "split_spec_sha256": split["spec_sha256"],
                "restored_npy_sha256": namespace["_sha256"](npy_path),
                "submission_csv_sha256": namespace["_sha256"](csv_path),
            }
            self.assertEqual(namespace["validate_part1_handoff"](sidecar, npy_path, csv_path), sidecar["restored_npy_sha256"])
            sidecar["input_sha256"]["noisy"] = "wrong-sidecar-hash"
            with self.assertRaisesRegex(RuntimeError, "clean/noisy input hashes"):
                namespace["validate_part1_handoff"](sidecar, npy_path, csv_path)

    def test_builder_cli_writes_both_notebooks(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            completed = subprocess.run([sys.executable, "scripts/build_submission_notebooks.py", "--output-dir", str(tmp_path)], capture_output=True, text=True, check=True)
            self.assertIn("23_part1.ipynb", completed.stdout)
            self.assertIn("23_part2.ipynb", completed.stdout)
            self.assertTrue((tmp_path / "23_part1.ipynb").is_file())
            self.assertTrue((tmp_path / "23_part2.ipynb").is_file())


if __name__ == "__main__":
    unittest.main()
