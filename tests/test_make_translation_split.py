from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.make_translation_split import SplitError, assign_groups
from scripts.validate_translation_split import validate_frozen_split

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_translation_split.py"


def write_corpus(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "asdfghjkl", "english"])
        writer.writerows(rows)


def synthetic_rows() -> list[tuple[str, str, str]]:
    pairs = [
        ("Alpha 123", "label one"),
        (" alpha   123 ", "different label"),
        ("beta item 12345", "label three"),
        ("beta item 67890", "different label"),
        ("gamma 111", "label five"),
        ("gamma 222", "different label"),
        ("delta 1", "label seven"),
        ("delta 2", "different label"),
        ("epsilon 333", "label nine"),
        ("epsilon 444", "different label"),
        ("zeta 555", "label eleven"),
        ("zeta 666", "different label"),
    ]
    return [(f"row-{index:02d}", source, english) for index, (source, english) in enumerate(pairs)]


def invoke(corpus: Path, output: Path, seed: int = 2301) -> subprocess.CompletedProcess[str]:
    with corpus.open(encoding="utf-8") as handle:
        expected_rows = sum(1 for _ in handle) - 1
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--seed",
            str(seed),
            "--expected-rows",
            str(expected_rows),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


class TranslationSplitTests(unittest.TestCase):
    def test_oversized_source_group_fails_closed(self):
        rows = [{"id": f"row-{index}", "asdfghjkl": "same source", "english": f"target {index}"} for index in range(3)]
        with self.assertRaises(SplitError):
            assign_groups(rows, targets={"fitting": 1, "development": 1, "audit": 1})

    def test_deterministic_and_source_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            corpus = directory / "corpus.csv"
            output_a = directory / "a.json"
            output_b = directory / "b.json"
            rows = synthetic_rows()
            write_corpus(corpus, rows)
            self.assertEqual(invoke(corpus, output_a).returncode, 0)
            first = json.loads(output_a.read_text(encoding="utf-8"))
            self.assertEqual(invoke(corpus, output_b).returncode, 0)
            repeated = json.loads(output_b.read_text(encoding="utf-8"))
            self.assertEqual(first["splits"], repeated["splits"])
            write_corpus(corpus, [(identifier, source, "changed entirely") for identifier, source, _ in rows])
            self.assertEqual(invoke(corpus, output_b).returncode, 0)
            second = json.loads(output_b.read_text(encoding="utf-8"))
            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(first["source_corpus_sha256"] != second["source_corpus_sha256"], True)

    def test_groups_do_not_leak_and_ids_are_disjoint_exhaustive(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            corpus, output = directory / "corpus.csv", directory / "split.json"
            rows = synthetic_rows()
            write_corpus(corpus, rows)
            result = invoke(corpus, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            splits = payload["splits"]
            self.assertEqual(payload["counts"], {"fitting": 8, "development": 2, "audit": 2})
            self.assertEqual(set().union(*(set(values) for values in splits.values())), {row[0] for row in rows})
            self.assertEqual(sum(map(len, splits.values())), len(rows))
            self.assertEqual(len(set().union(*(set(values) for values in splits.values()))), len(rows))
            location = {identifier: name for name, values in splits.items() for identifier in values}
            for left, right in (("row-00", "row-01"), ("row-02", "row-03"), ("row-04", "row-05")):
                self.assertEqual(location[left], location[right])

    def test_real_corpus_has_expected_hash_and_counts(self):
        corpus = ROOT / "data" / "23_Team_Toxic" / "train_parallel_corpus.csv"
        if not corpus.exists():
            self.skipTest("Team 23 corpus is not present")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "split.json"
            result = invoke(corpus, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], 300)
            self.assertEqual(payload["counts"], {"fitting": 210, "development": 45, "audit": 45})
            self.assertEqual(payload["source_corpus_sha256"], hashlib.sha256(corpus.read_bytes()).hexdigest())

    def test_frozen_team23_split_rejects_partition_changes(self):
        corpus = ROOT / "data" / "23_Team_Toxic" / "train_parallel_corpus.csv"
        split_path = corpus.parent / "translation_split.json"
        if not corpus.exists() or not split_path.exists():
            self.skipTest("Team 23 frozen split is not present")
        self.assertEqual(validate_frozen_split(corpus, split_path)["counts"], {"fitting": 210, "development": 45, "audit": 45})
        with tempfile.TemporaryDirectory() as directory:
            altered = json.loads(split_path.read_text(encoding="utf-8"))
            altered["splits"]["fitting"][0], altered["splits"]["audit"][0] = altered["splits"]["audit"][0], altered["splits"]["fitting"][0]
            modified_path = Path(directory) / "altered.json"
            modified_path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_frozen_split(corpus, modified_path)


if __name__ == "__main__":
    unittest.main()
