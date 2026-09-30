#!/usr/bin/env python3
"""Create a deterministic, source-only grouped split for the Team 23 corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "data" / "23_Team_Toxic" / "train_parallel_corpus.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "23_Team_Toxic" / "translation_split.json"
RULE_VERSION = "source-group-v1"
DEFAULT_SEED = 2301
TARGETS_300 = {"fitting": 210, "development": 45, "audit": 45}
_DIGITS = re.compile(r"\d+")
_HEX_TOKEN = re.compile(r"(?i)(?=[a-f0-9]{8,}\b)[a-f0-9]+\b")
_WHITESPACE = re.compile(r"\s+")


class SplitError(ValueError):
    """Raised when the corpus cannot safely receive a split."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source(source: str) -> str:
    """Canonical source form used for conservative grouping."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", source)).strip().casefold()


def template_source(source: str) -> str:
    """Mask only digit runs and long hex-like tokens after canonicalization."""
    normalized = normalize_source(source)
    masked = _HEX_TOKEN.sub("<HEX>", normalized)
    return _DIGITS.sub("<DIGITS>", masked)


def _source_keys(source: str) -> tuple[str, str, str]:
    return (f"exact:{source}", f"normalized:{normalize_source(source)}", f"template:{template_source(source)}")


def read_rows(path: Path, expected_rows: int | None = 300) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "asdfghjkl" not in reader.fieldnames:
            raise SplitError("corpus must contain id and asdfghjkl columns")
        rows = list(reader)
    if expected_rows is not None and len(rows) != expected_rows:
        raise SplitError(f"expected {expected_rows} rows, found {len(rows)}")
    ids = [row.get("id", "").strip() for row in rows]
    if any(not identifier for identifier in ids):
        raise SplitError("corpus contains an empty id")
    if len(set(ids)) != len(ids):
        raise SplitError("corpus contains duplicate ids")
    if any(not row.get("asdfghjkl", "").strip() for row in rows):
        raise SplitError("corpus contains an empty source")
    return rows


def _groups(rows: list[dict[str, str]]) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    first_by_key: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in _source_keys(row["asdfghjkl"]):
            previous = first_by_key.setdefault(key, index)
            union(index, previous)
    grouped: dict[int, list[int]] = {}
    for index in range(len(rows)):
        grouped.setdefault(find(index), []).append(index)
    return sorted((sorted(indices) for indices in grouped.values()), key=lambda x: x[0])


def _scaled_targets(total: int) -> dict[str, int]:
    if total == 300:
        return dict(TARGETS_300)
    development = round(total * 0.15)
    audit = round(total * 0.15)
    return {"fitting": total - development - audit, "development": development, "audit": audit}


def assign_groups(rows: list[dict[str, str]], seed: int = DEFAULT_SEED, targets: dict[str, int] | None = None) -> dict[str, list[str]]:
    """Assign complete source groups with a stable seeded ordering."""
    targets = dict(targets or _scaled_targets(len(rows)))
    if set(targets) != set(TARGETS_300) or sum(targets.values()) != len(rows):
        raise SplitError("split targets must cover every row exactly once")
    names = tuple(TARGETS_300)
    groups = _groups(rows)
    order = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}\0{','.join(rows[i]['id'] for i in group)}".encode("utf-8")
        ).hexdigest(),
    )
    order.sort(key=len, reverse=True)
    counts = {name: 0 for name in names}
    assigned = {name: [] for name in names}
    for group in order:
        size = len(group)
        fitting = [name for name in names if counts[name] + size <= targets[name]]
        if not fitting:
            raise SplitError("source group cannot fit any remaining split target")
        candidates = fitting
        chosen = min(
            candidates,
            key=lambda name: (
                counts[name] / targets[name] if targets[name] else float("inf"),
                hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).hexdigest(),
            ),
        )
        counts[chosen] += size
        assigned[chosen].extend(rows[i]["id"] for i in group)
    return {name: sorted(identifiers) for name, identifiers in assigned.items()}


def make_split(corpus: Path, seed: int = DEFAULT_SEED, expected_rows: int | None = 300) -> dict:
    if corpus.resolve() == DEFAULT_CORPUS.resolve():
        manifest_path = corpus.parent / "manifest.json"
        if not manifest_path.is_file():
            raise SplitError("Team 23 input manifest is missing; run preflight first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = manifest["files"][corpus.name]["sha256"]
        if sha256_file(corpus) != expected_hash:
            raise SplitError("Team 23 corpus hash differs from preflight manifest")
    rows = read_rows(corpus, expected_rows=expected_rows)
    assignments = assign_groups(rows, seed=seed)
    all_ids = sorted(row["id"].strip() for row in rows)
    split_ids = sorted(identifier for identifiers in assignments.values() for identifier in identifiers)
    if split_ids != all_ids or any(not identifiers for identifiers in assignments.values()):
        raise SplitError("split validation failed: IDs are not exhaustive or a split is empty")
    return {
        "schema_version": 1,
        "rule_version": RULE_VERSION,
        "seed": seed,
        "source_corpus": corpus.name,
        "source_corpus_sha256": sha256_file(corpus),
        "row_count": len(rows),
        "group_count": len(_groups(rows)),
        "counts": {name: len(identifiers) for name, identifiers in assignments.items()},
        "splits": assignments,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-rows", type=int, default=300, help="row-count gate (use only for synthetic tests)")
    args = parser.parse_args()
    try:
        payload = make_split(args.corpus, seed=args.seed, expected_rows=args.expected_rows)
        write_json(args.output, payload)
    except (OSError, SplitError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.output}: " + ", ".join(f"{k}={v}" for k, v in payload["counts"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
