#!/usr/bin/env python3
"""Read-only Team 23 input preflight and manifest writer."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "23_Team_Toxic"
MANIFEST_NAME = "manifest.json"
REQUIRED_FILES = (
    "boilerplate.ipynb",
    "test_weights_noisy.npy",
    "train_weights_clean.npy",
    "train_weights_noisy.npy",
    "test_queries.csv",
    "train_parallel_corpus.csv",
)
EXPECTED_ARRAYS = {
    "train_weights_clean.npy": 5_923_260,
    "train_weights_noisy.npy": 5_923_260,
    "test_weights_noisy.npy": 2_961_630,
}
EXPECTED_ROWS = {"test_queries.csv": 200, "train_parallel_corpus.csv": 300}
CHUNK_ELEMENTS = 1_048_576
HASH_CHUNK_BYTES = 1024 * 1024


class PreflightError(Exception):
    """A user-correctable input or validation failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise PreflightError(f"cannot hash {path.name}: {exc}") from exc
    return digest.hexdigest()


def npy_header(path: Path) -> tuple[dict[str, Any], int, str]:
    """Read only the NPY header; never enables pickle or memory mapping."""
    try:
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise PreflightError(f"{path.name}: invalid NPY magic")
            version_bytes = handle.read(2)
            if len(version_bytes) != 2:
                raise PreflightError(f"{path.name}: truncated NPY version")
            major, minor = version_bytes
            if (major, minor) == (1, 0):
                length_bytes = handle.read(2)
                length_format = "<H"
            elif (major, minor) in {(2, 0), (3, 0)}:
                length_bytes = handle.read(4)
                length_format = "<I"
            else:
                raise PreflightError(f"{path.name}: unsupported NPY version {(major, minor)}")
            if len(length_bytes) != struct.calcsize(length_format):
                raise PreflightError(f"{path.name}: truncated NPY header length")
            header_length = struct.unpack(length_format, length_bytes)[0]
            if header_length > 10_000:
                raise PreflightError(f"{path.name}: NPY header exceeds 10000 bytes")
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise PreflightError(f"{path.name}: truncated NPY header")
            try:
                metadata = ast.literal_eval(raw_header.decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
                raise PreflightError(f"{path.name}: malformed NPY header: {exc}") from exc
            if not isinstance(metadata, dict):
                raise PreflightError(f"{path.name}: NPY header is not a dictionary")
            offset = handle.tell()
    except OSError as exc:
        raise PreflightError(f"{path.name}: invalid NPY header: {exc}") from exc
    return metadata, offset, f"{major}.{minor}"


def inspect_npy(path: Path, expected_length: int) -> dict[str, Any]:
    metadata, offset, version = npy_header(path)
    descr = metadata.get("descr")
    shape = metadata.get("shape")
    if descr != "<f4":
        raise PreflightError(f"{path.name}: expected little-endian float32, got {descr!r}")
    if metadata.get("fortran_order") not in (False,):
        raise PreflightError(f"{path.name}: expected a C-order 1D array")
    if not isinstance(shape, tuple) or len(shape) != 1 or shape[0] != expected_length:
        raise PreflightError(f"{path.name}: expected shape ({expected_length},), got {shape!r}")

    finite_checked = 0
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            while finite_checked < expected_length:
                count = min(CHUNK_ELEMENTS, expected_length - finite_checked)
                raw = handle.read(count * 4)
                if len(raw) != count * 4:
                    raise PreflightError(
                        f"{path.name}: truncated payload at element {finite_checked} "
                        f"(expected {expected_length})"
                    )
                values = struct.unpack(f"<{count}f", raw)
                if not all(math.isfinite(value) for value in values):
                    raise PreflightError(f"{path.name}: non-finite value found near element {finite_checked}")
                finite_checked += count
    except OSError as exc:
        raise PreflightError(f"{path.name}: cannot read payload: {exc}") from exc
    return {
        "kind": "npy",
        "sha256": sha256_file(path),
        "dtype": "float32",
        "shape": [expected_length],
        "npy_version": version,
        "finite_values_checked": finite_checked,
    }


def inspect_csv(path: Path, expected_rows: int) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise PreflightError(f"{path.name}: empty CSV") from exc
            if not header or any(not column.strip() for column in header):
                raise PreflightError(f"{path.name}: header contains an empty column name")
            if len(set(header)) != len(header):
                raise PreflightError(f"{path.name}: duplicate header names")
            try:
                id_index = next(index for index, name in enumerate(header) if name.strip().lower() == "id")
            except StopIteration as exc:
                raise PreflightError(f"{path.name}: CSV header must contain an id column") from exc
            ids: set[str] = set()
            rows = 0
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise PreflightError(f"{path.name}: row {row_number} has {len(row)} fields, expected {len(header)}")
                identifier = row[id_index].strip()
                if not identifier:
                    raise PreflightError(f"{path.name}: empty id at row {row_number}")
                if identifier in ids:
                    raise PreflightError(f"{path.name}: duplicate id {identifier!r} at row {row_number}")
                ids.add(identifier)
                rows += 1
    except UnicodeDecodeError as exc:
        raise PreflightError(f"{path.name}: invalid UTF-8 CSV: {exc}") from exc
    except OSError as exc:
        raise PreflightError(f"{path.name}: cannot read CSV: {exc}") from exc
    if rows != expected_rows:
        raise PreflightError(f"{path.name}: expected {expected_rows} data rows, found {rows}")
    return {
        "kind": "csv",
        "sha256": sha256_file(path),
        "header": header,
        "row_count": rows,
        "unique_ids": len(ids),
    }


def inspect_notebook(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{path.name}: invalid notebook JSON: {exc}") from exc
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise PreflightError(f"{path.name}: notebook JSON has no cells list")
    return {"kind": "notebook", "sha256": sha256_file(path), "cell_count": len(cells)}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run(data_dir: Path) -> int:
    manifest_path = data_dir / MANIFEST_NAME
    # A prior success must not be mistaken for the result of a failed current run.
    try:
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest_path.unlink()
    except OSError as exc:
        print(f"preflight error: cannot remove old manifest: {exc}", file=sys.stderr)
        return 1

    try:
        if not data_dir.is_dir():
            raise PreflightError(f"data directory does not exist: {data_dir}")
        present = {entry.name for entry in data_dir.iterdir()}
        required = set(REQUIRED_FILES)
        missing = sorted(required - present)
        unexpected = sorted(present - required - {MANIFEST_NAME})
        if missing:
            raise PreflightError("missing required file(s): " + ", ".join(missing))
        if unexpected:
            raise PreflightError("unexpected input file(s): " + ", ".join(unexpected))

        files: dict[str, Any] = {}
        for name in REQUIRED_FILES:
            path = data_dir / name
            if name in EXPECTED_ARRAYS:
                files[name] = inspect_npy(path, EXPECTED_ARRAYS[name])
            elif name in EXPECTED_ROWS:
                files[name] = inspect_csv(path, EXPECTED_ROWS[name])
            else:
                files[name] = inspect_notebook(path)
        manifest = {
            "schema_version": 1,
            "team": 23,
            "data_directory": str(data_dir.resolve()),
            "required_files": list(REQUIRED_FILES),
            "files": files,
        }
        atomic_write_json(manifest_path, manifest)
    except (OSError, PreflightError) as exc:
        print(f"preflight error: {exc}", file=sys.stderr)
        return 1
    print(f"preflight ok: wrote {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Team 23 input directory")
    args = parser.parse_args(argv)
    return run(args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
