"""Fail closed unless the locally frozen Team 23 split is exactly reproduced."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.make_translation_split import make_split, sha256_file

APPROVED_SPLIT_SHA256 = "65fddacd277c3bdb87a7f3bb879dda220957ae5f8cdef966f5f3b1ee1abf92bf"
APPROVED_COUNTS = {"fitting": 210, "development": 45, "audit": 45}


def validate_frozen_split(corpus: Path, split_path: Path) -> dict:
    if sha256_file(split_path) != APPROVED_SPLIT_SHA256:
        raise ValueError("translation split hash differs from the approved Team 23 split")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("rule_version") != "source-group-v1" or payload.get("counts") != APPROVED_COUNTS:
        raise ValueError("translation split schema, rule, or counts differ from the approved plan")
    expected = make_split(corpus, seed=2301, expected_rows=300)
    if payload != expected:
        raise ValueError("translation split does not match source-only grouped reconstruction")
    return payload
