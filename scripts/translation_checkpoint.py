"""Fail-closed checks for locally produced translation trainer checkpoints."""

from __future__ import annotations

import math
from typing import Any

import torch


def _check_finite(value: Any, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite checkpoint tensor: {path}")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite checkpoint value: {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")


def validate_checkpoint(
    payload: Any,
    *,
    input_hashes: dict[str, str],
    split_ids: dict[str, list[str]],
    parameter_count: int,
    special_ids: dict[str, int],
) -> None:
    """Validate inference provenance and the complete recovery envelope.

    This does not deserialize untrusted files or replace strict model loading.
    It deliberately permits legacy v1 checkpoints without newer lineage fields.
    """
    required = {"contract", "model", "optimizer", "scheduler", "epoch", "step",
                "rng", "config", "input_hashes", "split_ids"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("translation checkpoint is missing required contract fields")
    if payload["contract"] != "translation-trainer-v1":
        raise ValueError("translation checkpoint contract is unsupported")
    for key in ("epoch", "step"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"checkpoint {key} must be a nonnegative integer")
    if payload["input_hashes"] != input_hashes or payload["split_ids"] != split_ids:
        raise ValueError("translation checkpoint inputs or split IDs do not match")
    config = payload["config"]
    if (not isinstance(config, dict)
            or config.get("trainer") != "translation-trainer-v1"
            or config.get("architecture") != "supplied_TranslationTransformer_v1"
            or config.get("parameter_count") != parameter_count
            or config.get("smoke") is not False
            or config.get("special_ids") != special_ids):
        raise ValueError("translation checkpoint configuration is incompatible")
    model = payload["model"]
    if (not isinstance(model, dict) or not model
            or any(not isinstance(v, torch.Tensor) for v in model.values())):
        raise ValueError("translation checkpoint model state must contain tensors")
    optimizer, scheduler = payload["optimizer"], payload["scheduler"]
    if (not isinstance(optimizer, dict) or not {"state", "param_groups"}.issubset(optimizer)
            or not isinstance(optimizer["state"], dict)
            or not isinstance(optimizer["param_groups"], list)
            or not optimizer["param_groups"]
            or not isinstance(scheduler, dict) or "last_epoch" not in scheduler):
        raise ValueError("translation checkpoint optimizer/scheduler state is incomplete")
    rng = payload["rng"]
    if (not isinstance(rng, dict) or not {"python", "numpy", "torch"}.issubset(rng)
            or not isinstance(rng["python"], tuple)
            or not isinstance(rng["numpy"], tuple)
            or not isinstance(rng["torch"], torch.Tensor)
            or rng["torch"].dtype != torch.uint8 or rng["torch"].ndim != 1):
        raise ValueError("translation checkpoint RNG state is incomplete")
    for key in ("model", "optimizer", "scheduler"):
        _check_finite(payload[key], key)
