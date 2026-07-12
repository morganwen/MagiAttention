# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
"""Frozen B300/SM103 cost calibration used by the DSA dispatch solver.

The coefficients are intentionally source-controlled rather than accepted as
runtime overrides. Step-8 calibration produces a signed JSON candidate; a
separate clean commit updates this module before the immutable measure image is
built. ``CALIBRATION_ID`` covers only the normalized predictor coefficients;
exact tensor-memory byte counts remain derived from the public config.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

from magi_attention.meta.solver.dsa_dispatch import DsaCostModel

CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_TARGET = "unfrozen-b300-sm103"

# Bootstrap values used only by the calibration revision. Step 8 replaces
# these with measured B300 coefficients before the formal measure revision.
_WEIGHTS = {
    0: {
        "token_weight": 1.0,
        "indexer_weight": 0.0,
        "fragment_overhead": 64.0,
        "window_transfer_weight": 1.0,
        "overlap_transfer_weight": 0.0,
        "compressed_owner_send_weight": 0.0,
        "compressed_remote_receive_weight": 0.0,
    },
    4: {
        "token_weight": 1.0,
        "indexer_weight": 1.0,
        "fragment_overhead": 64.0,
        "window_transfer_weight": 1.0,
        "overlap_transfer_weight": 1.0,
        "compressed_owner_send_weight": 1.0,
        "compressed_remote_receive_weight": 1.0,
    },
    128: {
        "token_weight": 1.0,
        "indexer_weight": 0.0,
        "fragment_overhead": 64.0,
        "window_transfer_weight": 1.0,
        "overlap_transfer_weight": 0.0,
        "compressed_owner_send_weight": 1.0,
        "compressed_remote_receive_weight": 1.0,
    },
}


def _normalized_payload() -> dict[str, object]:
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "target": CALIBRATION_TARGET,
        "weights": {str(ratio): _WEIGHTS[ratio] for ratio in sorted(_WEIGHTS)},
    }


CALIBRATION_ID = hashlib.sha256(
    json.dumps(_normalized_payload(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
CALIBRATED_WEIGHTS = MappingProxyType(
    {ratio: MappingProxyType(dict(values)) for ratio, values in _WEIGHTS.items()}
)


def get_dsa_cost_model(
    compress_ratio: int,
    *,
    token_memory_bytes: int,
    compressed_block_memory_bytes: int,
    remote_row_memory_bytes: int,
) -> DsaCostModel:
    """Return the immutable calibrated predictor plus exact memory bytes."""

    try:
        weights = dict(CALIBRATED_WEIGHTS[compress_ratio])
    except KeyError as error:
        raise ValueError(f"unsupported DSA compress ratio {compress_ratio}") from error
    return DsaCostModel(
        **weights,
        token_memory_bytes=token_memory_bytes,
        compressed_block_memory_bytes=compressed_block_memory_bytes,
        compressed_owner_send_memory_bytes=compressed_block_memory_bytes,
        compressed_remote_receive_memory_bytes=compressed_block_memory_bytes,
        compressed_global_memory_bytes=compressed_block_memory_bytes,
        remote_row_memory_bytes=remote_row_memory_bytes,
    )


def calibration_manifest() -> dict[str, object]:
    """Return a JSON-serializable copy for benchmark preflight/reporting."""

    return {**_normalized_payload(), "calibration_id": CALIBRATION_ID}


__all__ = [
    "CALIBRATED_WEIGHTS",
    "CALIBRATION_ID",
    "CALIBRATION_SCHEMA_VERSION",
    "CALIBRATION_TARGET",
    "calibration_manifest",
    "get_dsa_cost_model",
]
