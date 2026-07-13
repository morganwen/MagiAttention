# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen B300/SM103 cost calibration used by the DSA dispatch solver."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

from magi_attention.meta.solver.dsa_dispatch import DsaCostModel

CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_TARGET = "sampled-b300-sm103"
# Generated from run 'b300-cp8-sample-calibration-20260713T083426Z'.
_WEIGHTS = {
    0: {
        "token_weight": 0.0006550394535421351,
        "indexer_weight": 0.0,
        "fragment_overhead": 0.03995484067374769,
        "window_transfer_weight": 4.025848312478687e-24,
        "overlap_transfer_weight": 0.0,
        "compressed_owner_send_weight": 0.0,
        "compressed_remote_receive_weight": 0.0,
    },
    4: {
        "token_weight": 2.2490712248453346e-22,
        "indexer_weight": 1.4108896307465333e-06,
        "fragment_overhead": 9.259778144943177e-21,
        "window_transfer_weight": 0.08678825866072461,
        "overlap_transfer_weight": 0.662813633177838,
        "compressed_owner_send_weight": 0.11790754080176107,
        "compressed_remote_receive_weight": 0.5247903536818413,
    },
    128: {
        "token_weight": 0.000708633746416635,
        "indexer_weight": 0.0,
        "fragment_overhead": 3.171213766834836e-27,
        "window_transfer_weight": 0.0001664731893305034,
        "overlap_transfer_weight": 0.0,
        "compressed_owner_send_weight": 0.028282658080009817,
        "compressed_remote_receive_weight": 0.19844135449295092,
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
assert (
    CALIBRATION_ID == "df625f1805024f6c7c78e2bdcb07476c8be94347c3e762a80006a87aea0e882d"
)
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
    return {**_normalized_payload(), "calibration_id": CALIBRATION_ID}


__all__ = [
    "CALIBRATED_WEIGHTS",
    "CALIBRATION_ID",
    "CALIBRATION_SCHEMA_VERSION",
    "CALIBRATION_TARGET",
    "calibration_manifest",
    "get_dsa_cost_model",
]
