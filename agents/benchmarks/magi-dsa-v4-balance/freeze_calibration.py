#!/usr/bin/env python3
"""Render validated B300 calibration JSON as a source-controlled module.

This command belongs between the calibration and measure revisions.  It never
changes a running benchmark: generate the module, review and commit it, then
rebuild the immutable native image before invoking ``driver.py measure``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "magi_attention/meta/solver/dsa_calibration.py"
WEIGHT_NAMES = (
    "token_weight",
    "indexer_weight",
    "fragment_overhead",
    "window_transfer_weight",
    "overlap_transfer_weight",
    "compressed_owner_send_weight",
    "compressed_remote_receive_weight",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate(
    payload: Mapping[str, Any], *, allow_sampled_non_formal: bool = False
) -> dict[int, dict[str, float]]:
    if payload.get("schema_version") != 1:
        raise ValueError("calibration schema_version must be 1")
    target = payload.get("target")
    if target == "sampled-b300-sm103":
        if not allow_sampled_non_formal:
            raise ValueError("sampled calibration requires --allow-sampled-non-formal")
        if payload.get("formal") is not False:
            raise ValueError("sampled calibration must declare formal=false")
        if payload.get("scope") != "sampled_non_formal":
            raise ValueError("sampled calibration scope must be sampled_non_formal")
    elif target != "b300-sm103":
        raise ValueError("calibration target must be b300-sm103 or sampled-b300-sm103")
    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != {"0", "4", "128"}:
        raise ValueError("weights must contain exactly ratios 0, 4 and 128")
    weights: dict[int, dict[str, float]] = {}
    for ratio in (0, 4, 128):
        values = raw_weights[str(ratio)]
        if not isinstance(values, dict) or set(values) != set(WEIGHT_NAMES):
            raise ValueError(f"ratio {ratio} has an invalid coefficient set")
        normalized = {name: float(values[name]) for name in WEIGHT_NAMES}
        if any(value < 0 or not value < float("inf") for value in normalized.values()):
            raise ValueError(
                f"ratio {ratio} coefficients must be finite and non-negative"
            )
        if ratio != 4 and normalized["indexer_weight"] != 0.0:
            raise ValueError(f"ratio {ratio} must have zero indexer_weight")
        weights[ratio] = normalized
    return weights


def _render(
    payload: Mapping[str, Any], weights: Mapping[int, Mapping[str, float]]
) -> str:
    target = str(payload["target"])
    normalized_payload = {
        "schema_version": 1,
        "target": target,
        "weights": {str(ratio): dict(weights[ratio]) for ratio in sorted(weights)},
    }
    calibration_id = hashlib.sha256(_canonical_bytes(normalized_payload)).hexdigest()
    source_lines = [
        "# Copyright (c) 2025-2026 SandAI. All Rights Reserved.",
        "#",
        '# Licensed under the Apache License, Version 2.0 (the "License");',
        "# you may not use this file except in compliance with the License.",
        "# You may obtain a copy of the License at",
        "#",
        "#     http://www.apache.org/licenses/LICENSE-2.0",
        "#",
        "# Unless required by applicable law or agreed to in writing, software",
        '# distributed under the License is distributed on an "AS IS" BASIS,',
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "# See the License for the specific language governing permissions and",
        "# limitations under the License.",
        "",
        '"""Frozen B300/SM103 cost calibration used by the DSA dispatch solver."""',
        "",
        "from __future__ import annotations",
        "",
        "import hashlib",
        "import json",
        "from types import MappingProxyType",
        "",
        "from magi_attention.meta.solver.dsa_dispatch import DsaCostModel",
        "",
        "CALIBRATION_SCHEMA_VERSION = 1",
        f"CALIBRATION_TARGET = {json.dumps(target)}",
        f'# Generated from run {payload.get("source_run_id", "unknown")!r}.',
        "_WEIGHTS = {",
    ]
    for ratio in (0, 4, 128):
        source_lines.append(f"    {ratio}: {{")
        for name in WEIGHT_NAMES:
            source_lines.append(
                f"        {json.dumps(name)}: {weights[ratio][name]!r},"
            )
        source_lines.append("    },")
    source_lines.extend(
        [
            "}",
            "",
            "",
            "def _normalized_payload() -> dict[str, object]:",
            "    return {",
            '        "schema_version": CALIBRATION_SCHEMA_VERSION,',
            '        "target": CALIBRATION_TARGET,',
            '        "weights": {str(ratio): _WEIGHTS[ratio] for ratio in sorted(_WEIGHTS)},',
            "    }",
            "",
            "",
            "CALIBRATION_ID = hashlib.sha256(",
            '    json.dumps(_normalized_payload(), sort_keys=True, separators=(",", ":")).encode()',
            ").hexdigest()",
            "assert (",
            f'    CALIBRATION_ID == "{calibration_id}"',
            ")",
            "CALIBRATED_WEIGHTS = MappingProxyType(",
            "    {ratio: MappingProxyType(dict(values)) for ratio, values in _WEIGHTS.items()}",
            ")",
            "",
            "",
            "def get_dsa_cost_model(",
            "    compress_ratio: int,",
            "    *,",
            "    token_memory_bytes: int,",
            "    compressed_block_memory_bytes: int,",
            "    remote_row_memory_bytes: int,",
            ") -> DsaCostModel:",
            "    try:",
            "        weights = dict(CALIBRATED_WEIGHTS[compress_ratio])",
            "    except KeyError as error:",
            '        raise ValueError(f"unsupported DSA compress ratio {compress_ratio}") from error',
            "    return DsaCostModel(",
            "        **weights,",
            "        token_memory_bytes=token_memory_bytes,",
            "        compressed_block_memory_bytes=compressed_block_memory_bytes,",
            "        compressed_owner_send_memory_bytes=compressed_block_memory_bytes,",
            "        compressed_remote_receive_memory_bytes=compressed_block_memory_bytes,",
            "        compressed_global_memory_bytes=compressed_block_memory_bytes,",
            "        remote_row_memory_bytes=remote_row_memory_bytes,",
            "    )",
            "",
            "",
            "def calibration_manifest() -> dict[str, object]:",
            '    return {**_normalized_payload(), "calibration_id": CALIBRATION_ID}',
            "",
            "",
            "__all__ = [",
            '    "CALIBRATED_WEIGHTS",',
            '    "CALIBRATION_ID",',
            '    "CALIBRATION_SCHEMA_VERSION",',
            '    "CALIBRATION_TARGET",',
            '    "calibration_manifest",',
            '    "get_dsa_cost_model",',
            "]",
            "",
        ]
    )
    return "\n".join(source_lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output already matches instead of writing it",
    )
    parser.add_argument(
        "--allow-sampled-non-formal",
        action="store_true",
        help=(
            "explicitly permit a formal=false, scope=sampled_non_formal "
            "sampled-b300-sm103 input"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    weights = _validate(payload, allow_sampled_non_formal=args.allow_sampled_non_formal)
    source = _render(payload, weights)
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != source
        ):
            raise SystemExit(f"{args.output} does not match {args.input}")
        print(f"calibration source is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote frozen calibration module: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
