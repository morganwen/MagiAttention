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

from __future__ import annotations

import json
import os

import torch
from magi_attn_extensions.DSA.kernels.cutedsl.pack import (
    copy_dsa_rows,
    export_dsa_pack_aot,
    reduce_dsa_rows,
)

from scripts.test.dsa_pack_aot_manifest import COPY_SPECS, REDUCE_SPECS

_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "i32": torch.int32,
}
_DTYPE_TAGS = {dtype: tag for tag, dtype in _TORCH_DTYPES.items()}


def _prewarm_copy(dtype: torch.dtype, width: int) -> None:
    source = (
        torch.arange(4 * width, device="cuda", dtype=torch.float32)
        .reshape(4, width)
        .to(dtype)
    )
    mapping = torch.tensor((2, 0, 3, 2), device="cuda", dtype=torch.int32)
    actual = copy_dsa_rows(source, mapping)
    expected = source.index_select(0, mapping.long())
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.cuda.synchronize()
    print(f"copy dtype={dtype} width={width}: compiled and validated", flush=True)


def _prewarm_reduce(dtype: torch.dtype, width: int) -> None:
    source = (
        torch.arange(4 * width, device="cuda", dtype=torch.float32)
        .reshape(4, width)
        .to(dtype)
    )
    row_offsets = torch.tensor((0, 2, 3, 4), device="cuda", dtype=torch.int32)
    source_rows = torch.tensor((0, 2, 1, 3), device="cuda", dtype=torch.int32)
    actual = reduce_dsa_rows(source, row_offsets, source_rows)
    expected = torch.stack(
        (source[0].float() + source[2].float(), source[1].float(), source[3].float())
    ).to(dtype)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.cuda.synchronize()
    print(f"reduce dtype={dtype} width={width}: compiled and validated", flush=True)


def main() -> None:
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("Magi-DSA CuTe prewarm requires B300 SM103")
    copy_specs = tuple((_TORCH_DTYPES[tag], width) for tag, width in COPY_SPECS)
    reduce_specs = tuple((_TORCH_DTYPES[tag], width) for tag, width in REDUCE_SPECS)
    output_dir = os.environ.get("MAGI_DSA_CUTE_AOT_OUTPUT_DIR")
    if output_dir:
        os.environ["MAGI_DSA_CUTE_AOT_DIR"] = output_dir
        os.environ.pop("MAGI_DSA_CUTE_AOT_REQUIRED", None)
    for dtype, width in copy_specs:
        expected_path = (
            None
            if not output_dir
            else os.path.join(
                output_dir,
                f"copy_sm103_{_DTYPE_TAGS[dtype]}_w{width}.o",
            )
        )
        _prewarm_copy(dtype, width)
        if (
            output_dir
            and expected_path is not None
            and not os.path.isfile(expected_path)
        ):
            path = export_dsa_pack_aot("copy", 10, 3, dtype, width, output_dir)
            print(f"exported {path}", flush=True)
    for dtype, width in reduce_specs:
        expected_path = (
            None
            if not output_dir
            else os.path.join(
                output_dir,
                f"reduce_sm103_{_DTYPE_TAGS[dtype]}_w{width}.o",
            )
        )
        _prewarm_reduce(dtype, width)
        if (
            output_dir
            and expected_path is not None
            and not os.path.isfile(expected_path)
        ):
            path = export_dsa_pack_aot("reduce", 10, 3, dtype, width, output_dir)
            print(f"exported {path}", flush=True)
    print(
        json.dumps(
            {
                "cache_dir": os.environ.get("CUTE_DSL_CACHE_DIR"),
                "aot_input_dir": os.environ.get("MAGI_DSA_CUTE_AOT_DIR"),
                "aot_output_dir": output_dir,
                "copy_specs": [(str(dtype), width) for dtype, width in copy_specs],
                "device": torch.cuda.get_device_name(),
                "reduce_specs": [(str(dtype), width) for dtype, width in reduce_specs],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
