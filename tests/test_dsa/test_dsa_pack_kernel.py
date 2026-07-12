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

"""Architecture-aware DSA packing tests; current acceptance runs on SM103."""

from __future__ import annotations

import importlib.util
import signal
from contextlib import contextmanager

import pytest
import torch


def _dependencies_available() -> bool:
    try:
        return all(
            importlib.util.find_spec(module) is not None
            for module in ("cuda.bindings.driver", "cutlass", "quack", "tvm_ffi")
        )
    except (ImportError, ModuleNotFoundError):
        return False


def _supported_arch_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (
        9,
        10,
    )


pytestmark = [
    pytest.mark.dsa_kernel,
    pytest.mark.skipif(
        not _dependencies_available(),
        reason="DSA packing tests require CUDA Python, CuTe DSL, quack and TVM FFI",
    ),
]


def _packing_api():
    from magi_attention.kernel.cutedsl.dsa_pack import (
        copy_dsa_rows,
        make_dsa_device_copy_map,
        make_dsa_device_reduce_map,
        make_dsa_device_remap_lut,
        reduce_dsa_rows_csr,
        remap_dsa_indices,
    )

    return {
        "copy": copy_dsa_rows,
        "copy_map": make_dsa_device_copy_map,
        "reduce_map": make_dsa_device_reduce_map,
        "remap_lut": make_dsa_device_remap_lut,
        "reduce": reduce_dsa_rows_csr,
        "remap": remap_dsa_indices,
    }


@contextmanager
def _alarm(seconds: int, message: str):
    previous_handler = signal.getsignal(signal.SIGALRM)

    def fail(_signum, _frame):
        raise TimeoutError(message)

    signal.signal(signal.SIGALRM, fail)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def test_static_mapping_builders_reject_invalid_host_plans() -> None:
    api = _packing_api()

    with pytest.raises(ValueError, match="outside"):
        api["copy_map"]([0, 4], 4, "cuda")
    with pytest.raises(ValueError, match="neither -1"):
        api["remap_lut"]([0, -1, 3], 3, "cuda")
    with pytest.raises(ValueError, match="outside"):
        api["reduce_map"]([[0], [4]], 4, "cuda")
    with pytest.raises(TypeError, match="static host metadata"):
        api["copy_map"](torch.tensor([0], dtype=torch.int32), 1, "cuda")
    with pytest.raises(ValueError, match="source_row_count"):
        api["copy_map"]([], -1, "cuda")


@pytest.mark.skipif(not _supported_arch_available(), reason="requires SM90 or SM10x")
@pytest.mark.parametrize(
    ("dtype", "feature_shape"),
    (
        (torch.bfloat16, (128,)),
        (torch.bfloat16, (512,)),
        (torch.bfloat16, (2, 65)),  # scalar tail path after flattening
        (torch.bfloat16, (7168,)),  # representative model hidden width
        (torch.float32, (128,)),
        (torch.int32, (128,)),
    ),
)
def test_destination_to_source_copy_matches_torch(
    dtype: torch.dtype,
    feature_shape: tuple[int, ...],
) -> None:
    api = _packing_api()
    device = torch.device("cuda")
    if dtype.is_floating_point:
        source = torch.randn((32, *feature_shape), dtype=dtype, device=device)
    else:
        source = torch.randint(
            -1000,
            1000,
            (32, *feature_shape),
            dtype=dtype,
            device=device,
        )
    host_map = (9, 0, 9, 17, 3, 31, 1)
    mapping = api["copy_map"](host_map, source.size(0), device)
    actual = api["copy"](source, mapping)
    expected = source.index_select(
        0, torch.tensor(host_map, dtype=torch.int64, device=device)
    )
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _supported_arch_available(), reason="requires SM90 or SM10x")
def test_copy_zero_rows_and_contract_rejections() -> None:
    api = _packing_api()
    device = torch.device("cuda")
    source = torch.empty((0, 128), dtype=torch.bfloat16, device=device)
    empty_map = api["copy_map"]([], 0, device)
    assert tuple(api["copy"](source, empty_map).shape) == (0, 128)

    noncontiguous = torch.empty((4, 256), device=device)[:, ::2]
    mapping = api["copy_map"]([0], 4, device)
    with pytest.raises(ValueError, match="contiguous CUDA"):
        api["copy"](noncontiguous, mapping)
    with pytest.raises(ValueError, match="expects 4 source rows"):
        api["copy"](torch.empty((3, 128), dtype=torch.bfloat16, device=device), mapping)


@pytest.mark.skipif(not _supported_arch_available(), reason="requires SM90 or SM10x")
def test_block_and_token_remap_stays_int32_and_device_resident() -> None:
    api = _packing_api()
    device = torch.device("cuda")
    lut = api["remap_lut"]((5, -1, 1, 3, 0, -1, 2, 4), 6, device)
    logical = torch.tensor(
        [[2, -1, 0, 7], [4, 99, 2, 1]],
        dtype=torch.int32,
        device=device,
    )
    expected = torch.tensor(
        [[1, -1, 5, 4], [0, -1, 1, -1]],
        dtype=torch.int32,
        device=device,
    )
    actual = api["remap"](logical, lut)
    assert actual.dtype == torch.int32
    assert actual.device == logical.device
    assert torch.equal(actual, expected)
    assert torch.equal(api["remap"](logical, lut), actual)


def _reference_csr(
    source: torch.Tensor,
    rows: tuple[tuple[int, ...], ...],
    destination: torch.Tensor,
    *,
    accumulate: bool,
) -> torch.Tensor:
    expected = destination.clone()
    for destination_row, source_rows in enumerate(rows):
        if not source_rows:
            continue
        reduced = source.index_select(
            0,
            torch.tensor(source_rows, dtype=torch.int64, device=source.device),
        ).sum(dim=0, dtype=torch.float32)
        if accumulate:
            expected[destination_row].add_(reduced)
        else:
            expected[destination_row].copy_(reduced)
    return expected


@pytest.mark.skipif(not _supported_arch_available(), reason="requires SM90 or SM10x")
@pytest.mark.parametrize("feature_dim", (128, 130, 512))
@pytest.mark.parametrize("accumulate", (False, True))
def test_fp32_csr_reduce_matches_fixed_order_reference(
    feature_dim: int,
    accumulate: bool,
) -> None:
    api = _packing_api()
    device = torch.device("cuda")
    source = torch.randn((8, feature_dim), dtype=torch.float32, device=device)
    rows = ((), (3,), (1, 1, 4), (2, 5, 2, 2))
    mapping = api["reduce_map"](rows, source.size(0), device)
    destination = torch.randn((4, feature_dim), dtype=torch.float32, device=device)
    expected = _reference_csr(source, rows, destination, accumulate=accumulate)
    actual = api["reduce"](
        source,
        mapping,
        destination.clone(),
        accumulate=accumulate,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    repeated = api["reduce"](
        source,
        mapping,
        destination.clone(),
        accumulate=accumulate,
    )
    assert torch.equal(actual, repeated)


@pytest.mark.skipif(not _supported_arch_available(), reason="requires SM90 or SM10x")
def test_fp32_csr_extreme_fan_in_and_zero_rows() -> None:
    api = _packing_api()
    device = torch.device("cuda")
    source = torch.randn((64, 128), dtype=torch.float32, device=device)
    fan_in = tuple(index % source.size(0) for index in range(4096))
    rows = ((), fan_in, (), (7, 7, 7))
    mapping = api["reduce_map"](rows, source.size(0), device)
    destination = torch.full((4, 128), 13.0, dtype=torch.float32, device=device)
    actual = api["reduce"](source, mapping, destination.clone(), accumulate=False)
    expected = _reference_csr(source, rows, destination, accumulate=False)
    # The kernel intentionally uses one fixed sequential FP32 order, whereas
    # torch.sum uses a tree reduction; a 4096-way fan-in exposes their expected
    # rounding difference.
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-3)
    assert torch.equal(actual[0], destination[0])
    assert torch.equal(actual[2], destination[2])

    empty_source = torch.empty((0, 128), dtype=torch.float32, device=device)
    empty_mapping = api["reduce_map"](((), ()), 0, device)
    preserved = api["reduce"](
        empty_source,
        empty_mapping,
        destination[:2].clone(),
        accumulate=True,
    )
    assert torch.equal(preserved, destination[:2])


@pytest.mark.skipif(not _supported_arch_available(), reason="requires SM90 or SM10x")
def test_post_compile_kernel_watchdogs() -> None:
    api = _packing_api()
    device = torch.device("cuda")
    source = torch.randn((64, 512), dtype=torch.float32, device=device)
    copy_map = api["copy_map"]((9, 0, 9, 17, 3, 31, 1), 64, device)
    remap_lut = api["remap_lut"]((5, -1, 1, 3, 0, -1, 2, 4), 6, device)
    logical = torch.tensor([2, -1, 0, 7, 4, 99], dtype=torch.int32, device=device)
    reduce_rows = ((), (3,), (1, 1, 4), (2, 5, 2, 2))
    reduce_map = api["reduce_map"](reduce_rows, 64, device)
    destination = torch.zeros((4, 512), dtype=torch.float32, device=device)

    # Warm every specialization before watchdog timing.
    api["copy"](source, copy_map)
    api["remap"](logical, remap_lut)
    api["reduce"](source, reduce_map, destination.clone(), accumulate=False)
    torch.cuda.synchronize()

    operations = (
        lambda: api["copy"](source, copy_map),
        lambda: api["remap"](logical, remap_lut),
        lambda: api["reduce"](
            source, reduce_map, destination.clone(), accumulate=False
        ),
    )
    for operation in operations:
        with _alarm(10, "one compiled DSA packing kernel exceeded 10 seconds"):
            operation()
            torch.cuda.synchronize()

    with _alarm(30, "post-compile DSA packing batch exceeded 30 seconds"):
        for _ in range(20):
            for operation in operations:
                operation()
        torch.cuda.synchronize()
