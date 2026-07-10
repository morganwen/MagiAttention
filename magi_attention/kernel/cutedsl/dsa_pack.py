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

"""Device-resident Magi_DSA row packing, id remapping and CSR reduction.

The public frontend and mapping schema deliberately do not encode a GPU
architecture.  Step 4 is validated on SM90, while the kernels use only
global-memory copies and register arithmetic shared by SM90 and SM100.  The
architecture is part of every compile-cache key so one process or persistent
cache can safely serve both families.

All maps are built from static host plans and validated before upload.  Dynamic
Indexer results only consume the device remap LUT; no data-dependent D2H is
performed here.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, const_expr

from .cache_utils import get_jit_cache

_NUM_THREADS = 256
_VECTOR_BITS = 128
_INT32_MAX = torch.iinfo(torch.int32).max
_SUPPORTED_ARCH_MAJORS = (9, 10)
_SUPPORTED_COPY_DTYPES = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
}

_copy_compile_cache = get_jit_cache("dsa_pack_copy")
_remap_compile_cache = get_jit_cache("dsa_pack_remap")
_csr_compile_cache = get_jit_cache("dsa_pack_csr")


def _validate_device_int32_vector(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        tensor.dtype != torch.int32
        or tensor.ndim != 1
        or not tensor.is_cuda
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous CUDA int32 vector")


def _host_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        raise TypeError(
            f"{name} must be static host metadata, not a torch.Tensor; "
            "dynamic device metadata must never be copied to the host"
        )
    result: list[int] = []
    for value in values:
        try:
            converted = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{name} entries must be integers") from exc
        if not -1 <= converted <= _INT32_MAX:
            raise ValueError(f"{name} entry {converted} does not fit int32")
        result.append(converted)
    return tuple(result)


def _validate_row_count(name: str, row_count: int) -> int:
    try:
        row_count = operator.index(row_count)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not 0 <= row_count <= _INT32_MAX:
        raise ValueError(f"{name} must be in [0, {_INT32_MAX}], got {row_count}")
    return row_count


def _cuda_device(device: torch.device | str | int) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"DSA device maps require a CUDA device, got {resolved}")
    return resolved


@dataclass(frozen=True)
class DsaDeviceCopyMap:
    """Validated destination-row to source-row map resident on one GPU."""

    destination_to_source: torch.Tensor
    source_row_count: int

    def __post_init__(self) -> None:
        _validate_device_int32_vector(
            "destination_to_source", self.destination_to_source
        )
        _validate_row_count("source_row_count", self.source_row_count)

    @property
    def destination_row_count(self) -> int:
        return self.destination_to_source.numel()


@dataclass(frozen=True)
class DsaDeviceRemapLut:
    """Logical id to packed-local row lookup table resident on one GPU."""

    logical_to_local: torch.Tensor
    local_row_count: int

    def __post_init__(self) -> None:
        _validate_device_int32_vector("logical_to_local", self.logical_to_local)
        _validate_row_count("local_row_count", self.local_row_count)


@dataclass(frozen=True)
class DsaDeviceReduceMap:
    """Validated destination-row CSR map resident on one GPU."""

    row_offsets: torch.Tensor
    source_rows: torch.Tensor
    source_row_count: int

    def __post_init__(self) -> None:
        _validate_device_int32_vector("row_offsets", self.row_offsets)
        _validate_device_int32_vector("source_rows", self.source_rows)
        _validate_row_count("source_row_count", self.source_row_count)
        if self.row_offsets.device != self.source_rows.device:
            raise ValueError("CSR metadata tensors must share one CUDA device")
        if self.row_offsets.numel() < 1:
            raise ValueError("row_offsets must contain at least the initial zero")

    @property
    def destination_row_count(self) -> int:
        return self.row_offsets.numel() - 1


def make_dsa_device_copy_map(
    destination_to_source: Sequence[int],
    source_row_count: int,
    device: torch.device | str | int,
) -> DsaDeviceCopyMap:
    """Validate one static row-copy map on the host, then upload it once."""

    source_row_count = _validate_row_count("source_row_count", source_row_count)
    mapping = _host_int_tuple("destination_to_source", destination_to_source)
    for source_row in mapping:
        if not 0 <= source_row < source_row_count:
            raise ValueError(
                f"copy source row {source_row} is outside [0, {source_row_count})"
            )
    device = _cuda_device(device)
    return DsaDeviceCopyMap(
        torch.tensor(mapping, dtype=torch.int32, device=device),
        source_row_count,
    )


def make_dsa_device_remap_lut(
    logical_to_local: Sequence[int],
    local_row_count: int,
    device: torch.device | str | int,
) -> DsaDeviceRemapLut:
    """Validate a static logical-id LUT; ``-1`` denotes an unavailable row."""

    local_row_count = _validate_row_count("local_row_count", local_row_count)
    mapping = _host_int_tuple("logical_to_local", logical_to_local)
    for local_row in mapping:
        if local_row != -1 and not 0 <= local_row < local_row_count:
            raise ValueError(
                f"remap row {local_row} is neither -1 nor in " f"[0, {local_row_count})"
            )
    device = _cuda_device(device)
    return DsaDeviceRemapLut(
        torch.tensor(mapping, dtype=torch.int32, device=device),
        local_row_count,
    )


def make_dsa_device_reduce_map(
    source_rows_per_destination: Sequence[Sequence[int]],
    source_row_count: int,
    device: torch.device | str | int,
) -> DsaDeviceReduceMap:
    """Validate a static destination-row CSR reduction map and upload it."""

    if isinstance(source_rows_per_destination, torch.Tensor):
        raise TypeError("CSR rows must be static host metadata, not a torch.Tensor")
    source_row_count = _validate_row_count("source_row_count", source_row_count)
    row_offsets = [0]
    source_rows: list[int] = []
    for destination_row, values in enumerate(source_rows_per_destination):
        rows = _host_int_tuple(f"source rows for destination {destination_row}", values)
        for source_row in rows:
            if not 0 <= source_row < source_row_count:
                raise ValueError(
                    f"CSR source row {source_row} is outside "
                    f"[0, {source_row_count})"
                )
        source_rows.extend(rows)
        if len(source_rows) > _INT32_MAX:
            raise ValueError("CSR contribution count does not fit int32")
        row_offsets.append(len(source_rows))

    device = _cuda_device(device)
    return DsaDeviceReduceMap(
        row_offsets=torch.tensor(row_offsets, dtype=torch.int32, device=device),
        source_rows=torch.tensor(source_rows, dtype=torch.int32, device=device),
        source_row_count=source_row_count,
    )


def _make_fake_matrix(dtype, rows, feature_dim: int) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        dtype,
        (rows, feature_dim),
        stride=(feature_dim, 1),
        assumed_align=16,
    )


def _make_fake_vector(dtype, length, alignment: int) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        dtype,
        (length,),
        stride=(1,),
        assumed_align=alignment,
    )


class _DsaRowCopy:
    def __init__(self, dtype, feature_dim: int) -> None:
        self.dtype = dtype
        self.feature_dim = feature_dim
        vector_elems = _VECTOR_BITS // dtype.width
        self.vector_elems = vector_elems if feature_dim % vector_elems == 0 else 1
        self.vectors_per_row = feature_dim // self.vector_elems
        self.copy_bits = self.vector_elems * dtype.width

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        destination_to_source: cute.Tensor,
        destination: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(source.element_type != self.dtype):
            raise TypeError("row-copy source has an unexpected dtype")
        if const_expr(destination.element_type != self.dtype):
            raise TypeError("row-copy destination has an unexpected dtype")
        if const_expr(destination_to_source.element_type != Int32):
            raise TypeError("destination_to_source must use int32")
        if const_expr(source.shape[1] != self.feature_dim):
            raise ValueError("row-copy source width does not match specialization")
        if const_expr(destination.shape[1] != self.feature_dim):
            raise ValueError("row-copy destination width does not match specialization")

        total_vectors = destination.shape[0] * self.vectors_per_row
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=self.copy_bits,
        )
        self.kernel(
            source,
            destination_to_source,
            destination,
            copy_atom,
        ).launch(
            grid=[cute.ceil_div(total_vectors, _NUM_THREADS), 1, 1],
            block=[_NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        destination_to_source: cute.Tensor,
        destination: cute.Tensor,
        copy_atom: cute.CopyAtom,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        linear_vector = block_idx * _NUM_THREADS + thread_idx
        total_vectors = destination.shape[0] * self.vectors_per_row
        if linear_vector < total_vectors:
            destination_row = linear_vector // self.vectors_per_row
            vector_idx = linear_vector % self.vectors_per_row
            source_row = Int32(destination_to_source[destination_row])
            source_element = (
                source_row * self.feature_dim + vector_idx * self.vector_elems
            )
            destination_element = (
                destination_row * self.feature_dim + vector_idx * self.vector_elems
            )
            source_element = cute.assume(source_element, divby=self.vector_elems)
            destination_element = cute.assume(
                destination_element, divby=self.vector_elems
            )
            source_vector = cute.make_tensor(
                source.iterator + source_element,
                (self.vector_elems,),
            )
            destination_vector = cute.make_tensor(
                destination.iterator + destination_element,
                (self.vector_elems,),
            )
            registers = cute.make_rmem_tensor((self.vector_elems,), self.dtype)
            cute.copy(copy_atom, source_vector, registers)
            cute.copy(copy_atom, registers, destination_vector)


class _DsaIndexRemap:
    @cute.jit
    def __call__(
        self,
        logical_indices: cute.Tensor,
        logical_to_local: cute.Tensor,
        destination: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(logical_indices.element_type != Int32):
            raise TypeError("logical_indices must use int32")
        if const_expr(logical_to_local.element_type != Int32):
            raise TypeError("logical_to_local must use int32")
        if const_expr(destination.element_type != Int32):
            raise TypeError("remap destination must use int32")
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=32
        )
        self.kernel(
            logical_indices,
            logical_to_local,
            destination,
            copy_atom,
        ).launch(
            grid=[cute.ceil_div(destination.shape[0], _NUM_THREADS), 1, 1],
            block=[_NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logical_indices: cute.Tensor,
        logical_to_local: cute.Tensor,
        destination: cute.Tensor,
        copy_atom: cute.CopyAtom,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        linear_idx = block_idx * _NUM_THREADS + thread_idx
        if linear_idx < destination.shape[0]:
            logical_id = Int32(logical_indices[linear_idx])
            mapped = Int32(-1)
            if logical_id >= 0:
                if logical_id < logical_to_local.shape[0]:
                    mapped = Int32(logical_to_local[logical_id])
            registers = cute.make_rmem_tensor((1,), Int32)
            registers[0] = mapped
            destination_element = cute.make_tensor(
                destination.iterator + linear_idx,
                (1,),
            )
            cute.copy(copy_atom, registers, destination_element)


class _DsaFp32CsrReduce:
    def __init__(self, feature_dim: int, accumulate: bool) -> None:
        self.feature_dim = feature_dim
        self.accumulate = accumulate
        vector_elems = _VECTOR_BITS // Float32.width
        self.vector_elems = vector_elems if feature_dim % vector_elems == 0 else 1
        self.vectors_per_row = feature_dim // self.vector_elems
        self.copy_bits = self.vector_elems * Float32.width

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        row_offsets: cute.Tensor,
        source_rows: cute.Tensor,
        destination: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(source.element_type != Float32):
            raise TypeError("CSR source must use float32")
        if const_expr(destination.element_type != Float32):
            raise TypeError("CSR destination must use float32")
        if const_expr(row_offsets.element_type != Int32):
            raise TypeError("row_offsets must use int32")
        if const_expr(source_rows.element_type != Int32):
            raise TypeError("source_rows must use int32")
        if const_expr(source.shape[1] != self.feature_dim):
            raise ValueError("CSR source width does not match specialization")
        if const_expr(destination.shape[1] != self.feature_dim):
            raise ValueError("CSR destination width does not match specialization")

        total_vectors = destination.shape[0] * self.vectors_per_row
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Float32,
            num_bits_per_copy=self.copy_bits,
        )
        self.kernel(
            source,
            row_offsets,
            source_rows,
            destination,
            copy_atom,
        ).launch(
            grid=[cute.ceil_div(total_vectors, _NUM_THREADS), 1, 1],
            block=[_NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        row_offsets: cute.Tensor,
        source_rows: cute.Tensor,
        destination: cute.Tensor,
        copy_atom: cute.CopyAtom,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        linear_vector = block_idx * _NUM_THREADS + thread_idx
        total_vectors = destination.shape[0] * self.vectors_per_row
        if linear_vector < total_vectors:
            destination_row = linear_vector // self.vectors_per_row
            vector_idx = linear_vector % self.vectors_per_row
            begin = Int32(row_offsets[destination_row])
            end = Int32(row_offsets[destination_row + 1])
            if begin < end:
                destination_element = (
                    destination_row * self.feature_dim + vector_idx * self.vector_elems
                )
                destination_element = cute.assume(
                    destination_element, divby=self.vector_elems
                )
                destination_vector = cute.make_tensor(
                    destination.iterator + destination_element,
                    (self.vector_elems,),
                )
                accumulator = cute.make_rmem_tensor((self.vector_elems,), Float32)
                if const_expr(self.accumulate):
                    cute.copy(copy_atom, destination_vector, accumulator)
                else:
                    accumulator.fill(0.0)

                source_registers = cute.make_rmem_tensor((self.vector_elems,), Float32)
                for item in cutlass.range(begin, end, unroll=1):
                    source_row = Int32(source_rows[item])
                    source_element = (
                        source_row * self.feature_dim + vector_idx * self.vector_elems
                    )
                    source_element = cute.assume(
                        source_element, divby=self.vector_elems
                    )
                    source_vector = cute.make_tensor(
                        source.iterator + source_element,
                        (self.vector_elems,),
                    )
                    cute.copy(copy_atom, source_vector, source_registers)
                    accumulator.store(accumulator.load() + source_registers.load())
                cute.copy(copy_atom, accumulator, destination_vector)


def _tensor_arch(tensor: torch.Tensor) -> tuple[int, int]:
    arch = torch.cuda.get_device_capability(tensor.device)
    if arch[0] not in _SUPPORTED_ARCH_MAJORS:
        supported = ", ".join(f"SM{major}x" for major in _SUPPORTED_ARCH_MAJORS)
        raise RuntimeError(
            f"DSA packing supports {supported}; device {tensor.device} is SM{arch[0]}{arch[1]}"
        )
    return arch


def _flatten_cuda_matrix(
    name: str,
    tensor: torch.Tensor,
    *,
    allowed_dtypes: Sequence[torch.dtype],
) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have a row dimension and trailing features")
    if tensor.dtype not in allowed_dtypes:
        expected = ", ".join(str(dtype) for dtype in allowed_dtypes)
        raise TypeError(f"{name} dtype must be one of {expected}, got {tensor.dtype}")
    trailing_shape = tuple(tensor.shape[1:])
    feature_dim = math.prod(trailing_shape)
    if feature_dim <= 0:
        raise ValueError(f"{name} trailing feature width must be positive")
    return tensor.view(tensor.size(0), feature_dim), feature_dim, trailing_shape


def _compile_copy_kernel(
    dtype: torch.dtype,
    feature_dim: int,
    arch: tuple[int, int],
):
    key = ("copy", arch, dtype, feature_dim)
    if key not in _copy_compile_cache:
        cute_dtype = _SUPPORTED_COPY_DTYPES[dtype]
        source_rows = cute.sym_int()
        destination_rows = cute.sym_int()
        source = _make_fake_matrix(cute_dtype, source_rows, feature_dim)
        mapping = _make_fake_vector(Int32, destination_rows, 4)
        destination = _make_fake_matrix(cute_dtype, destination_rows, feature_dim)
        _copy_compile_cache[key] = cute.compile(
            _DsaRowCopy(cute_dtype, feature_dim),
            source,
            mapping,
            destination,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _copy_compile_cache[key]


def _compile_remap_kernel(arch: tuple[int, int]):
    key = ("remap", arch)
    if key not in _remap_compile_cache:
        index_count = cute.sym_int()
        lut_size = cute.sym_int()
        indices = _make_fake_vector(Int32, index_count, 4)
        lut = _make_fake_vector(Int32, lut_size, 4)
        destination = _make_fake_vector(Int32, index_count, 4)
        _remap_compile_cache[key] = cute.compile(
            _DsaIndexRemap(),
            indices,
            lut,
            destination,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _remap_compile_cache[key]


def _compile_csr_kernel(
    feature_dim: int,
    arch: tuple[int, int],
    accumulate: bool,
):
    key = ("csr", arch, torch.float32, feature_dim, accumulate)
    if key not in _csr_compile_cache:
        source_count = cute.sym_int()
        destination_count = cute.sym_int()
        row_offset_count = cute.sym_int()
        item_count = cute.sym_int()
        source = _make_fake_matrix(Float32, source_count, feature_dim)
        row_offsets = _make_fake_vector(Int32, row_offset_count, 4)
        source_rows = _make_fake_vector(Int32, item_count, 4)
        destination = _make_fake_matrix(Float32, destination_count, feature_dim)
        _csr_compile_cache[key] = cute.compile(
            _DsaFp32CsrReduce(feature_dim, accumulate),
            source,
            row_offsets,
            source_rows,
            destination,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _csr_compile_cache[key]


@torch.no_grad()
def copy_dsa_rows(
    source: torch.Tensor,
    mapping: DsaDeviceCopyMap,
) -> torch.Tensor:
    """Gather rows according to a validated device-resident int32 map."""

    source_2d, feature_dim, trailing_shape = _flatten_cuda_matrix(
        "source", source, allowed_dtypes=tuple(_SUPPORTED_COPY_DTYPES)
    )
    if not isinstance(mapping, DsaDeviceCopyMap):
        raise TypeError("mapping must be a DsaDeviceCopyMap")
    if source.size(0) != mapping.source_row_count:
        raise ValueError(
            f"copy map expects {mapping.source_row_count} source rows, "
            f"got {source.size(0)}"
        )
    if mapping.destination_to_source.device != source.device:
        raise ValueError("row-copy source and mapping must share one CUDA device")
    output = torch.empty(
        (mapping.destination_row_count, *trailing_shape),
        dtype=source.dtype,
        device=source.device,
    )
    if mapping.destination_row_count == 0:
        return output
    arch = _tensor_arch(source)
    with torch.cuda.device(source.device):
        _compile_copy_kernel(source.dtype, feature_dim, arch)(
            source_2d,
            mapping.destination_to_source,
            output.view(output.size(0), feature_dim),
        )
    return output


@torch.no_grad()
def remap_dsa_indices(
    logical_indices: torch.Tensor,
    mapping: DsaDeviceRemapLut,
) -> torch.Tensor:
    """Map device-resident logical ids to packed-local rows, preserving ``-1``."""

    if not isinstance(logical_indices, torch.Tensor):
        raise TypeError("logical_indices must be a torch.Tensor")
    if (
        logical_indices.dtype != torch.int32
        or not logical_indices.is_cuda
        or not logical_indices.is_contiguous()
    ):
        raise ValueError("logical_indices must be a contiguous CUDA int32 tensor")
    if not isinstance(mapping, DsaDeviceRemapLut):
        raise TypeError("mapping must be a DsaDeviceRemapLut")
    if mapping.logical_to_local.device != logical_indices.device:
        raise ValueError("logical indices and remap LUT must share one CUDA device")
    output = torch.empty_like(logical_indices)
    if logical_indices.numel() == 0:
        return output
    arch = _tensor_arch(logical_indices)
    with torch.cuda.device(logical_indices.device):
        _compile_remap_kernel(arch)(
            logical_indices.view(-1),
            mapping.logical_to_local,
            output.view(-1),
        )
    return output


@torch.no_grad()
def reduce_dsa_rows_csr(
    source: torch.Tensor,
    mapping: DsaDeviceReduceMap,
    destination: torch.Tensor | None = None,
    *,
    accumulate: bool = False,
) -> torch.Tensor:
    """Reduce FP32 rows in fixed CSR order into an FP32 accumulator.

    Non-empty destination rows are overwritten when ``accumulate=False`` and
    incremented otherwise.  Empty CSR rows always preserve the destination,
    which lets GroupReduce restore only communicated owner rows without
    disturbing local-only rows.
    """

    source_2d, feature_dim, trailing_shape = _flatten_cuda_matrix(
        "source", source, allowed_dtypes=(torch.float32,)
    )
    if not isinstance(mapping, DsaDeviceReduceMap):
        raise TypeError("mapping must be a DsaDeviceReduceMap")
    if source.size(0) != mapping.source_row_count:
        raise ValueError(
            f"CSR map expects {mapping.source_row_count} source rows, "
            f"got {source.size(0)}"
        )
    if mapping.row_offsets.device != source.device:
        raise ValueError("CSR source and mapping must share one CUDA device")
    expected_shape = (mapping.destination_row_count, *trailing_shape)
    if destination is None:
        destination = torch.zeros(
            expected_shape,
            dtype=torch.float32,
            device=source.device,
        )
    else:
        if not isinstance(destination, torch.Tensor):
            raise TypeError("destination must be a torch.Tensor")
        if tuple(destination.shape) != expected_shape:
            raise ValueError(
                f"CSR destination must have shape {expected_shape}, "
                f"got {tuple(destination.shape)}"
            )
        if (
            destination.dtype != torch.float32
            or destination.device != source.device
            or not destination.is_contiguous()
        ):
            raise ValueError(
                "CSR destination must be contiguous FP32 on the source device"
            )
    if mapping.destination_row_count == 0 or mapping.source_rows.numel() == 0:
        return destination
    arch = _tensor_arch(source)
    with torch.cuda.device(source.device):
        _compile_csr_kernel(feature_dim, arch, accumulate)(
            source_2d,
            mapping.row_offsets,
            mapping.source_rows,
            destination.view(destination.size(0), feature_dim),
        )
    return destination


__all__ = [
    "DsaDeviceCopyMap",
    "DsaDeviceReduceMap",
    "DsaDeviceRemapLut",
    "copy_dsa_rows",
    "make_dsa_device_copy_map",
    "make_dsa_device_reduce_map",
    "make_dsa_device_remap_lut",
    "reduce_dsa_rows_csr",
    "remap_dsa_indices",
]
