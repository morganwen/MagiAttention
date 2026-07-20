"""Vectorized SM100 row packing and mixed-precision CSR reduction for Magi-DSA."""

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

import math
import os
from functools import lru_cache
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, const_expr

_NUM_THREADS = 256
_VECTOR_BITS = 128
_SUPPORTED_DTYPES = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
}
_AOT_MODULES: dict[tuple[str, int, int, torch.dtype, int], object] = {}


def _dtype_tag(dtype: torch.dtype) -> str:
    tags = {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.int32: "i32",
    }
    return tags[dtype]


def _aot_identity(
    operation: str,
    major: int,
    minor: int,
    dtype: torch.dtype,
    width: int,
) -> tuple[str, str]:
    base = f"{operation}_sm{major}{minor}_{_dtype_tag(dtype)}_w{width}"
    return f"{base}.o", f"magi_dsa_{base}"


def _load_aot_kernel(
    operation: str,
    major: int,
    minor: int,
    dtype: torch.dtype,
    width: int,
):
    root = os.environ.get("MAGI_DSA_CUTE_AOT_DIR")
    if not root:
        return None
    filename, symbol = _aot_identity(operation, major, minor, dtype, width)
    path = Path(root) / filename
    if not path.is_file():
        if os.environ.get("MAGI_DSA_CUTE_AOT_REQUIRED") == "1":
            raise RuntimeError(f"required Magi-DSA CuTe AOT object is missing: {path}")
        return None
    from cutlass.base_dsl.export.external_binary_module import ExternalBinaryModule

    module = ExternalBinaryModule(str(path), enable_tvm_ffi=True)
    function = getattr(module, symbol)
    _AOT_MODULES[(operation, major, minor, dtype, width)] = module
    return function


def _fake_matrix(dtype, rows, width: int) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        dtype,
        (rows, width),
        stride=(width, 1),
        assumed_align=16,
    )


def _fake_vector(dtype, length, alignment: int) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        dtype, (length,), stride=(1,), assumed_align=alignment
    )


class _DsaRowCopy:
    def __init__(self, dtype, width: int) -> None:
        self.dtype = dtype
        self.width = width
        self.vector_elements = _VECTOR_BITS // dtype.width
        self.vectors_per_row = width // self.vector_elements

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        source_row_for_destination: cute.Tensor,
        destination: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(
            source.element_type != self.dtype or destination.element_type != self.dtype
        ):
            raise TypeError("DSA copy tensor dtype does not match its specialization")
        if const_expr(source_row_for_destination.element_type != Int32):
            raise TypeError("DSA copy map must use int32")
        if const_expr(
            source.shape[1] != self.width or destination.shape[1] != self.width
        ):
            raise ValueError("DSA copy row width does not match its specialization")
        total_vectors = destination.shape[0] * self.vectors_per_row
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=_VECTOR_BITS,
        )
        self.kernel(source, source_row_for_destination, destination, copy_atom).launch(
            grid=[cute.ceil_div(total_vectors, _NUM_THREADS), 1, 1],
            block=[_NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        source_row_for_destination: cute.Tensor,
        destination: cute.Tensor,
        copy_atom: cute.CopyAtom,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        linear_vector = block_idx * _NUM_THREADS + thread_idx
        total_vectors = destination.shape[0] * self.vectors_per_row
        if linear_vector < total_vectors:
            destination_row = linear_vector // self.vectors_per_row
            vector_index = linear_vector % self.vectors_per_row
            source_row = Int32(source_row_for_destination[destination_row])
            source_element = (
                source_row * self.width + vector_index * self.vector_elements
            )
            destination_element = (
                destination_row * self.width + vector_index * self.vector_elements
            )
            source_element = cute.assume(source_element, divby=self.vector_elements)
            destination_element = cute.assume(
                destination_element, divby=self.vector_elements
            )
            source_vector = cute.make_tensor(
                source.iterator + source_element, (self.vector_elements,)
            )
            destination_vector = cute.make_tensor(
                destination.iterator + destination_element,
                (self.vector_elements,),
            )
            registers = cute.make_rmem_tensor((self.vector_elements,), self.dtype)
            cute.copy(copy_atom, source_vector, registers)
            cute.copy(copy_atom, registers, destination_vector)


class _DsaRowCsrReduce:
    def __init__(self, dtype, width: int) -> None:
        self.dtype = dtype
        self.width = width
        self.vector_elements = _VECTOR_BITS // dtype.width
        self.vectors_per_row = width // self.vector_elements

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        row_offsets: cute.Tensor,
        source_rows: cute.Tensor,
        destination: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if const_expr(
            source.element_type != self.dtype or destination.element_type != self.dtype
        ):
            raise TypeError("DSA CSR tensor dtype does not match its specialization")
        if const_expr(
            row_offsets.element_type != Int32 or source_rows.element_type != Int32
        ):
            raise TypeError("DSA CSR metadata must use int32")
        if const_expr(
            source.shape[1] != self.width or destination.shape[1] != self.width
        ):
            raise ValueError("DSA CSR row width does not match its specialization")
        total_vectors = destination.shape[0] * self.vectors_per_row
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=_VECTOR_BITS,
        )
        self.kernel(source, row_offsets, source_rows, destination, copy_atom).launch(
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
            vector_index = linear_vector % self.vectors_per_row
            begin = Int32(row_offsets[destination_row])
            end = Int32(row_offsets[destination_row + 1])
            accumulator = cute.make_rmem_tensor((self.vector_elements,), Float32)
            accumulator.fill(0.0)
            source_registers = cute.make_rmem_tensor(
                (self.vector_elements,), self.dtype
            )
            for item in cutlass.range(begin, end, unroll=1):
                source_row = Int32(source_rows[item])
                source_element = (
                    source_row * self.width + vector_index * self.vector_elements
                )
                source_element = cute.assume(source_element, divby=self.vector_elements)
                source_vector = cute.make_tensor(
                    source.iterator + source_element, (self.vector_elements,)
                )
                cute.copy(copy_atom, source_vector, source_registers)
                accumulator.store(
                    accumulator.load() + source_registers.load().to(Float32)
                )
            destination_element = (
                destination_row * self.width + vector_index * self.vector_elements
            )
            destination_element = cute.assume(
                destination_element, divby=self.vector_elements
            )
            destination_vector = cute.make_tensor(
                destination.iterator + destination_element,
                (self.vector_elements,),
            )
            destination_registers = cute.make_rmem_tensor(
                (self.vector_elements,), self.dtype
            )
            destination_registers.store(accumulator.load().to(self.dtype))
            cute.copy(copy_atom, destination_registers, destination_vector)


@lru_cache(maxsize=32)
def _compile_copy_kernel(
    major: int,
    minor: int,
    dtype: torch.dtype,
    width: int,
):
    if major != 10 or minor != 3:
        raise RuntimeError("Magi-DSA release packing requires B300 SM103")
    aot = _load_aot_kernel("copy", major, minor, dtype, width)
    if aot is not None:
        return aot
    cute_dtype = _SUPPORTED_DTYPES[dtype]
    vector_elements = _VECTOR_BITS // cute_dtype.width
    if width % vector_elements:
        raise ValueError("DSA row width must be divisible by the 128-bit vector width")
    source_rows = cute.sym_int()
    destination_rows = cute.sym_int()
    source = _fake_matrix(cute_dtype, source_rows, width)
    row_map = _fake_vector(Int32, destination_rows, 4)
    destination = _fake_matrix(cute_dtype, destination_rows, width)
    return cute.compile(
        _DsaRowCopy(cute_dtype, width),
        source,
        row_map,
        destination,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=32)
def _compile_reduce_kernel(
    major: int,
    minor: int,
    dtype: torch.dtype,
    width: int,
):
    if major != 10 or minor != 3:
        raise RuntimeError("Magi-DSA release packing requires B300 SM103")
    aot = _load_aot_kernel("reduce", major, minor, dtype, width)
    if aot is not None:
        return aot
    cute_dtype = _SUPPORTED_DTYPES[dtype]
    vector_elements = _VECTOR_BITS // cute_dtype.width
    if width % vector_elements:
        raise ValueError("DSA row width must be divisible by the 128-bit vector width")
    source_count = cute.sym_int()
    destination_count = cute.sym_int()
    offset_count = cute.sym_int()
    item_count = cute.sym_int()
    source = _fake_matrix(cute_dtype, source_count, width)
    row_offsets = _fake_vector(Int32, offset_count, 4)
    source_rows = _fake_vector(Int32, item_count, 4)
    destination = _fake_matrix(cute_dtype, destination_count, width)
    return cute.compile(
        _DsaRowCsrReduce(cute_dtype, width),
        source,
        row_offsets,
        source_rows,
        destination,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _validate_source(source: torch.Tensor) -> tuple[torch.Tensor, int, tuple[int, int]]:
    if not source.is_cuda or not source.is_contiguous():
        raise ValueError("DSA packing source must be a contiguous CUDA tensor")
    if source.ndim < 2:
        raise ValueError("DSA packing source must have at least two dimensions")
    capability = torch.cuda.get_device_capability(source.device)
    if capability != (10, 3):
        raise RuntimeError("Magi-DSA release packing requires B300 SM103")
    width = math.prod(source.shape[1:])
    return source.view(source.shape[0], width), width, capability


def _validate_map(metadata: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if (
        metadata.dtype != torch.int32
        or metadata.ndim != 1
        or not metadata.is_cuda
        or not metadata.is_contiguous()
        or metadata.device != source.device
    ):
        raise ValueError(f"{name} must be contiguous CUDA int32 metadata")


def copy_dsa_rows(
    source: torch.Tensor, source_row_for_destination: torch.Tensor
) -> torch.Tensor:
    """Gather rows using a device-resident int32 map on the caller's stream."""

    source_2d, width, capability = _validate_source(source)
    if source.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("DSA copy supports BF16, FP32, and int32")
    _validate_map(source_row_for_destination, source, "source_row_for_destination")
    output = torch.empty(
        (source_row_for_destination.numel(), *source.shape[1:]),
        dtype=source.dtype,
        device=source.device,
    )
    if not output.shape[0]:
        return output
    _compile_copy_kernel(*capability, source.dtype, width)(
        source_2d,
        source_row_for_destination,
        output.view(output.shape[0], width),
    )
    return output


def reduce_dsa_rows(
    source: torch.Tensor,
    row_offsets: torch.Tensor,
    source_rows: torch.Tensor,
) -> torch.Tensor:
    """Reduce repeated rows using FP32 accumulation and a device CSR map."""

    source_2d, width, capability = _validate_source(source)
    if source.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("DSA CSR reduction supports BF16 and FP32")
    _validate_map(row_offsets, source, "row_offsets")
    _validate_map(source_rows, source, "source_rows")
    if row_offsets.numel() < 1:
        raise ValueError("row_offsets must contain at least one element")
    output_rows = row_offsets.numel() - 1
    output = torch.empty(
        (output_rows, *source.shape[1:]), dtype=source.dtype, device=source.device
    )
    if not output_rows:
        return output
    _compile_reduce_kernel(*capability, source.dtype, width)(
        source_2d,
        row_offsets,
        source_rows,
        output.view(output_rows, width),
    )
    return output


def clear_dsa_pack_cache() -> None:
    _compile_copy_kernel.cache_clear()
    _compile_reduce_kernel.cache_clear()
    _AOT_MODULES.clear()


def export_dsa_pack_aot(
    operation: str,
    major: int,
    minor: int,
    dtype: torch.dtype,
    width: int,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Export one bounded CuTe specialization as a TVM-FFI AOT object."""

    if operation == "copy":
        compiled = _compile_copy_kernel(major, minor, dtype, width)
    elif operation == "reduce":
        compiled = _compile_reduce_kernel(major, minor, dtype, width)
    else:
        raise ValueError("DSA AOT operation must be copy or reduce")
    if not hasattr(compiled, "export_to_c"):
        raise RuntimeError("cannot export a specialization that was loaded from AOT")
    filename, symbol = _aot_identity(operation, major, minor, dtype, width)
    path = Path(output_dir) / filename
    compiled.export_to_c(str(path), symbol)
    return path


__all__ = [
    "clear_dsa_pack_cache",
    "copy_dsa_rows",
    "export_dsa_pack_aot",
    "reduce_dsa_rows",
]
