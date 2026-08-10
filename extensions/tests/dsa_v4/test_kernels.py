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

import pytest
import torch
from magi_attn_extensions.DSA import comm as dsa_comm_module
from magi_attn_extensions.DSA import dist as dist_dsa_module
from magi_attn_extensions.DSA import modeling as dsa_layer_module
from magi_attn_extensions.DSA.backend import DsaIndexerSelection
from magi_attn_extensions.DSA.comm import (
    finish_dsa_reverse_route,
    finish_dsa_tensor_route,
    route_dsa_tensor,
    start_dsa_reverse_route,
    start_dsa_tensor_route,
)
from magi_attn_extensions.DSA.config import DsaRatio, MagiDSAConfig
from magi_attn_extensions.DSA.kernels.triton import rope as dsa_rope_module
from magi_attn_extensions.DSA.kernels.triton.compressor import (
    fused_csa_compressor_reduce,
)
from magi_attn_extensions.DSA.kernels.triton.diagnostics import (
    dsa_nonfinite_block_stats,
    dsa_nonfinite_row_counts,
)
from magi_attn_extensions.DSA.kernels.triton.gradients import (
    fused_dsa_mask_empty_indexer_gradients,
    fused_dsa_scale_indexer_gradients,
)
from magi_attn_extensions.DSA.kernels.triton.indices import build_csa_index_tensors
from magi_attn_extensions.DSA.kernels.triton.kl import fused_dsa_selected_kl_state
from magi_attn_extensions.DSA.kernels.triton.projection import fused_dsa_scale_cast
from magi_attn_extensions.DSA.kernels.triton.reductions import fused_dsa_row_logsumexp
from magi_attn_extensions.DSA.kernels.triton.rope import (
    fused_dsa_rope,
    fused_dsa_rope_hadamard,
)
from magi_attn_extensions.DSA.modeling import (
    DsaIndexer,
    DsaRMSNorm,
    MagiDSALayer,
    _yarn_inverse_frequencies,
    apply_dsa_rope,
    apply_normalized_hadamard,
)
from magi_attn_extensions.DSA.packing import make_dsa_device_rank_plan
from magi_attn_extensions.DSA.runtime import MagiDSARuntimeMgr
from magi_attn_extensions.DSA.solver import build_dsa_execution_plan
from magi_attn_extensions.DSA.types import MagiDSAInput, MagiDSAPackedMeta


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _small_config(ratio: DsaRatio) -> MagiDSAConfig:
    return MagiDSAConfig(
        ratio=ratio,
        hidden_size=32,
        q_lora_rank=16,
        num_query_heads=4,
        head_dim=16,
        rope_dim=4,
        indexer_heads=4,
        indexer_head_dim=8,
        indexer_topk=4,
        window_size=4,
        indexer_atom_size=4,
    )


def test_fused_csa_index_builder_preserves_independent_ki_kv_maps() -> None:
    topk_ids = torch.tensor(
        [[0, 3, 5, -1], [4, 1, 2, 0], [-1, -1, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_lengths = torch.tensor([3, 2, 0], dtype=torch.int32, device="cuda")
    attention_map = torch.tensor([5, 0, 4, 2, 1, 3], dtype=torch.int32, device="cuda")
    indexer_map = torch.tensor([1, 4, 0, 5, 3, 2], dtype=torch.int32, device="cuda")
    # A raw window is a contiguous run, so it is a base plus a length.
    window_base = torch.tensor([7, 4, 2], dtype=torch.int32, device="cuda")
    window_lengths = torch.tensor([3, 2, 1], dtype=torch.int32, device="cuda")

    attention, lengths, indexer, compressed = build_csa_index_tensors(
        topk_ids,
        topk_lengths,
        attention_map,
        indexer_map,
        window_base,
        window_lengths,
        3,
        raw_bank_rows=10,
    )

    assert torch.equal(
        attention.cpu(),
        torch.tensor(
            [
                [15, 12, 13, -1, 7, 8, 9],
                [11, 10, -1, -1, 4, 5, -1],
                [-1, -1, -1, -1, 2, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(lengths.cpu(), torch.tensor([6, 4, 1], dtype=torch.int32))
    assert torch.equal(
        indexer.cpu(),
        torch.tensor(
            [[1, 5, 2, -1], [3, 4, -1, -1], [-1, -1, -1, -1]],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        compressed.cpu(),
        torch.tensor(
            [[5, 2, 3, -1], [1, 0, -1, -1], [-1, -1, -1, -1]],
            dtype=torch.int32,
        ),
    )


def test_fused_csa_index_builder_handles_an_empty_compressed_domain() -> None:
    topk_ids = torch.full((2, 4), -1, dtype=torch.int32, device="cuda")
    topk_lengths = torch.zeros(2, dtype=torch.int32, device="cuda")
    empty_map = torch.empty(0, dtype=torch.int32, device="cuda")
    window_base = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    window_lengths = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    expected_window = torch.tensor(
        [[3, 4], [5, -1]], dtype=torch.int32, device="cuda"
    )
    attention, lengths, indexer, compressed = build_csa_index_tensors(
        topk_ids,
        topk_lengths,
        empty_map,
        empty_map,
        window_base,
        window_lengths,
        2,
        raw_bank_rows=6,
    )
    assert torch.equal(attention[:, :4], topk_ids)
    assert torch.equal(attention[:, 4:], expected_window)
    assert torch.equal(lengths, window_lengths)
    assert torch.equal(indexer, topk_ids)
    assert torch.equal(compressed, topk_ids)


def test_fused_csa_index_builder_supports_pro_1024_plus_128_width() -> None:
    rows = 3
    topk_width = 1024
    window_width = 128
    map_rows = 2048
    raw_bank_rows = 4096
    columns = torch.arange(topk_width, dtype=torch.int32, device="cuda")
    topk_ids = torch.stack(
        tuple((columns + row * 137).remainder(map_rows) for row in range(rows))
    )
    topk_lengths = torch.tensor([1024, 1000, 0], dtype=torch.int32, device="cuda")
    topk_ids = topk_ids.masked_fill(
        columns.unsqueeze(0) >= topk_lengths.unsqueeze(1), -1
    )
    attention_map = torch.arange(map_rows, dtype=torch.int32, device="cuda").roll(17)
    indexer_map = torch.arange(map_rows, dtype=torch.int32, device="cuda").flip(0)
    window_columns = torch.arange(window_width, dtype=torch.int32, device="cuda")
    window_lengths = torch.tensor([128, 17, 0], dtype=torch.int32, device="cuda")
    window_base = torch.full((rows,), 23, dtype=torch.int32, device="cuda")
    window_rows = (window_columns.unsqueeze(0) + 23).expand(rows, -1).clone()
    window_rows.masked_fill_(
        window_columns.unsqueeze(0) >= window_lengths.unsqueeze(1), -1
    )

    attention, lengths, indexer, compressed = build_csa_index_tensors(
        topk_ids,
        topk_lengths,
        attention_map,
        indexer_map,
        window_base,
        window_lengths,
        window_width,
        raw_bank_rows=raw_bank_rows,
    )

    valid_topk = topk_ids >= 0
    safe_ids = topk_ids.clamp_min(0).long()
    expected_indexer = indexer_map.index_select(0, safe_ids.flatten()).view_as(topk_ids)
    expected_indexer.masked_fill_(~valid_topk, -1)
    expected_compressed = attention_map.index_select(0, safe_ids.flatten()).view_as(
        topk_ids
    )
    expected_compressed.masked_fill_(~valid_topk, -1)
    expected_attention = torch.cat(
        (
            expected_compressed.masked_fill(~valid_topk, -raw_bank_rows)
            + raw_bank_rows,
            window_rows,
        ),
        dim=1,
    )
    expected_attention[:, :topk_width].masked_fill_(~valid_topk, -1)

    assert attention.shape == (rows, 1152)
    assert torch.equal(attention, expected_attention)
    assert torch.equal(lengths, topk_lengths + window_lengths)
    assert torch.equal(indexer, expected_indexer)
    assert torch.equal(compressed, expected_compressed)


@pytest.mark.parametrize("inverse", [False, True])
def test_fused_v4_rope_matches_eager_forward_backward_and_is_reentrant(
    inverse: bool,
) -> None:
    torch.manual_seed(91)
    config = MagiDSAConfig(ratio=4)
    positions = torch.tensor(
        [0, 1, 63, 65535, 131071],
        device="cuda",
        dtype=torch.int32,
    )
    inverse_frequencies = _yarn_inverse_frequencies(config).cuda()
    source = torch.randn(
        positions.numel(),
        config.indexer_heads,
        config.indexer_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    source_before = source.detach().clone()
    reference_source = source.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(source)

    actual = fused_dsa_rope(
        source,
        positions,
        inverse_frequencies,
        config.rope_dim,
        inverse=inverse,
    )
    reference = apply_dsa_rope(
        reference_source,
        positions,
        inverse_frequencies,
        config.rope_dim,
        inverse=inverse,
    )
    torch.testing.assert_close(actual.float(), reference.float(), atol=2e-2, rtol=2e-2)
    assert actual.is_contiguous()
    assert actual.data_ptr() != source.data_ptr()
    assert torch.equal(source, source_before)

    first_gradient = torch.autograd.grad(
        actual,
        source,
        gradient,
        retain_graph=True,
    )[0]
    second_gradient = torch.autograd.grad(actual, source, gradient)[0]
    reference_gradient = torch.autograd.grad(reference, reference_source, gradient)[0]
    opposite_rotation = apply_dsa_rope(
        gradient,
        positions,
        inverse_frequencies,
        config.rope_dim,
        inverse=not inverse,
    )
    torch.testing.assert_close(
        first_gradient.float(),
        reference_gradient.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        first_gradient.float(),
        opposite_rotation.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.equal(first_gradient, second_gradient)

    accumulated_source = source.detach().clone().requires_grad_(True)
    first_output = fused_dsa_rope(
        accumulated_source,
        positions,
        inverse_frequencies,
        config.rope_dim,
        inverse=inverse,
    )
    second_output = fused_dsa_rope(
        accumulated_source,
        positions,
        inverse_frequencies,
        config.rope_dim,
        inverse=inverse,
    )
    (first_output.float().sum() + second_output.float().sum()).backward()
    assert accumulated_source.grad is not None
    assert torch.isfinite(accumulated_source.grad).all()


def test_cuda_layer_output_rope_uses_inverse_fused_rotation(monkeypatch) -> None:
    config = _small_config(128)
    layer = MagiDSALayer(config).cuda()
    positions = torch.tensor([0, 3, 1, 8], device="cuda", dtype=torch.int32)
    raw_output = torch.randn(
        positions.numel(),
        config.num_query_heads,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    calls: list[bool] = []

    def capture_fused_rotation(
        tensor: torch.Tensor,
        row_positions: torch.Tensor,
        inverse_frequencies: torch.Tensor,
        rope_dim: int,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        calls.append(inverse)
        return apply_dsa_rope(
            tensor,
            row_positions,
            inverse_frequencies,
            rope_dim,
            inverse=inverse,
        )

    monkeypatch.setattr(dsa_rope_module, "fused_dsa_rope", capture_fused_rotation)
    actual = layer.inverse_output_rope(raw_output, positions)
    expected = apply_dsa_rope(
        raw_output,
        positions,
        layer.output_inverse_frequencies,
        config.rope_dim,
        inverse=True,
    )
    assert calls == [True]
    torch.testing.assert_close(actual, expected)


def test_quack_rms_norm_matches_eager_forward_backward() -> None:
    torch.manual_seed(911)
    source = torch.randn(
        37,
        512,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_source = source.detach().clone().requires_grad_(True)
    norm = DsaRMSNorm(512, 1e-6).cuda()
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.1)
    reference_weight = norm.weight.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(source)

    actual = norm(source)
    reference_value = reference_source.float()
    reference = (
        reference_value
        * torch.rsqrt(reference_value.square().mean(dim=-1, keepdim=True) + norm.eps)
        * reference_weight
    ).to(source.dtype)
    torch.testing.assert_close(actual.float(), reference.float(), atol=2e-2, rtol=2e-2)

    actual_dx, actual_dw = torch.autograd.grad(
        actual,
        (source, norm.weight),
        gradient,
    )
    reference_dx, reference_dw = torch.autograd.grad(
        reference,
        (reference_source, reference_weight),
        gradient,
    )
    torch.testing.assert_close(
        actual_dx.float(), reference_dx.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(actual_dw, reference_dw, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("columns", [1, 37, 1024, 32768])
def test_fused_indexer_row_logsumexp_matches_torch(columns: int) -> None:
    torch.manual_seed(913 + columns)
    rows = 4
    lengths = torch.tensor(
        [0, 1, max(columns // 2, 1), columns],
        device="cuda",
        dtype=torch.int32,
    )
    scores = torch.randn(rows, columns, device="cuda", dtype=torch.float32)
    column_ids = torch.arange(columns, device="cuda").unsqueeze(0)
    scores = scores.masked_fill(column_ids >= lengths.unsqueeze(1), float("-inf"))

    actual = fused_dsa_row_logsumexp(scores, lengths)
    reference = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("output_dim", [16, 128, 512])
def test_fused_csa_compressor_reduction_matches_eager_forward_backward(
    output_dim: int,
) -> None:
    torch.manual_seed(920 + output_dim)
    rows = 7
    projected_kv = torch.randn(
        rows,
        8,
        2 * output_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    projected_gate = torch.randn_like(projected_kv, requires_grad=True)
    ape = torch.randn(
        4,
        2 * output_dim,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    valid_rows = torch.rand(rows, 8, device="cuda") > 0.25
    valid_rows[:, 4:] = True
    valid_rows[0, :4] = False
    reference_kv = projected_kv.detach().clone().requires_grad_(True)
    reference_gate = projected_gate.detach().clone().requires_grad_(True)
    reference_ape = ape.detach().clone().requires_grad_(True)

    actual = fused_csa_compressor_reduce(
        projected_kv,
        projected_gate,
        ape,
        valid_rows,
        output_dim,
        torch.bfloat16,
    )
    previous_kv = reference_kv[:, :4, :output_dim]
    current_kv = reference_kv[:, 4:, output_dim:]
    previous_gate = (
        reference_gate[:, :4, :output_dim] + reference_ape[None, :, :output_dim]
    )
    current_gate = (
        reference_gate[:, 4:, output_dim:] + reference_ape[None, :, output_dim:]
    )
    values = torch.cat((previous_kv, current_kv), dim=1)
    logits = torch.cat((previous_gate, current_gate), dim=1).masked_fill(
        ~valid_rows.unsqueeze(-1), float("-inf")
    )
    reference = (values * logits.softmax(dim=1)).sum(dim=1).to(torch.bfloat16)
    torch.testing.assert_close(actual.float(), reference.float(), atol=5e-3, rtol=5e-3)

    gradient = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        actual,
        (projected_kv, projected_gate, ape),
        gradient,
        retain_graph=True,
    )
    reference_gradients = torch.autograd.grad(
        reference,
        (reference_kv, reference_gate, reference_ape),
        gradient,
    )
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients
    ):
        torch.testing.assert_close(
            actual_gradient,
            reference_gradient,
            atol=2e-2,
            rtol=2e-2,
        )
    repeated_gradients = torch.autograd.grad(
        actual,
        (projected_kv, projected_gate, ape),
        gradient,
    )
    for first, second in zip(actual_gradients, repeated_gradients):
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_fused_selected_kl_state_matches_eager_with_empty_rows() -> None:
    torch.manual_seed(927)
    rows, topk_width = 13, 512
    lengths = torch.tensor(
        [0, 1, 2, 512, 511, 3, 512, 1, 256, 512, 2, 512, 4],
        device="cuda",
        dtype=torch.int32,
    )
    columns = torch.arange(topk_width, device="cuda").unsqueeze(0)
    valid = columns < lengths.unsqueeze(1)
    target = torch.rand(rows, topk_width, device="cuda", dtype=torch.float32)
    predict = torch.rand_like(target)
    target = target.masked_fill(~valid, 0.0)
    predict = predict.masked_fill(~valid, 0.0)
    target[1, 0] = 0.0
    predict[2, 0] = 0.0
    loss_coeff = 0.3

    loss = fused_dsa_selected_kl_state(
        target,
        predict,
        lengths,
        loss_coeff,
    )
    minimum = math.exp(-100.0)
    target_2d = target.masked_fill(~valid, 0.0)
    predict_2d = predict.masked_fill(~valid, 0.0)
    log_target = target_2d.clamp_min(minimum).log().clamp(-100.0, 0.0)
    log_predict = predict_2d.clamp_min(minimum).log().clamp(-100.0, 0.0)
    log_predict = log_predict.masked_fill(~valid, 0.0)
    reference_loss = (target_2d * (log_target - log_predict)).sum(
        dim=-1
    ).mean() * loss_coeff

    torch.testing.assert_close(loss, reference_loss, atol=2e-5, rtol=2e-5)


def test_fused_indexer_gradient_postprocessing_matches_eager() -> None:
    torch.manual_seed(928)
    grad_q = torch.randn(13, 4, 8, device="cuda", dtype=torch.bfloat16)
    grad_weights = torch.randn(13, 4, device="cuda", dtype=torch.bfloat16)
    grad_k = torch.randn(17, 8, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor(
        [0, 1, 2, 0, 4, 3, 0, 1, 2, 3, 4, 1, 0],
        device="cuda",
        dtype=torch.int32,
    )
    empty_rows = lengths == 0

    masked_q, masked_weights = fused_dsa_mask_empty_indexer_gradients(
        grad_q,
        grad_weights,
        lengths,
    )
    torch.testing.assert_close(
        masked_q,
        grad_q.masked_fill(empty_rows[:, None, None], 0),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        masked_weights,
        grad_weights.masked_fill(empty_rows[:, None], 0),
        atol=0.0,
        rtol=0.0,
    )

    grad_loss = torch.tensor(0.375, device="cuda", dtype=torch.float32)
    scaled_q, scaled_weights, scaled_k = fused_dsa_scale_indexer_gradients(
        masked_q,
        masked_weights,
        grad_k,
        grad_loss,
    )
    for actual, reference in (
        (scaled_q, masked_q * grad_loss),
        (scaled_weights, masked_weights * grad_loss),
        (scaled_k, grad_k * grad_loss),
    ):
        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)


def test_fused_nonfinite_row_diagnostics_match_torch() -> None:
    values = torch.tensor(
        [
            [0.0, 1.0, -2.0, 3.0, 4.0],
            [float("nan"), 1.0, float("inf"), float("nan"), 0.0],
            [float("-inf"), 2.0, float("-inf"), 4.0, 5.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    actual = dsa_nonfinite_row_counts(values).cpu()
    expected = torch.tensor(
        [[0, 0, 0, -1], [2, 1, 0, 0], [0, 0, 2, -1]],
        dtype=torch.int32,
    )
    assert torch.equal(actual, expected)


def test_fused_nonfinite_block_diagnostics_match_torch() -> None:
    values = torch.arange(2053, device="cuda", dtype=torch.float32) - 1024
    values[17] = float("nan")
    values[1027] = float("inf")
    values[2051] = float("-inf")
    counts, max_abs = dsa_nonfinite_block_stats(values)
    assert torch.equal(
        counts.cpu(),
        torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        max_abs.cpu(),
        torch.tensor([1024.0, 1023.0, 1028.0]),
        atol=0.0,
        rtol=0.0,
    )


def test_cuda_main_compressor_dispatches_fused_v4_rope(monkeypatch) -> None:
    config = _small_config(4)
    layer = MagiDSALayer(config).cuda()
    assert layer.compressor is not None
    compressed_rows = 7
    packed_x = torch.randn(
        compressed_rows,
        config.compressor_support,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    valid_rows = torch.ones(
        compressed_rows,
        config.compressor_support,
        device="cuda",
        dtype=torch.bool,
    )
    positions = torch.arange(compressed_rows, device="cuda", dtype=torch.int32)
    calls: list[tuple[torch.Size, torch.dtype, int]] = []

    def eager_rms_norm(
        tensor: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        value = tensor.float()
        return (
            value
            * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
            * weight
        ).to(tensor.dtype)

    def capture_fused_rotation(
        tensor: torch.Tensor,
        row_positions: torch.Tensor,
        inverse_frequencies: torch.Tensor,
        rope_dim: int,
    ) -> torch.Tensor:
        calls.append((tensor.shape, tensor.dtype, rope_dim))
        return apply_dsa_rope(
            tensor,
            row_positions,
            inverse_frequencies,
            rope_dim,
        )

    monkeypatch.setattr(dsa_layer_module, "_apply_fused_rms_norm", eager_rms_norm)
    monkeypatch.setattr(dsa_rope_module, "fused_dsa_rope", capture_fused_rotation)
    result = layer.compressor(packed_x, valid_rows, positions)
    assert calls == [
        (
            torch.Size((compressed_rows, 1, config.head_dim)),
            torch.bfloat16,
            config.rope_dim,
        )
    ]
    assert result.shape == (compressed_rows, config.head_dim)


def test_fused_v4_rope_hadamard_matches_eager_forward_backward() -> None:
    torch.manual_seed(92)
    config = MagiDSAConfig(ratio=4)
    positions = torch.tensor(
        [0, 1, 63, 65535, 131071],
        device="cuda",
        dtype=torch.int32,
    )
    inverse_frequencies = _yarn_inverse_frequencies(config).cuda()
    source = torch.randn(
        positions.numel(),
        config.indexer_heads,
        config.indexer_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_source = source.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(source)

    actual = fused_dsa_rope_hadamard(
        source,
        positions,
        inverse_frequencies,
        config.rope_dim,
    )
    reference = apply_normalized_hadamard(
        apply_dsa_rope(
            reference_source,
            positions,
            inverse_frequencies,
            config.rope_dim,
        )
    )
    torch.testing.assert_close(actual.float(), reference.float(), atol=2e-2, rtol=2e-2)

    actual_gradient = torch.autograd.grad(actual, source, gradient)[0]
    reference_gradient = torch.autograd.grad(reference, reference_source, gradient)[0]
    torch.testing.assert_close(
        actual_gradient.float(),
        reference_gradient.float(),
        atol=2e-2,
        rtol=2e-2,
    )


def test_fused_v4_rope_hadamard_fuses_fp32_to_bf16_projection_cast() -> None:
    torch.manual_seed(925)
    config = MagiDSAConfig(ratio=4)
    positions = torch.tensor(
        [0, 1, 63, 65535, 131071],
        device="cuda",
        dtype=torch.int32,
    )
    inverse_frequencies = _yarn_inverse_frequencies(config).cuda()
    source = torch.randn(
        positions.numel(),
        config.indexer_heads,
        config.indexer_head_dim,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_source = source.detach().clone().requires_grad_(True)
    gradient = torch.randn(source.shape, device="cuda", dtype=torch.bfloat16)

    actual = fused_dsa_rope_hadamard(
        source,
        positions,
        inverse_frequencies,
        config.rope_dim,
        output_dtype=torch.bfloat16,
    )
    reference = fused_dsa_rope_hadamard(
        reference_source.to(torch.bfloat16),
        positions,
        inverse_frequencies,
        config.rope_dim,
    )
    torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
    actual_gradient = torch.autograd.grad(actual, source, gradient)[0]
    reference_gradient = torch.autograd.grad(reference, reference_source, gradient)[0]
    torch.testing.assert_close(
        actual_gradient,
        reference_gradient,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_fused_projection_scale_cast_matches_eager_forward_backward(
    input_dtype: torch.dtype,
) -> None:
    torch.manual_seed(926)
    source = torch.randn(
        8193,
        device="cuda",
        dtype=input_dtype,
        requires_grad=True,
    )
    reference_source = source.detach().clone().requires_grad_(True)
    scale = 0.125
    gradient = torch.randn(source.shape, device="cuda", dtype=torch.bfloat16)

    actual = fused_dsa_scale_cast(source, scale, torch.bfloat16)
    reference = (reference_source * scale).to(torch.bfloat16)
    torch.testing.assert_close(actual.float(), reference.float(), atol=0.0, rtol=0.0)
    actual_gradient = torch.autograd.grad(actual, source, gradient)[0]
    reference_gradient = torch.autograd.grad(reference, reference_source, gradient)[0]
    torch.testing.assert_close(
        actual_gradient,
        reference_gradient,
        atol=0.0,
        rtol=0.0,
    )


def test_cuda_indexer_query_projection_dispatches_fused_v4_rotation(
    monkeypatch,
) -> None:
    config = _small_config(4)
    indexer = DsaIndexer(config).cuda()
    tokens = 7
    x = torch.randn(
        tokens,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    qr = torch.randn(
        tokens,
        config.q_lora_rank,
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    calls: list[tuple[torch.Size, torch.dtype, int, torch.dtype | None]] = []

    def capture_fused_rotation(
        tensor: torch.Tensor,
        row_positions: torch.Tensor,
        inverse_frequencies: torch.Tensor,
        rope_dim: int,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        calls.append((tensor.shape, tensor.dtype, rope_dim, output_dtype))
        rotated_input = tensor if output_dtype is None else tensor.to(output_dtype)
        return apply_normalized_hadamard(
            apply_dsa_rope(
                rotated_input,
                row_positions,
                inverse_frequencies,
                rope_dim,
            )
        )

    monkeypatch.setattr(
        dsa_rope_module,
        "fused_dsa_rope_hadamard",
        capture_fused_rotation,
    )
    q, weights = indexer.project_queries(x, qr, positions, detach_trunk=False)
    assert calls == [
        (
            torch.Size((tokens, config.indexer_heads, config.indexer_head_dim)),
            torch.bfloat16,
            config.rope_dim,
            torch.bfloat16,
        )
    ]
    assert q.shape == (tokens, config.indexer_heads, config.indexer_head_dim)
    assert weights.shape == (tokens, config.indexer_heads)


def test_cp1_token_layout_route_is_identity_and_reentrant() -> None:
    config = _small_config(4)
    plan = build_dsa_execution_plan(config, (0, 17), (17,))
    device_plan = make_dsa_device_rank_plan(plan, 0, config, torch.device("cuda"))
    route = device_plan.token_layout_route
    assert route is not None
    source = torch.randn(
        17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    first = route_dsa_tensor(source, route, None)
    second = route_dsa_tensor(source, route, None)
    assert torch.equal(first, source)
    assert torch.equal(second, source)
    gradient = torch.autograd.grad(first.float().sum() + second.float().sum(), source)[
        0
    ]
    assert torch.equal(gradient, torch.full_like(source, 2))


def test_cp1_typed_route_supports_two_private_inflight_transfers() -> None:
    config = _small_config(4)
    plan = build_dsa_execution_plan(config, (0, 17), (17,))
    device_plan = make_dsa_device_rank_plan(plan, 0, config, torch.device("cuda"))
    route = device_plan.token_layout_route
    assert route is not None
    first_source = torch.randn(
        17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    second_source = torch.randn(
        17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    first_transfer = start_dsa_tensor_route(first_source, route, None)
    second_transfer = start_dsa_tensor_route(second_source, route, None)
    assert first_transfer.output.data_ptr() != second_transfer.output.data_ptr()

    first = finish_dsa_tensor_route(first_transfer)
    second = finish_dsa_tensor_route(second_transfer)
    assert torch.equal(first, first_source)
    assert torch.equal(second, second_source)
    with pytest.raises(RuntimeError, match="already been finished"):
        finish_dsa_tensor_route(first_transfer)

    loss = first.float().sum() + second.float().sum()
    first_gradients = torch.autograd.grad(
        loss,
        (first_source, second_source),
        retain_graph=True,
    )
    second_gradients = torch.autograd.grad(loss, (first_source, second_source))
    for first_gradient, second_gradient, source in zip(
        first_gradients,
        second_gradients,
        (first_source, second_source),
    ):
        assert torch.equal(first_gradient, torch.ones_like(source))
        assert torch.equal(second_gradient, first_gradient)


def test_cp1_reverse_route_supports_two_private_inflight_transfers() -> None:
    config = _small_config(4)
    plan = build_dsa_execution_plan(config, (0, 17), (17,))
    device_plan = make_dsa_device_rank_plan(plan, 0, config, torch.device("cuda"))
    route = device_plan.token_layout_route
    assert route is not None
    first_consumer = torch.randn(17, 8, device="cuda", dtype=torch.bfloat16)
    second_consumer = torch.randn(17, 8, device="cuda", dtype=torch.bfloat16)

    first_transfer = start_dsa_reverse_route(first_consumer, route, None)
    second_transfer = start_dsa_reverse_route(second_consumer, route, None)
    assert first_transfer.output.data_ptr() != second_transfer.output.data_ptr()

    first = finish_dsa_reverse_route(first_transfer)
    second = finish_dsa_reverse_route(second_transfer)
    assert torch.equal(first, first_consumer)
    assert torch.equal(second, second_consumer)
    with pytest.raises(RuntimeError, match="already been finished"):
        finish_dsa_reverse_route(first_transfer)


def test_unused_typed_route_edge_does_not_materialize_a_reverse(
    monkeypatch,
) -> None:
    config = _small_config(4)
    plan = build_dsa_execution_plan(config, (0, 17), (17,))
    device_plan = make_dsa_device_rank_plan(plan, 0, config, torch.device("cuda"))
    route = device_plan.token_layout_route
    assert route is not None
    source = torch.randn(
        17,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    transfer = start_dsa_tensor_route(source, route, None)
    consumer = finish_dsa_tensor_route(transfer)

    class _IgnoreConsumerFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, direct, ignored):
            del ctx, ignored
            return direct.clone()

        @staticmethod
        def backward(ctx, gradient):
            del ctx
            return gradient, None

    def reject_reverse(*args, **kwargs):
        del args, kwargs
        raise AssertionError("an unused route edge launched a reverse transfer")

    monkeypatch.setattr(dsa_comm_module, "_launch_group_reduce", reject_reverse)
    output = _IgnoreConsumerFunction.apply(source, consumer)
    gradient = torch.autograd.grad(output.float().sum(), source)[0]
    assert torch.equal(gradient, torch.ones_like(source))


def test_csa_orchestration_projects_queries_between_initial_start_and_finish(
    monkeypatch,
) -> None:
    torch.manual_seed(93)
    config = _small_config(4)
    layer = MagiDSALayer(config).cuda()
    assert layer.indexer is not None
    tokens = 17
    meta = MagiDSAPackedMeta((0, tokens), (tokens,))
    runtime = MagiDSARuntimeMgr(config)
    handle = runtime.prepare_execution(
        meta,
        torch.device("cuda"),
        local_token_capacity=tokens,
        health_check=False,
    )
    dsa_input = MagiDSAInput(
        x=torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16),
        qr=torch.randn(tokens, config.q_lora_rank, device="cuda", dtype=torch.bfloat16),
        q=torch.randn(
            tokens,
            config.num_query_heads,
            config.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        latent_kv=torch.randn(
            tokens, config.head_dim, device="cuda", dtype=torch.bfloat16
        ),
        sink=torch.randn(config.num_query_heads, device="cuda", dtype=torch.float32),
        packed_meta=meta,
    )

    events: list[str] = []
    original_start = start_dsa_tensor_route
    # The schedule drives its own waits, so the observable wait is the transfer
    # method rather than the autograd-attaching finish helper.
    original_wait = dsa_comm_module.DsaRouteTransfer.wait
    original_project = layer.indexer.project_queries

    def record_start(source, route, group, *, attention_mode=None):
        assert attention_mode == "csa"
        if route.name == "COMPRESSED_KI":
            assert source.requires_grad
        events.append(f"start:{route.name}")
        return original_start(
            source,
            route,
            group,
            attention_mode=attention_mode,
        )

    def record_wait(transfer):
        events.append(f"finish:{transfer.route.name}")
        return original_wait(transfer)

    def record_project(*args, **kwargs):
        events.append("project:indexer_q")
        return original_project(*args, **kwargs)

    def fake_indexer(q_indexer, k_indexer, weights, mapping, backend_config):
        del k_indexer, weights
        assert handle.csa_indexer_stream is not None
        assert (
            torch.cuda.current_stream().cuda_stream
            == handle.csa_indexer_stream.cuda_stream
        )
        assert torch.cuda.current_stream().cuda_stream != 0
        events.append("indexer")
        rows = q_indexer.shape[0]
        global_ids = torch.full(
            (rows, backend_config.indexer_topk),
            -1,
            dtype=torch.int32,
            device=q_indexer.device,
        )
        return DsaIndexerSelection(
            global_ids=global_ids,
            lengths=torch.zeros(rows, dtype=torch.int32, device=q_indexer.device),
            lse=torch.full(
                (rows,), float("-inf"), dtype=torch.float32, device=q_indexer.device
            ),
            logical_score_calls=1,
            logical_topk_calls=1,
        )

    def fake_csa_attention_kl(q, *args, **kwargs):
        del args, kwargs
        assert handle.csa_indexer_stream is not None
        assert (
            torch.cuda.current_stream().cuda_stream
            == handle.csa_indexer_stream.cuda_stream
        )
        assert torch.cuda.current_stream().cuda_stream != 0
        events.append("attention")
        lse = torch.zeros(q.shape[:2], dtype=torch.float32, device=q.device)
        events.append("kl")
        return (
            q.clone(),
            torch.zeros((), dtype=torch.float32, device=q.device),
            lse,
        )

    monkeypatch.setattr(dist_dsa_module, "start_dsa_tensor_route", record_start)
    monkeypatch.setattr(dsa_comm_module.DsaRouteTransfer, "wait", record_wait)
    monkeypatch.setattr(layer.indexer, "project_queries", record_project)
    monkeypatch.setattr(dist_dsa_module, "run_grouped_dsa_indexer", fake_indexer)
    monkeypatch.setattr(
        dist_dsa_module,
        "dsa_csa_attention_kl",
        fake_csa_attention_kl,
    )
    layer.indexer.compressor.register_forward_pre_hook(
        lambda _module, _inputs: events.append("compressor:indexer")
    )
    assert layer.compressor is not None
    layer.compressor.register_forward_pre_hook(
        lambda _module, _inputs: events.append("compressor:main")
    )

    result = runtime.calc_dsa(layer.projections(), dsa_input, handle)

    assert result.output.shape == dsa_input.q.shape
    expected_events = {
        "start:WINDOW_KV",
        "start:OVERLAP_X",
        "project:indexer_q",
        "finish:OVERLAP_X",
        "compressor:indexer",
        "start:COMPRESSED_KI",
        "compressor:main",
        "start:COMPRESSED_KV",
        "finish:COMPRESSED_KI",
        "indexer",
        "finish:COMPRESSED_KV",
        "finish:WINDOW_KV",
        "attention",
        "kl",
    }
    assert set(events) == expected_events
    assert all(events.count(event) == 1 for event in expected_events)
    starts = [event for event in events if event.startswith("start:")]
    assert starts == [
        "start:WINDOW_KV",
        "start:OVERLAP_X",
        "start:COMPRESSED_KI",
        "start:COMPRESSED_KV",
    ]
    assert events.index("start:OVERLAP_X") < events.index("project:indexer_q")
    assert events.index("project:indexer_q") < events.index("finish:OVERLAP_X")
    assert events.index("finish:COMPRESSED_KI") < events.index("indexer")
    assert events.index("attention") < events.index("kl")


@pytest.mark.parametrize("ratio", [4, 128])
def test_all_ratio_device_plans_materialize_static_attention_maps(
    ratio: DsaRatio,
) -> None:
    config = _small_config(ratio)
    plan = build_dsa_execution_plan(
        config, (0, 17, 278), (31, 0, 100, 147)
    )
    for rank in range(plan.cp_size):
        device_plan = make_dsa_device_rank_plan(
            plan, rank, config, torch.device("cuda")
        )
        assert device_plan.local_q_positions.shape == (
            plan.rank_plans[rank].local_token_count,
        )
        # The window is a base and a length per query, not a padded matrix.
        assert device_plan.attention.window_base.shape == (
            plan.rank_plans[rank].local_token_count,
        )
        assert (
            device_plan.attention.window_length.numel()
            == plan.rank_plans[rank].local_token_count
        )
        assert device_plan.compression is not None
        assert (device_plan.indexer is not None) == (ratio == 4)
