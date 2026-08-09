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

import pytest
import torch
from magi_attn_extensions.DSA.config import DsaRatio, MagiDSAConfig
from magi_attn_extensions.DSA.modeling import (
    MagiDSALayer,
    _yarn_inverse_frequencies,
    apply_dsa_rope,
)
from magi_attn_extensions.DSA.reference import (
    _compress_global,
    assert_backend_native_topk_outputs_close,
    backend_native_topk_global_ids,
    dsa_position_ids,
    dsa_reference,
    validate_backend_native_topk,
    validate_backend_native_topk_pair,
)

# Relative, not "from tests.dsa_v4...": under extensions/ that absolute path is
# captured by the repository-root tests package, which is a different suite.
from .distributed_worker import (
    _compare_natural_snapshots,
    _cp2_backend_config,
    _reference_csa_index_scores,
)


def _config(ratio: DsaRatio) -> MagiDSAConfig:
    return MagiDSAConfig(
        ratio=ratio,
        hidden_size=8,
        q_lora_rank=6,
        num_query_heads=2,
        head_dim=8,
        rope_dim=4,
        indexer_heads=2,
        indexer_head_dim=4,
        indexer_topk=3,
        window_size=3,
        original_seq_len=0,
        compress_rope_theta=10000.0,
        indexer_atom_size=4,
    )


def test_cp2_correctness_uses_the_complete_pro_release_schema() -> None:
    config = _cp2_backend_config()
    config.validate_release_contract()


def _inputs(
    config: MagiDSAConfig, total_tokens: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    generator = torch.Generator().manual_seed(seed)
    tensors = (
        torch.randn(total_tokens, config.hidden_size, generator=generator),
        torch.randn(total_tokens, config.q_lora_rank, generator=generator),
        torch.randn(
            total_tokens, config.num_query_heads, config.head_dim, generator=generator
        ),
        torch.randn(total_tokens, config.head_dim, generator=generator),
        torch.randn(config.num_query_heads, generator=generator, dtype=torch.float32),
    )
    return tuple(tensor.requires_grad_(True) for tensor in tensors)


def _manual_trailing_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool,
) -> torch.Tensor:
    prefix = tensor[..., :-rope_dim]
    rotary = tensor[..., -rope_dim:]
    angle = positions.float().unsqueeze(-1) * inverse_frequencies.unsqueeze(0)
    if inverse:
        angle = -angle
    while angle.ndim < tensor.ndim:
        angle = angle.unsqueeze(1)
    even = rotary[..., 0::2]
    odd = rotary[..., 1::2]
    rotated = torch.stack(
        (
            even * angle.cos() - odd * angle.sin(),
            odd * angle.cos() + even * angle.sin(),
        ),
        dim=-1,
    ).flatten(-2)
    return torch.cat((prefix, rotated), dim=-1)


@pytest.mark.parametrize("inverse", [False, True])
def test_cpu_rope_formula_and_backward_use_opposite_rotation(inverse: bool) -> None:
    config = _config(4)
    positions = torch.tensor([0, 5, 1, 7], dtype=torch.int64)
    frequencies = _yarn_inverse_frequencies(config)
    source = torch.randn(
        positions.numel(),
        config.num_query_heads,
        config.head_dim,
        dtype=torch.float32,
        requires_grad=True,
    )
    gradient = torch.randn_like(source)

    actual = apply_dsa_rope(
        source,
        positions,
        frequencies,
        config.rope_dim,
        inverse=inverse,
    )
    expected = _manual_trailing_rope(
        source,
        positions,
        frequencies,
        config.rope_dim,
        inverse=inverse,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual_gradient = torch.autograd.grad(actual, source, gradient)[0]
    opposite_rotation = _manual_trailing_rope(
        gradient,
        positions,
        frequencies,
        config.rope_dim,
        inverse=not inverse,
    )
    torch.testing.assert_close(
        actual_gradient,
        opposite_rotation,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("ratio", [4, 128])
def test_layer_output_rope_frequencies_are_nonpersistent(ratio: DsaRatio) -> None:
    config = _config(ratio)
    layer = MagiDSALayer(config)
    torch.testing.assert_close(
        layer.output_inverse_frequencies,
        _yarn_inverse_frequencies(config),
    )
    assert "output_inverse_frequencies" not in layer.state_dict()
    positions = torch.tensor([0, 4, 1], dtype=torch.int64)
    raw_output = torch.randn(positions.numel(), config.num_query_heads, config.head_dim)
    actual = layer.inverse_output_rope(raw_output, positions)
    expected = _manual_trailing_rope(
        raw_output,
        positions,
        layer.output_inverse_frequencies,
        config.rope_dim,
        inverse=True,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("ratio", "cu_seqlens"), [(4, (0, 12, 20)), (128, (0, 130, 132))]
)
def test_reference_applies_official_inverse_rope_output_contract(
    ratio: DsaRatio,
    cu_seqlens: tuple[int, ...],
    monkeypatch,
) -> None:
    config = _config(ratio)
    layer = MagiDSALayer(config)
    inputs = _inputs(config, cu_seqlens[-1], seed=1701 + ratio)
    captured: dict[str, torch.Tensor] = {}

    def apply_official_output_boundary(
        raw_output: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if captured:
            raise AssertionError("output inverse RoPE must run exactly once")
        captured["raw_output"] = raw_output
        captured["positions"] = positions
        return _manual_trailing_rope(
            raw_output,
            positions,
            layer.output_inverse_frequencies,
            config.rope_dim,
            inverse=True,
        )

    monkeypatch.setattr(layer, "inverse_output_rope", apply_official_output_boundary)
    result = dsa_reference(layer, *inputs, cu_seqlens)

    positions = dsa_position_ids(cu_seqlens, device=torch.device("cpu"))
    assert torch.equal(captured["positions"], positions)
    expected = _manual_trailing_rope(
        captured["raw_output"],
        positions,
        layer.output_inverse_frequencies,
        config.rope_dim,
        inverse=True,
    )
    torch.testing.assert_close(result.output, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        result.output[..., : -config.rope_dim],
        captured["raw_output"][..., : -config.rope_dim],
        atol=0.0,
        rtol=0.0,
    )
    sample_starts = torch.tensor(cu_seqlens[:-1], dtype=torch.int64)
    torch.testing.assert_close(
        result.output.index_select(0, sample_starts),
        captured["raw_output"].index_select(0, sample_starts),
        atol=0.0,
        rtol=0.0,
    )
    nonzero_rows = positions > 0
    assert not torch.allclose(
        result.output[nonzero_rows, :, -config.rope_dim :],
        captured["raw_output"][nonzero_rows, :, -config.rope_dim :],
    )


@pytest.mark.parametrize("ratio", [4, 128])
def test_reference_forward_backward_for_all_layer_forms(ratio: DsaRatio) -> None:
    config = _config(ratio)
    layer = MagiDSALayer(config)
    cu_seqlens = (0, 5, 5, 136)
    inputs = _inputs(config, cu_seqlens[-1], seed=ratio)
    x, qr, q, latent_kv, sink = inputs
    result = dsa_reference(layer, x, qr, q, latent_kv, sink, cu_seqlens)

    assert result.output.shape == inputs[2].shape
    assert result.sparse_lse.shape == (cu_seqlens[-1], config.num_query_heads)
    assert result.indexer_lse.shape == (cu_seqlens[-1],)
    assert result.topk_ids.shape == (
        cu_seqlens[-1],
        config.indexer_topk if ratio == 4 else 0,
    )
    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.sparse_lse).all()
    assert torch.isfinite(result.kl)

    (result.output.float().square().mean() + result.kl).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert latent_kv.grad is not None and torch.isfinite(latent_kv.grad).all()
    assert sink.grad is not None and torch.isfinite(sink.grad).all()
    if ratio == 0:
        assert x.grad is None
        assert qr.grad is None
        assert list(layer.parameters()) == []
    elif ratio == 4:
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert qr.grad is not None and torch.isfinite(qr.grad).all()
        assert layer.indexer is not None
        assert all(parameter.grad is not None for parameter in layer.parameters())
    else:
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert qr.grad is None
        assert layer.compressor is not None
        assert all(
            parameter.grad is not None for parameter in layer.compressor.parameters()
        )


def test_csa_natural_topk_and_lse_match_direct_scores_exactly() -> None:
    config = _config(4)
    layer = MagiDSALayer(config)
    cu_seqlens = (0, 20)
    x, qr, q, latent_kv, sink = _inputs(config, 20, seed=11)
    result = dsa_reference(layer, x, qr, q, latent_kv, sink, cu_seqlens)

    assert layer.indexer is not None
    positions = torch.arange(20)
    q_index, weights = layer.indexer.project_queries(
        x,
        qr,
        positions,
        detach_trunk=False,
    )
    compressed_ki, _, _ = _compress_global(layer.indexer.compressor, x, cu_seqlens)
    score_weights = weights.float()
    scores = (
        torch.einsum("qhd,kd->qhk", q_index.float(), compressed_ki.float()).relu()
        * config.indexer_head_dim**-0.5
        * score_weights.unsqueeze(-1)
    ).sum(dim=1)
    for position in range(20):
        visible = (position + 1) // 4
        length = min(config.indexer_topk, visible)
        assert result.topk_length[position].item() == length
        if not length:
            assert torch.all(result.topk_ids[position] == -1)
            assert torch.isneginf(result.indexer_lse[position])
            continue
        expected = torch.topk(
            scores[position, :visible], k=length, sorted=True
        ).indices.to(torch.int32)
        assert torch.equal(result.topk_ids[position, :length], expected)
        assert torch.all(result.topk_ids[position, length:] == -1)
        torch.testing.assert_close(
            result.indexer_lse[position],
            torch.logsumexp(scores[position, :visible], dim=0),
        )


def test_indexer_projection_applies_only_the_head_averaging_scale() -> None:
    config = _config(4)
    layer = MagiDSALayer(config)
    assert layer.indexer is not None
    x, qr, _, _, _ = _inputs(config, 5, seed=19)
    positions = torch.arange(5)

    _, score_weights = layer.indexer.project_queries(
        x,
        qr,
        positions,
        detach_trunk=False,
    )
    unscaled = x.float() @ layer.indexer.weights_proj.weight.float().t()
    expected = unscaled * config.indexer_heads**-0.5
    torch.testing.assert_close(score_weights.float(), expected)
    complete_prescale = expected * config.indexer_head_dim**-0.5
    assert not torch.equal(score_weights.float(), complete_prescale)


def test_distributed_raw_score_reference_applies_backend_score_scale() -> None:
    torch.manual_seed(20)
    config = _config(4)
    layer = MagiDSALayer(config)
    assert layer.indexer is not None
    cu_seqlens = (0, 8, 20)
    inputs = _inputs(config, cu_seqlens[-1], seed=20)

    actual = _reference_csa_index_scores(layer, inputs, cu_seqlens)
    x, qr = inputs[:2]
    positions = dsa_position_ids(cu_seqlens, device=x.device)
    q_index, weights = layer.indexer.project_queries(
        x,
        qr,
        positions,
        detach_trunk=False,
    )
    compressed_ki, block_offsets, block_counts = _compress_global(
        layer.indexer.compressor,
        x,
        cu_seqlens,
    )
    expected = torch.full_like(actual, float("-inf"))
    for sample_id, (q_begin, q_end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        block_begin = block_offsets[sample_id]
        block_end = block_begin + block_counts[sample_id]
        dots = torch.einsum(
            "qhd,kd->qhk",
            q_index[q_begin:q_end].float(),
            compressed_ki[block_begin:block_end].float(),
        )
        expected[q_begin:q_end, block_begin:block_end] = (
            dots.relu()
            * config.indexer_head_dim**-0.5
            * weights[q_begin:q_end].float().unsqueeze(-1)
        ).sum(dim=1)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    finite = torch.isfinite(expected)
    unscaled = expected[finite] * config.indexer_head_dim**0.5
    assert not torch.equal(actual[finite], unscaled)


def test_csa_explicit_detach_only_cuts_indexer_activation_trunk() -> None:
    config = _config(4)
    layer = MagiDSALayer(config)
    cu_seqlens = (0, 20)
    x, qr, q, latent_kv, sink = _inputs(config, 20, seed=23)

    attached = dsa_reference(
        layer,
        x,
        qr,
        q,
        latent_kv,
        sink,
        cu_seqlens,
        detach_indexer_trunk=False,
    )
    attached.kl.backward()
    assert x.grad is not None and torch.count_nonzero(x.grad) > 0
    assert qr.grad is not None and torch.count_nonzero(qr.grad) > 0
    assert q.grad is None
    assert latent_kv.grad is None
    assert sink.grad is None
    assert layer.indexer is not None
    attached_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in layer.indexer.named_parameters()
        if parameter.grad is not None
    }
    assert set(attached_parameter_gradients) == {
        name for name, _ in layer.indexer.named_parameters()
    }

    layer.zero_grad(set_to_none=True)
    x.grad = None
    qr.grad = None
    detached = dsa_reference(
        layer,
        x,
        qr,
        q,
        latent_kv,
        sink,
        cu_seqlens,
        detach_indexer_trunk=True,
    )
    detached.kl.backward()
    assert x.grad is None
    assert qr.grad is None
    assert q.grad is None
    assert latent_kv.grad is None
    assert sink.grad is None
    torch.testing.assert_close(detached.kl, attached.kl.detach())
    for name, parameter in layer.indexer.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, attached_parameter_gradients[name])


def test_csa_overlap_compressor_matches_official_two_branch_formula() -> None:
    config = _config(4)
    layer = MagiDSALayer(config)
    assert layer.compressor is not None
    compressor = layer.compressor
    with torch.no_grad():
        compressor.wkv.weight.zero_()
        compressor.wgate.weight.zero_()
        compressor.ape.zero_()
        compressor.norm.weight.fill_(1.0)
        compressor.wkv.weight[: config.head_dim, : config.head_dim].copy_(
            torch.eye(config.head_dim)
        )
        compressor.wkv.weight[config.head_dim :, : config.head_dim].copy_(
            torch.eye(config.head_dim)
        )

    x = (
        torch.arange(8 * config.hidden_size, dtype=torch.float32).view(
            8, config.hidden_size
        )
        / 32.0
    )
    compressed, _, _ = _compress_global(compressor, x, (0, 8))
    first_unscaled = x[:4].mean(dim=0)
    second_unscaled = x.mean(dim=0)
    expected_unscaled = torch.stack((first_unscaled, second_unscaled))
    expected = expected_unscaled * torch.rsqrt(
        expected_unscaled.square().mean(dim=-1, keepdim=True) + config.norm_eps
    )
    expected = apply_dsa_rope(
        expected,
        torch.tensor([0, 4]),
        compressor.inverse_frequencies,
        config.rope_dim,
    )
    torch.testing.assert_close(compressed, expected, atol=1e-6, rtol=1e-6)


def test_backend_native_cutoff_tie_allows_any_cutoff_subset() -> None:
    scores = torch.tensor([9.0, 8.0, 7.0, 7.0, 7.0, 2.0])
    native = backend_native_topk_global_ids(scores, 3, global_offset=100)
    validate_backend_native_topk(scores, native, 3, global_offset=100)
    validate_backend_native_topk(
        scores, torch.tensor([100, 101, 102]), 3, global_offset=100
    )
    validate_backend_native_topk(
        scores, torch.tensor([100, 101, 104]), 3, global_offset=100
    )
    with pytest.raises(AssertionError, match="strictly above"):
        validate_backend_native_topk(
            scores, torch.tensor([100, 102, 104]), 3, global_offset=100
        )

    near_scores = scores.clone()
    near_scores[2] = 7.001
    near_native = backend_native_topk_global_ids(near_scores, 3)
    validate_backend_native_topk(near_scores, near_native, 3)
    assert set(near_native.tolist()) == {0, 1, 2}


def test_backend_native_topk_pair_allows_tied_set_and_order_differences() -> None:
    actual_ids = torch.tensor([[7, 2, 6, -1], [-1, -1, -1, -1]])
    expected_ids = torch.tensor([[5, 7, 2, -9], [-3, -3, -3, -3]])
    lengths = torch.tensor([3, 0], dtype=torch.int32)
    validate_backend_native_topk_pair(
        actual_ids,
        lengths,
        expected_ids,
        lengths.clone(),
    )
    assert not torch.equal(
        torch.sort(actual_ids[0, :3]).values,
        torch.sort(expected_ids[0, :3]).values,
    )


def test_backend_native_output_comparison_exempts_only_set_changed_rows() -> None:
    actual_ids = torch.tensor([[7, 3, -1], [8, 2, -1]], dtype=torch.int32)
    expected_ids = torch.tensor([[3, 6, -1], [2, 8, -1]], dtype=torch.int32)
    lengths = torch.tensor([2, 2], dtype=torch.int32)
    actual_output = torch.zeros(2, 2, 2)
    expected_output = actual_output.clone()
    actual_output[0] = 3.0

    diagnostics = assert_backend_native_topk_outputs_close(
        actual_output,
        expected_output,
        actual_ids,
        lengths,
        expected_ids,
        lengths.clone(),
        row_ids=torch.tensor([101, 205]),
    )

    assert diagnostics["canonical_topk_exact"] is False
    assert diagnostics["canonical_topk_mismatch_local_rows"] == [0]
    assert diagnostics["canonical_topk_mismatch_global_rows"] == [101]
    assert diagnostics["output_tie_exempt_rows"] == 1
    assert diagnostics["output_compared_rows"] == 1
    assert diagnostics["output_non_tie_max_abs"] == 0.0
    assert diagnostics["output_max_abs"] == 3.0
    assert diagnostics["output_tie_exempt_row_diagnostics"] == [
        {
            "actual_output_max_abs": 3.0,
            "expected_output_max_abs": 0.0,
            "global_row": 101,
            "local_row": 0,
            "output_max_abs_difference": 3.0,
        }
    ]

    actual_output[1] = 1.0
    with pytest.raises(AssertionError, match="equal canonical Top-K sets"):
        assert_backend_native_topk_outputs_close(
            actual_output,
            expected_output,
            actual_ids,
            lengths,
            expected_ids,
            lengths,
        )


def test_backend_native_output_comparison_requires_finite_exempt_rows() -> None:
    actual_output = torch.tensor([[[float("nan")]]])
    expected_output = torch.zeros_like(actual_output)
    with pytest.raises(AssertionError, match="non-finite"):
        assert_backend_native_topk_outputs_close(
            actual_output,
            expected_output,
            torch.tensor([[7, 3]], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([[7, 6]], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        )


def _q16_natural_snapshot(*, changed_output: bool) -> dict[str, torch.Tensor]:
    output = torch.zeros(2, 1, 2)
    if changed_output:
        output[0] = 3.0
    return {
        "grad_x": torch.zeros(2, 2),
        "indexer_lse": torch.zeros(2),
        "kl_global": torch.zeros(()),
        "output": output,
        "sparse_lse": torch.zeros(2, 1),
        "topk_ids": torch.tensor(
            [[7, 3, -1], [8, 2, -1]] if changed_output else [[7, 6, -1], [2, 8, -1]],
            dtype=torch.int32,
        ),
        "topk_length": torch.tensor([2, 2], dtype=torch.int32),
    }


@pytest.mark.parametrize(
    "field",
    ("sparse_lse", "indexer_lse", "kl_global", "grad_x"),
)
def test_natural_snapshot_tie_exemption_does_not_cover_other_values(
    field: str,
) -> None:
    actual = _q16_natural_snapshot(changed_output=True)
    expected = _q16_natural_snapshot(changed_output=False)
    metrics, _ = _compare_natural_snapshots(
        actual,
        expected,
        label="Q16 test",
        gradient_names=("grad_x",),
        compare_parameter_values=False,
    )
    assert metrics["output_tie_exempt_rows"] == 1
    assert metrics["output_non_tie_max_abs"] == 0.0

    actual[field].add_(1.0)
    with pytest.raises(AssertionError):
        _compare_natural_snapshots(
            actual,
            expected,
            label="Q16 test",
            gradient_names=("grad_x",),
            compare_parameter_values=False,
        )


@pytest.mark.parametrize(
    ("actual_ids", "actual_lengths", "message"),
    [
        (torch.tensor([[7, 2, 2, -1]]), torch.tensor([3]), "not unique"),
        (torch.tensor([[7, -1, 5, -1]]), torch.tensor([3]), "contains padding"),
        (torch.tensor([[7, 2, 5, 4]]), torch.tensor([3]), "padding contains"),
        (torch.tensor([[7, 2, 5, -1]]), torch.tensor([2]), "lengths differ"),
    ],
)
def test_backend_native_topk_pair_rejects_structural_violations(
    actual_ids: torch.Tensor,
    actual_lengths: torch.Tensor,
    message: str,
) -> None:
    expected_ids = torch.tensor([[5, 7, 2, -1]])
    expected_lengths = torch.tensor([3])
    with pytest.raises(AssertionError, match=message):
        validate_backend_native_topk_pair(
            actual_ids,
            actual_lengths,
            expected_ids,
            expected_lengths,
        )
