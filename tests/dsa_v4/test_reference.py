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

from magi_attention.dsa_config import DsaRatio, MagiDSAConfig
from magi_attention.dsa_layer import MagiDSALayer, apply_dsa_rope
from magi_attention.functional.dsa_reference import (
    _compress_global,
    deterministic_topk_global_ids,
    dsa_reference,
    validate_canonical_topk,
    validate_deterministic_topk,
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


def _inputs(
    config: MagiDSAConfig, total_tokens: int, seed: int = 0
) -> tuple[torch.Tensor, ...]:
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


@pytest.mark.parametrize("ratio", [0, 4, 128])
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
    scores = (
        torch.einsum("qhd,kd->qhk", q_index.float(), compressed_ki.float()).relu()
        * weights.float().unsqueeze(-1)
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


def test_cutoff_tie_policy_uses_ascending_global_id() -> None:
    scores = torch.tensor([9.0, 8.0, 7.0, 7.0, 7.0, 2.0])
    expected = torch.tensor([100, 101, 102])
    assert torch.equal(
        deterministic_topk_global_ids(scores, 3, global_offset=100), expected
    )
    validate_deterministic_topk(scores, expected, 3, global_offset=100)
    with pytest.raises(AssertionError, match="score/global-ID"):
        validate_deterministic_topk(
            scores, torch.tensor([100, 101, 104]), 3, global_offset=100
        )

    near_scores = scores.clone()
    near_scores[2] = 7.001
    assert torch.equal(
        deterministic_topk_global_ids(near_scores, 3), torch.tensor([0, 1, 2])
    )


def test_q14_canonical_topk_ignores_cross_backend_internal_order() -> None:
    actual_ids = torch.tensor([[7, 2, 5, -1], [-1, -1, -1, -1]])
    expected_ids = torch.tensor([[5, 7, 2, -9], [-3, -3, -3, -3]])
    lengths = torch.tensor([3, 0], dtype=torch.int32)
    validate_canonical_topk(
        actual_ids,
        lengths,
        expected_ids,
        lengths.clone(),
    )
    assert not torch.equal(actual_ids, expected_ids)


@pytest.mark.parametrize(
    ("actual_ids", "actual_lengths", "message"),
    [
        (torch.tensor([[7, 2, 6, -1]]), torch.tensor([3]), "canonical"),
        (torch.tensor([[7, 2, 2, -1]]), torch.tensor([3]), "not unique"),
        (torch.tensor([[7, -1, 5, -1]]), torch.tensor([3]), "contains padding"),
        (torch.tensor([[7, 2, 5, 4]]), torch.tensor([3]), "padding contains"),
        (torch.tensor([[7, 2, 5, -1]]), torch.tensor([2]), "lengths differ"),
    ],
)
def test_q14_canonical_topk_rejects_contract_violations(
    actual_ids: torch.Tensor,
    actual_lengths: torch.Tensor,
    message: str,
) -> None:
    expected_ids = torch.tensor([[5, 7, 2, -1]])
    expected_lengths = torch.tensor([3])
    with pytest.raises(AssertionError, match=message):
        validate_canonical_topk(
            actual_ids,
            actual_lengths,
            expected_ids,
            expected_lengths,
        )
