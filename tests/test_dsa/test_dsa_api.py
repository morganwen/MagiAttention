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

"""Step-1 contract and CP=1 parity tests for the public Magi_DSA API."""

import copy
import io
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from magi_attention.api import DsaPackedMeta, MagiDSAInput, calc_dsa
from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr
from magi_attention.experimental.dsa_v4 import DSAv4Compressor, MagiDSAV4Config
from magi_attention.testing.precision import assert_close as assert_precision_close


def _make_config(ratio: int, backend: str = "reference", **kwargs) -> MagiDSAV4Config:
    values = dict(
        compress_ratio=ratio,
        hidden_size=64,
        q_lora_rank=64,
        softmax_scale=512**-0.5,
        backend=backend,
    )
    values.update(kwargs)
    return MagiDSAV4Config(**values)


def _make_input(
    cfg: MagiDSAV4Config,
    lengths: list[int],
    *,
    requires_grad: bool = True,
) -> MagiDSAInput:
    total = sum(lengths)

    def make(*shape):
        tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        return tensor.requires_grad_(requires_grad)

    sink = torch.randn(cfg.num_heads, device="cuda", dtype=torch.float32)
    sink.requires_grad_(requires_grad)
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32
    )
    return MagiDSAInput(
        x=make(total, cfg.hidden_size),
        qr=make(total, cfg.q_lora_rank),
        q=make(total, cfg.num_heads, cfg.kv_dim),
        latent_kv=make(total, cfg.kv_dim),
        sink=sink,
        packed_meta=DsaPackedMeta(cu_seqlens),
    )


def _clone_input(dsa_input: MagiDSAInput) -> MagiDSAInput:
    def clone(tensor):
        return tensor.detach().clone().requires_grad_(tensor.requires_grad)

    return MagiDSAInput(
        x=clone(dsa_input.x),
        qr=clone(dsa_input.qr),
        q=clone(dsa_input.q),
        latent_kv=clone(dsa_input.latent_kv),
        sink=clone(dsa_input.sink),
        packed_meta=DsaPackedMeta(dsa_input.packed_meta.cu_seqlens.clone()),
    )


def _collect_input_grads(dsa_input: MagiDSAInput) -> dict[str, torch.Tensor | None]:
    return {
        name: None if tensor.grad is None else tensor.grad.detach().clone()
        for name, tensor in (
            ("x", dsa_input.x),
            ("qr", dsa_input.qr),
            ("q", dsa_input.q),
            ("latent_kv", dsa_input.latent_kv),
            ("sink", dsa_input.sink),
        )
    }


def _assert_optional_close(
    actual, expected, *, rtol=0.0, atol=0.0, mismatch_threshold=0.0
):
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert_precision_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            mismatch_threshold=mismatch_threshold,
            allow_none=False,
            print_tensor_when_mismatch=False,
        )


class TestDsaContract:
    def test_packed_meta_rejects_bad_shape_dtype_and_bounds(self):
        with pytest.raises(ValueError, match="1-D"):
            DsaPackedMeta(torch.zeros(2, 2, dtype=torch.int32)).validate()
        with pytest.raises(TypeError, match="torch.int32"):
            DsaPackedMeta(torch.tensor([0, 4], dtype=torch.int64)).validate()
        with pytest.raises(ValueError, match=r"\[0\]"):
            DsaPackedMeta(torch.tensor([1, 4], dtype=torch.int32)).validate()
        with pytest.raises(ValueError, match="non-decreasing"):
            DsaPackedMeta(torch.tensor([0, 5, 4], dtype=torch.int32)).validate()
        with pytest.raises(ValueError, match="packed token count"):
            DsaPackedMeta(torch.tensor([0, 4], dtype=torch.int32)).validate(5)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("num_heads", 32),
            ("kv_dim", 256),
            ("rope_dim", 32),
            ("window_size", 64),
            ("topk", 128),
            ("indexer_heads", 32),
            ("indexer_dim", 64),
            ("params_dtype", "float32"),
            ("use_sparse_loss", False),
            ("calculate_per_token_loss", True),
        ],
    )
    def test_runtime_rejects_non_contract_config(self, field, value):
        cfg = replace(_make_config(0), **{field: value})
        with pytest.raises(ValueError):
            MagiDSARuntimeMgr(cfg)

    def test_runtime_builds_cp8_forward_capability(self):
        sentinel_group = object()
        with (
            patch(
                "magi_attention.dsa_runtime_mgr.dist.is_initialized", return_value=True
            ),
            patch("magi_attention.dsa_runtime_mgr.dist.get_rank", return_value=7),
            patch("magi_attention.dsa_runtime_mgr.dist.get_world_size", return_value=8),
        ):
            runtime = MagiDSARuntimeMgr(_make_config(0), cp_group=sentinel_group)
        assert runtime.plan.cp_rank == 7
        assert runtime.plan.cp_size == 8
        assert runtime.plan.compress_ratio == 0
        assert runtime.plan.communication_ready

    @pytest.mark.parametrize("world_size", [3, 4, 5, 6, 7, 9])
    def test_runtime_rejects_unvalidated_cp_size(self, world_size):
        sentinel_group = object()
        with (
            patch(
                "magi_attention.dsa_runtime_mgr.dist.is_initialized", return_value=True
            ),
            patch("magi_attention.dsa_runtime_mgr.dist.get_rank", return_value=0),
            patch(
                "magi_attention.dsa_runtime_mgr.dist.get_world_size",
                return_value=world_size,
            ),
            pytest.raises(ValueError, match="CP sizes 1, 2, or 8"),
        ):
            MagiDSARuntimeMgr(_make_config(0), cp_group=sentinel_group)

    def test_runtime_rejects_unknown_dispatch_policy(self):
        with pytest.raises(ValueError, match="dispatch_policy"):
            MagiDSARuntimeMgr(_make_config(0), dispatch_policy="unknown")

    def test_runtime_caches_fragment_plan_by_packed_layout_and_policy(self):
        runtime = MagiDSARuntimeMgr(_make_config(4))
        packed_meta = DsaPackedMeta(torch.tensor([0, 257, 270], dtype=torch.int32))
        balanced = runtime.get_dispatch_plan(packed_meta)
        assert balanced.sample_lengths == (257, 13)
        assert balanced.compress_ratio == 4
        assert balanced.cp_size == 1
        assert runtime.get_dispatch_plan(packed_meta) is balanced
        assert runtime.plan_cache_size == 1

        sequential = runtime.get_dispatch_plan(packed_meta, policy="sequential")
        assert sequential.policy == "sequential"
        assert runtime.plan_cache_size == 2

    def test_runtime_deepcopy_and_module_serialization_rebuild_locks(self):
        runtime = MagiDSARuntimeMgr(_make_config(4))
        packed_meta = DsaPackedMeta(torch.tensor([0, 257], dtype=torch.int32))
        expected = runtime.get_dispatch_plan(packed_meta)

        copied = copy.deepcopy(runtime)
        assert copied.get_dispatch_plan(packed_meta).plan_hash == expected.plan_hash

        buffer = io.BytesIO()
        torch.save(runtime, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        assert restored.get_dispatch_plan(packed_meta).plan_hash == expected.plan_hash

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_input_rejects_shape_dtype_and_device_mismatch(self):
        runtime = MagiDSARuntimeMgr(_make_config(0)).cuda()
        good = _make_input(runtime.config, [8])
        runtime.validate_input(good)

        with pytest.raises(ValueError, match="q must have shape"):
            runtime.validate_input(replace(good, q=good.q[:, :, :-1]))
        with pytest.raises(TypeError, match="q must have dtype"):
            runtime.validate_input(replace(good, q=good.q.float()))
        with pytest.raises(TypeError, match="sink must have dtype"):
            runtime.validate_input(replace(good, sink=good.sink.bfloat16()))
        with pytest.raises(ValueError, match="share device"):
            runtime.validate_input(replace(good, sink=good.sink.cpu()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCompressor:
    def test_tail_rule_and_short_sample(self):
        cfg4 = _make_config(4)
        comp4 = DSAv4Compressor(cfg4, head_dim=cfg4.kv_dim).cuda()
        x4 = torch.randn(10, 1, cfg4.hidden_size, device="cuda", dtype=torch.bfloat16)
        assert comp4(x4).shape == (2, 1, cfg4.kv_dim)

        cfg128 = _make_config(128)
        comp128 = DSAv4Compressor(cfg128, head_dim=cfg128.kv_dim).cuda()
        x128 = torch.randn(
            100, 1, cfg128.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        assert comp128(x128) is None

    def test_ratio4_overlap_boundary(self):
        cfg = _make_config(4)
        compressor = DSAv4Compressor(cfg, head_dim=cfg.kv_dim).cuda()
        x = torch.randn(8, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        base = compressor(x)

        next_block_changed = x.clone()
        next_block_changed[4] += 1
        torch.testing.assert_close(base[0], compressor(next_block_changed)[0])

        previous_block_changed = x.clone()
        previous_block_changed[0] += 1
        assert not torch.equal(base[1], compressor(previous_block_changed)[1])


def _run_public_legacy_parity(ratio: int, backend: str) -> None:
    torch.manual_seed(20260709 + ratio)
    cfg = _make_config(ratio, backend=backend)
    runtime = MagiDSARuntimeMgr(cfg).cuda().train()
    lengths = {0: [7, 13], 4: [5, 27], 128: [130, 3]}[ratio]
    legacy_input = _make_input(cfg, lengths)
    public_input = _clone_input(legacy_input)
    output_grad = torch.randn(
        sum(lengths), cfg.num_heads, cfg.kv_dim, device="cuda", dtype=torch.float32
    )

    legacy_output, legacy_kl = runtime.dsa_module.forward_packed(
        legacy_input.x,
        legacy_input.qr,
        legacy_input.q,
        legacy_input.latent_kv,
        legacy_input.sink,
        legacy_input.packed_meta.cu_seqlens,
    )
    legacy_output = legacy_output.reshape_as(output_grad)
    (legacy_output.float().mul(output_grad).sum() + legacy_kl).backward()
    legacy_input_grads = _collect_input_grads(legacy_input)
    legacy_param_grads = {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in runtime.dsa_module.named_parameters()
    }
    runtime.zero_grad(set_to_none=True)

    public_output, public_kl = calc_dsa(public_input, runtime)
    (public_output.float().mul(output_grad).sum() + public_kl).backward()

    torch.testing.assert_close(public_output, legacy_output, rtol=0, atol=0)
    torch.testing.assert_close(public_kl, legacy_kl, rtol=0, atol=0)
    assert public_output.shape == (sum(lengths), 64, 512)
    assert public_kl.shape == torch.Size([])
    assert public_kl.dtype == torch.float32

    public_input_grads = _collect_input_grads(public_input)
    grad_rtol, grad_atol = (0.0, 0.0) if backend == "reference" else (2e-2, 1e-3)
    mismatch_threshold = 0.0 if backend == "reference" else 1e-3
    for name, expected in legacy_input_grads.items():
        _assert_optional_close(
            public_input_grads[name],
            expected,
            rtol=grad_rtol,
            atol=grad_atol,
            mismatch_threshold=mismatch_threshold,
        )
    for name, param in runtime.dsa_module.named_parameters():
        _assert_optional_close(
            param.grad,
            legacy_param_grads[name],
            rtol=grad_rtol,
            atol=grad_atol,
            mismatch_threshold=mismatch_threshold,
        )

    assert public_input.sink.grad is not None
    assert public_input.sink.grad.abs().sum() > 0
    if ratio == 4:
        assert public_kl > 0
        assert public_input.qr.grad is None
    else:
        assert public_kl == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_public_api_reference_matches_legacy_full_gradients(ratio):
    _run_public_legacy_parity(ratio, backend="reference")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="B300/CUDA required")
def test_public_api_saved_state_keeps_indexer_topk_not_sparse_indices():
    import magi_attention.functional.dist_dsa as dist_dsa_module
    from magi_attention.experimental.dsa_v4.reference import get_window_topk_idxs

    torch.manual_seed(20260710)
    cfg = _make_config(4, backend="reference")
    runtime = MagiDSARuntimeMgr(cfg).cuda().train()
    lengths = (20, 12)
    dsa_input = _make_input(cfg, list(lengths))
    saved = []
    final_sparse_indices = []

    score_values = (5.0, 7.0, 7.0, 3.0, 6.0)

    def deterministic_scores(q_idx, _weights, k_idx):
        values = torch.tensor(
            score_values[: k_idx.size(0)],
            dtype=torch.float32,
            device=q_idx.device,
        )
        return values.view(1, 1, -1).expand(q_idx.size(1), q_idx.size(0), -1)

    run_attention = dist_dsa_module._run_attention_forward_state

    def capture_sparse_indices(q, kv, sink, sparse_indices, runtime_mgr):
        final_sparse_indices.append(sparse_indices.detach().clone())
        return run_attention(q, kv, sink, sparse_indices, runtime_mgr)

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with (
        patch(
            "magi_attention.experimental.dsa_v4.indexer.compute_index_scores",
            side_effect=deterministic_scores,
        ),
        patch.object(
            dist_dsa_module,
            "_run_attention_forward_state",
            side_effect=capture_sparse_indices,
        ),
        torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
    ):
        output, kl_loss = calc_dsa(dsa_input, runtime)

    assert len(saved) == 9
    for actual, expected in zip(
        saved[:5],
        (
            dsa_input.x,
            dsa_input.qr,
            dsa_input.q,
            dsa_input.latent_kv,
            dsa_input.sink,
        ),
    ):
        assert actual.data_ptr() == expected.data_ptr()
    assert saved[5].data_ptr() == output.data_ptr()
    assert saved[6].dtype == torch.float32  # LSE (empty for reference seam)
    topk_idx = saved[7]
    topk_length = saved[8]
    assert topk_idx.dtype == torch.int32
    assert topk_idx.shape == (dsa_input.x.size(0), cfg.topk)
    assert topk_length.dtype == torch.int32
    assert topk_length.shape == (dsa_input.x.size(0),)
    torch.testing.assert_close(
        topk_length,
        (topk_idx >= 0).sum(dim=-1, dtype=torch.int32),
        rtol=0,
        atol=0,
    )

    assert len(final_sparse_indices) == len(lengths)
    row_begin = 0
    global_block_begin = 0
    for length, sparse_indices in zip(lengths, final_sparse_indices):
        row_end = row_begin + length
        sample_topk = topk_idx[row_begin:row_end]
        sample_topk_length = topk_length[row_begin:row_end]
        block_count = length // cfg.compress_ratio

        # The Indexer state is fixed-width global logical block ids.  Sparse
        # attention consumes a separate window + sample-local KV-row tensor.
        assert sparse_indices.dtype == torch.int32
        assert sparse_indices.shape == (length, cfg.window_size + cfg.topk)
        expected_window = (
            get_window_topk_idxs(cfg.window_size, 1, length, sparse_indices.device)
            .squeeze(0)
            .to(torch.int32)
        )
        torch.testing.assert_close(
            sparse_indices[:, : cfg.window_size], expected_window, rtol=0, atol=0
        )
        expected_compressed_rows = torch.where(
            sample_topk >= 0,
            sample_topk - global_block_begin + length,
            torch.full_like(sample_topk, -1),
        )
        torch.testing.assert_close(
            sparse_indices[:, cfg.window_size :],
            expected_compressed_rows,
            rtol=0,
            atol=0,
        )

        for position in range(length):
            visible = min((position + 1) // cfg.compress_ratio, block_count, cfg.topk)
            expected_local = sorted(
                range(visible), key=lambda block: (-score_values[block], block)
            )
            expected_global = torch.tensor(
                [global_block_begin + block for block in expected_local],
                dtype=torch.int32,
                device=sample_topk.device,
            )
            assert sample_topk_length[position].item() == visible
            torch.testing.assert_close(
                sample_topk[position, :visible], expected_global, rtol=0, atol=0
            )
            assert torch.all(sample_topk[position, visible:] == -1)

        row_begin = row_end
        global_block_begin += block_count

    compressed_rows = sum(length // cfg.compress_ratio for length in lengths)
    forbidden_shapes = {
        (compressed_rows, cfg.kv_dim),
        (compressed_rows, cfg.indexer_dim),
    }
    assert all(tuple(tensor.shape) not in forbidden_shapes for tensor in saved)

    torch.cuda.empty_cache()
    (output.float().sum() + kl_loss).backward()


def _kernel_dependencies_available() -> bool:
    try:
        from cudnn import DSA  # noqa: F401
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.dsa_kernel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not _kernel_dependencies_available(),
    reason="frozen FlashMLA and cudnn-frontend DSA packages required",
)
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_public_api_kernel_matches_legacy_full_gradients(ratio):
    _run_public_legacy_parity(ratio, backend="kernel")


@pytest.mark.dsa_kernel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not _kernel_dependencies_available(),
    reason="frozen cudnn-frontend DSA package required",
)
def test_kernel_indexer_topk_canonicalizes_equal_score_boundary():
    from magi_attention.experimental.dsa_v4.kernels import indexer_select_kernel

    sq, sk, heads, dim, topk = 128, 32, 64, 128, 8
    q_idx = torch.zeros(sq, 1, heads, dim, dtype=torch.bfloat16, device="cuda")
    k_idx = torch.zeros(sk, 1, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(sq, 1, heads, dtype=torch.bfloat16, device="cuda")

    selected = indexer_select_kernel(q_idx, k_idx, weights, topk, ratio=4)
    assert selected.shape == (1, sq, topk)
    assert selected.dtype == torch.int32
    for position in range(sq):
        valid = min((position + 1) // 4, topk)
        torch.testing.assert_close(
            selected[0, position, :valid],
            torch.arange(valid, dtype=torch.int32, device="cuda"),
            rtol=0,
            atol=0,
        )
        assert torch.all(selected[0, position, valid:] == -1)
