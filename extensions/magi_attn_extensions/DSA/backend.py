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

from dataclasses import dataclass
from typing import SupportsInt, cast

import torch
import torch.distributed as dist

from magi_attention.utils import nvtx

from .config import DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT, MagiDSAConfig
from .nvtx import dsa_cudnn_call_range, dsa_nvtx_range
from .packing import DsaDeviceIndexerMap, DsaDeviceRoutePlan
from .phase import dsa_phase


def _attention_mode(ratio: int) -> str:
    try:
        return {0: "w", 4: "csa", 128: "hca"}[ratio]
    except KeyError as error:
        raise ValueError(f"unsupported DSA ratio: {ratio}") from error


@dataclass(frozen=True)
class DsaIndexerSelection:
    """Grouped Indexer output in global compressed-block ID space."""

    global_ids: torch.Tensor
    lengths: torch.Tensor
    lse: torch.Tensor
    logical_score_calls: int
    logical_topk_calls: int


def _unpack_flashmla_sparse_result(
    result: object,
    *,
    require_indexer_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(result, (tuple, list)):
        raise RuntimeError("FlashMLA sparse forward must return a tuple or list")
    expected_outputs = 4 if require_indexer_lse else 3
    if len(result) != expected_outputs:
        expected_abi = (
            "(output, max_logits, sparse_lse, compressed_lse)"
            if require_indexer_lse
            else "(output, max_logits, sparse_lse)"
        )
        raise RuntimeError(
            "Magi-DSA received an incompatible FlashMLA sparse-forward ABI; "
            f"expected {expected_abi}"
        )
    if not all(isinstance(value, torch.Tensor) for value in result):
        raise RuntimeError("FlashMLA sparse forward returned a non-tensor output")
    output = result[0]
    lse = result[2]
    compressed_lse = result[3] if require_indexer_lse else None
    return output, lse, compressed_lse


def _current_cu_stream():
    from cuda.bindings import driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _require_explicit_cudnn_stream(stream: object, operation: str) -> object:
    """Reject the legacy default stream before entering affected cuDNN wrappers."""

    if int(cast(SupportsInt, stream)) == 0:
        raise RuntimeError(
            f"{operation} requires an explicit nonzero CUDA stream with the "
            "official unmodified cuDNN frontend"
        )
    return stream


def _require_current_cudnn_stream(stream: object, operation: str) -> object:
    """Require wrapper allocations and backend kernels to use the same stream."""

    current_stream = _current_cu_stream()
    if int(cast(SupportsInt, stream)) != int(cast(SupportsInt, current_stream)):
        raise RuntimeError(
            f"{operation} requires the CUDA stream argument to match the current "
            "PyTorch stream"
        )
    return stream


def _validate_release_backend(config: MagiDSAConfig, device: torch.device) -> None:
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("Magi-DSA v4 kernels require B300 SM103")
    try:
        config.validate_release_contract()
    except ValueError as error:
        raise ValueError(
            "the fixed FlashMLA/cuDNN backend supports only the official "
            "DeepSeek-V4-Pro dimensions"
        ) from error


def _finalize_backend_topk(
    seq_lens: torch.Tensor,
    backend_local_ids: torch.Tensor,
    sample_block_offsets: torch.Tensor,
    *,
    output_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map backend-native local Top-K IDs to canonical global block IDs.

    The fixed cuDNN wrapper owns selection and ordering, including exact-score
    ties. This function only applies per-row global offsets and output padding;
    it never inspects scores, reselects candidates, or reorders backend IDs.
    """

    if backend_local_ids.ndim != 2:
        raise ValueError("backend Top-K IDs must be a rank-2 tensor")
    rows, topk = backend_local_ids.shape
    if seq_lens.shape != (rows,) or sample_block_offsets.shape != (rows,):
        raise ValueError("backend Top-K metadata has an invalid shape")
    if backend_local_ids.dtype != torch.int32:
        raise TypeError("backend Top-K IDs must use int32")
    if seq_lens.dtype != torch.int32 or sample_block_offsets.dtype != torch.int32:
        raise TypeError("backend Top-K metadata must use int32")

    resolved_output_width = topk if output_width is None else int(output_width)
    enabled = backend_local_ids.is_cuda
    scope = "indexer::topk::backend_finalize"
    with dsa_nvtx_range(scope, enabled=enabled):
        if topk == 0:
            if resolved_output_width:
                raise ValueError(
                    "an empty backend Top-K cannot fill a non-empty output"
                )
            return backend_local_ids, torch.zeros_like(seq_lens)
        from .kernels.triton.indices import finalize_dsa_topk

        return finalize_dsa_topk(
            seq_lens,
            backend_local_ids,
            sample_block_offsets,
            resolved_output_width,
        )


@torch.no_grad()
def run_grouped_dsa_indexer(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    mapping: DsaDeviceIndexerMap,
    config: MagiDSAConfig,
) -> DsaIndexerSelection:
    """Invoke grouped THD score and top-k exactly once each for this rank."""

    _validate_release_backend(config, q_indexer.device)
    if (
        q_indexer.dtype != torch.bfloat16
        or k_indexer.dtype != torch.bfloat16
        or weights.dtype != torch.bfloat16
    ):
        raise TypeError("DSA Indexer Q/K/weights must use BF16")
    for name, tensor in (
        ("q_indexer", q_indexer),
        ("k_indexer", k_indexer),
        ("weights", weights),
    ):
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA")
    if q_indexer.shape != (
        mapping.seq_lens.numel(),
        config.indexer_heads,
        config.indexer_head_dim,
    ):
        raise ValueError("grouped Indexer Q has an invalid shape")
    if weights.shape != (q_indexer.shape[0], config.indexer_heads):
        raise ValueError("grouped Indexer weights have an invalid shape")
    if k_indexer.shape != (mapping.packed_k_rows, config.indexer_head_dim):
        raise ValueError("grouped Indexer K has an invalid shape")
    if mapping.backend_max_seqlen_k < mapping.logical_max_seqlen_k:
        raise ValueError("Indexer backend K width is smaller than its logical width")
    if (
        mapping.backend_max_seqlen_k
        and mapping.backend_max_seqlen_k % DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
    ):
        raise ValueError("Indexer backend K width violates score-row alignment")

    total_q = q_indexer.shape[0]
    if total_q == 0 or mapping.logical_max_seqlen_k == 0:
        with nvtx.add_nvtx_event("Magi_DSA/indexer"):
            with nvtx.add_nvtx_event("magi_dsa::indexer_score"):
                with dsa_nvtx_range("indexer::score::empty_output"):
                    lse = torch.full(
                        (total_q,),
                        float("-inf"),
                        dtype=torch.float32,
                        device=q_indexer.device,
                    )
            with nvtx.add_nvtx_event("magi_dsa::indexer_topk"):
                with dsa_nvtx_range("indexer::topk::empty_output"):
                    global_ids = torch.full(
                        (total_q, config.indexer_topk),
                        -1,
                        dtype=torch.int32,
                        device=q_indexer.device,
                    )
                    lengths = torch.zeros(
                        (total_q,), dtype=torch.int32, device=q_indexer.device
                    )
        return DsaIndexerSelection(
            global_ids=global_ids,
            lengths=lengths,
            lse=lse,
            logical_score_calls=1,
            logical_topk_calls=1,
        )

    from cudnn import DSA

    stream = _require_explicit_cudnn_stream(
        _current_cu_stream(),
        "grouped Indexer score/Top-K",
    )
    with nvtx.add_nvtx_event("Magi_DSA/indexer"):
        with nvtx.add_nvtx_event("magi_dsa::indexer_score"):
            with dsa_nvtx_range("indexer::score::cudnn_backend"):
                with dsa_cudnn_call_range("indexer_score"):
                    scores = DSA.indexer_forward_wrapper(
                        q_indexer,
                        k_indexer.unsqueeze(1),
                        weights,
                        ratio=4,
                        qhead_per_kv_head=config.indexer_heads,
                        sm_scale=config.indexer_head_dim**-0.5,
                        stream=stream,
                        cu_seqlens_q=mapping.q_cu_seqlens,
                        cu_seqlens_k=mapping.k_cu_seqlens,
                        max_seqlen_q=mapping.max_seqlen_q,
                        max_seqlen_k=mapping.backend_max_seqlen_k,
                        q_causal_offsets=mapping.q_causal_offsets,
                    )["scores"]
                expected_score_shape = (
                    total_q,
                    mapping.backend_max_seqlen_k,
                )
                if (
                    scores.shape != expected_score_shape
                    or scores.dtype != torch.float32
                ):
                    raise RuntimeError("cuDNN Indexer score output has an invalid ABI")
                if not scores.is_cuda or not scores.is_contiguous():
                    raise RuntimeError(
                        "cuDNN Indexer score output must be compact contiguous; "
                        "check the aligned backend max_seqlen_k"
                    )
            with dsa_nvtx_range("indexer::score::logsumexp"):
                from .kernels.triton.reductions import fused_dsa_row_logsumexp

                with dsa_nvtx_range("indexer::score::logsumexp::fused_triton"):
                    lse = fused_dsa_row_logsumexp(scores, mapping.seq_lens)

        with nvtx.add_nvtx_event("magi_dsa::indexer_topk"):
            kernel_topk = min(config.indexer_topk, scores.shape[1])
            with dsa_nvtx_range("indexer::topk::cudnn_backend"):
                with dsa_cudnn_call_range("indexer_topk"):
                    result = DSA.indexer_top_k_wrapper(
                        scores,
                        mapping.seq_lens,
                        kernel_topk,
                        next_n=1,
                        return_val=False,
                        stream=stream,
                    )
            if result["values"] is not None:
                raise RuntimeError("cuDNN Top-K returned values in IDs-only mode")
            global_ids, lengths = _finalize_backend_topk(
                mapping.seq_lens,
                result["indices"],
                mapping.q_sample_block_offsets,
                output_width=config.indexer_topk,
            )
    return DsaIndexerSelection(
        global_ids=global_ids,
        lengths=lengths,
        lse=lse,
        logical_score_calls=1,
        logical_topk_calls=1,
    )


def _run_dsa_sparse_attention_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    softmax_scale: float,
    indexer_topk: int,
    attention_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scope = f"attention::{attention_mode}::sparse_attention"
    if q.shape[0] == 0:
        with dsa_nvtx_range(f"{scope}::empty_forward"):
            output = q.new_empty(q.shape)
            lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
            compressed_lse = torch.empty(
                (q.shape[0], q.shape[1] if indexer_topk else 0),
                dtype=torch.float32,
                device=q.device,
            )
        return output, lse, compressed_lse

    import flash_mla

    with dsa_nvtx_range(f"{scope}::flashmla_forward"):
        if indexer_topk:
            result = flash_mla.flash_mla_sparse_fwd(
                q,
                kv.unsqueeze(1),
                indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=q.shape[-1],
                attn_sink=sink,
                topk_length=None,
                indexer_topk=indexer_topk,
            )
        else:
            result = flash_mla.flash_mla_sparse_fwd(
                q,
                kv.unsqueeze(1),
                indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=q.shape[-1],
                attn_sink=sink,
                topk_length=lengths,
            )
        output, lse, optional_compressed_lse = _unpack_flashmla_sparse_result(
            result,
            require_indexer_lse=bool(indexer_topk),
        )
        compressed_lse = (
            optional_compressed_lse
            if optional_compressed_lse is not None
            else torch.empty(
                (q.shape[0], 0),
                dtype=torch.float32,
                device=q.device,
            )
        )
    return output, lse, compressed_lse


def _run_dsa_sparse_attention_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    output: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    softmax_scale: float,
    use_sentinel_width: bool,
    attention_mode: str,
    stream: object,
) -> dict[str, torch.Tensor]:
    from cudnn import DSA

    stream = _require_current_cudnn_stream(
        stream,
        "sparse attention backward",
    )
    scope = f"attention::{attention_mode}::sparse_attention"
    with dsa_phase(f"attention::{attention_mode}::sparse_backward"):
        with dsa_cudnn_call_range("sparse_attention_backward"):
            with dsa_nvtx_range(f"{scope}::cudnn_backward"):
                return DSA.sparse_attention_backward_wrapper(
                    q,
                    kv,
                    output,
                    dout.contiguous(),
                    lse,
                    sink,
                    indices,
                    softmax_scale=softmax_scale,
                    topk_length=None if use_sentinel_width else lengths,
                    stream=stream,
                )


class _DsaSparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
        use_sentinel_width: bool,
        indexer_topk: int,
        attention_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, lse, compressed_lse = _run_dsa_sparse_attention_forward(
            q,
            kv,
            sink,
            indices,
            lengths,
            softmax_scale=softmax_scale,
            indexer_topk=indexer_topk,
            attention_mode=attention_mode,
        )
        ctx.save_for_backward(q, kv, output, lse, sink, indices, lengths)
        ctx.attention_mode = attention_mode
        ctx.softmax_scale = float(softmax_scale)
        ctx.use_sentinel_width = bool(use_sentinel_width)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(lse, compressed_lse)
        return output, lse, compressed_lse

    @staticmethod
    def backward(
        ctx,
        dout: torch.Tensor | None,
        dlse: torch.Tensor | None,
        dcompressed_lse: torch.Tensor | None,
    ):
        del dlse, dcompressed_lse
        q, kv, output, lse, sink, indices, lengths = ctx.saved_tensors
        if dout is None:
            return (None,) * 9
        if q.shape[0] == 0:
            return (
                torch.zeros_like(q),
                torch.zeros_like(kv),
                torch.zeros_like(sink),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        result = _run_dsa_sparse_attention_backward(
            q,
            kv,
            output,
            dout,
            lse,
            sink,
            indices,
            lengths,
            softmax_scale=ctx.softmax_scale,
            use_sentinel_width=ctx.use_sentinel_width,
            attention_mode=ctx.attention_mode,
            stream=_current_cu_stream(),
        )
        return (
            result["dq"],
            result["dkv"],
            result["d_sink"],
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dsa_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    config: MagiDSAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashMLA sparse forward paired with the fixed cuDNN DSA backward."""

    _validate_release_backend(config, q.device)
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("DSA attention Q/KV must use BF16")
    if (
        sink.dtype != torch.float32
        or indices.dtype != torch.int32
        or lengths.dtype != torch.int32
    ):
        raise TypeError("DSA sink/indices/lengths have invalid dtypes")
    if q.shape != (indices.shape[0], config.num_query_heads, config.head_dim):
        raise ValueError("DSA attention Q has an invalid shape")
    if kv.ndim != 2 or kv.shape[1] != config.head_dim:
        raise ValueError("DSA attention KV bank has an invalid shape")
    if indices.ndim != 2 or lengths.shape != (q.shape[0],):
        raise ValueError("DSA sparse indices or lengths have an invalid shape")
    use_sentinel_width = config.ratio == 4
    indexer_topk = config.indexer_topk if use_sentinel_width else 0
    attention_mode = _attention_mode(config.ratio)
    if (
        use_sentinel_width
        and indices.shape[1] != config.indexer_topk + config.window_size
    ):
        raise ValueError("CSA FlashMLA indices must use fixed compressed+window width")
    with dsa_nvtx_range(f"attention::{attention_mode}::sparse_attention::forward"):
        return _DsaSparseAttentionFunction.apply(
            q,
            kv,
            sink,
            indices,
            lengths,
            config.head_dim**-0.5,
            use_sentinel_width,
            indexer_topk,
            attention_mode,
        )


@dataclass(frozen=True)
class _DsaSelectedKlForwardState:
    loss: torch.Tensor
    unit_gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ready_event: torch.cuda.Event | None


def _run_dsa_selected_kl_forward(
    q_indexer: torch.Tensor,
    weights: torch.Tensor,
    selected_k_indexer: torch.Tensor,
    q_attention: torch.Tensor,
    k_attention: torch.Tensor,
    compressed_lse: torch.Tensor,
    selection: DsaIndexerSelection,
    indexer_indices: torch.Tensor,
    attention_indices: torch.Tensor,
    *,
    loss_coeff: float,
    attention_scale: float,
    indexer_scale: float,
) -> _DsaSelectedKlForwardState:
    total_q = q_indexer.shape[0]
    if total_q == 0 or selected_k_indexer.shape[0] == 0:
        return _DsaSelectedKlForwardState(
            torch.zeros((), dtype=torch.float32, device=q_indexer.device),
            None,
            None,
        )

    from cudnn import DSA

    stream = _require_explicit_cudnn_stream(
        _current_cu_stream(),
        "selected-KL recompute/Indexer backward",
    )
    topk_length = selection.lengths.unsqueeze(0)
    with dsa_phase("kl_recompute"):
        with dsa_nvtx_range("selected_kl::cudnn_indexer_recompute"):
            # The fixed selected-recompute wrapper has no sm_scale argument.
            # Fold only the Indexer dot-product scale into the already
            # head-averaged weights for this call; backward consumes the raw
            # model-side weights together with the same explicit sm_scale.
            recompute_weights = (weights.float() * float(indexer_scale)).to(
                weights.dtype
            )
            with dsa_cudnn_call_range("selected_indexer_recompute"):
                selected_predict = DSA.sparse_indexer_score_recompute_wrapper(
                    q_indexer.unsqueeze(0),
                    selected_k_indexer.unsqueeze(0),
                    recompute_weights.unsqueeze(0),
                    indexer_indices.unsqueeze(0),
                    qhead_per_kv_head=q_indexer.shape[1],
                    topk_length=topk_length,
                    stream=stream,
                )["predict"]
        with dsa_nvtx_range("selected_kl::cudnn_attention_recompute"):
            with dsa_cudnn_call_range("selected_attention_recompute"):
                target = DSA.sparse_attn_score_recompute_wrapper(
                    q_attention.unsqueeze(0),
                    k_attention.unsqueeze(0),
                    compressed_lse.unsqueeze(0),
                    attention_indices.unsqueeze(0),
                    softmax_scale=attention_scale,
                    qhead_per_kv_head=q_attention.shape[1],
                    topk_length=topk_length,
                    stream=stream,
                )["target"]
    from .kernels.triton.kl import fused_dsa_selected_kl_state

    with dsa_nvtx_range("selected_kl::loss_and_teacher_fused"):
        loss = fused_dsa_selected_kl_state(
            target[0].contiguous(),
            selected_predict[0].contiguous(),
            selection.lengths,
            float(loss_coeff),
        )

    if loss_coeff > 0:
        with dsa_phase("indexer_unit_gradient_precompute"):
            with dsa_nvtx_range("selected_kl::cudnn_indexer_backward"):
                with dsa_cudnn_call_range("indexer_backward"):
                    result = DSA.indexer_backward_wrapper(
                        q_indexer.unsqueeze(0),
                        weights.unsqueeze(0),
                        selected_k_indexer.unsqueeze(0),
                        target,
                        selected_predict,
                        indexer_indices.unsqueeze(0),
                        sm_scale=float(indexer_scale),
                        loss_coeff=float(loss_coeff),
                        grad_loss=1.0,
                        block_I=128,
                        topk_indices_global=False,
                        stream=stream,
                    )
        saved_grad_q = result["d_index_q"][0]
        saved_grad_weights = result["d_weights"][0]
        saved_grad_k = result["d_index_k"][0]
    else:
        saved_grad_q = torch.zeros_like(q_indexer)
        saved_grad_weights = torch.zeros_like(weights)
        saved_grad_k = torch.zeros_like(selected_k_indexer)
    with dsa_nvtx_range("selected_kl::unit_gradient_ready_event_record"):
        unit_gradient_ready_event = torch.cuda.Event()
        unit_gradient_ready_event.record(torch.cuda.current_stream(q_indexer.device))
    return _DsaSelectedKlForwardState(
        loss,
        (saved_grad_q, saved_grad_weights, saved_grad_k),
        unit_gradient_ready_event,
    )


class _DsaCsaAttentionKlFunction(torch.autograd.Function):
    """Pair CSA sparse attention and selected KL under one backward scheduler."""

    @staticmethod
    def forward(
        ctx,
        q_attention: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        attention_indices: torch.Tensor,
        attention_lengths: torch.Tensor,
        q_indexer: torch.Tensor,
        weights: torch.Tensor,
        compressed_ki_local: torch.Tensor,
        selected_k_indexer: torch.Tensor,
        teacher_k_attention: torch.Tensor,
        selection: DsaIndexerSelection,
        indexer_indices: torch.Tensor,
        teacher_attention_indices: torch.Tensor,
        compressed_ki_route: DsaDeviceRoutePlan,
        cp_group: dist.ProcessGroup | None,
        sparse_backward_stream: torch.cuda.Stream,
        loss_coeff: float,
        attention_scale: float,
        indexer_scale: float,
        indexer_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, sparse_lse, compressed_lse = _run_dsa_sparse_attention_forward(
            q_attention,
            kv,
            sink,
            attention_indices,
            attention_lengths,
            softmax_scale=attention_scale,
            indexer_topk=indexer_topk,
            attention_mode="csa",
        )
        _validate_selected_kl_inputs(
            q_indexer,
            compressed_lse,
            q_attention,
            selection,
        )
        kl_state = _run_dsa_selected_kl_forward(
            q_indexer,
            weights,
            selected_k_indexer,
            q_attention,
            teacher_k_attention,
            compressed_lse,
            selection,
            indexer_indices,
            teacher_attention_indices,
            loss_coeff=loss_coeff,
            attention_scale=attention_scale,
            indexer_scale=indexer_scale,
        )

        saved_tensors = [
            q_attention,
            kv,
            output,
            sparse_lse,
            sink,
            attention_indices,
            attention_lengths,
        ]
        ctx.kl_empty = kl_state.unit_gradients is None
        if kl_state.unit_gradients is None:
            ctx.indexer_shapes = (
                q_indexer.shape,
                weights.shape,
                selected_k_indexer.shape,
            )
            ctx.indexer_dtypes = (
                q_indexer.dtype,
                weights.dtype,
                selected_k_indexer.dtype,
            )
            ctx.indexer_device = q_indexer.device
        else:
            if kl_state.ready_event is None:
                raise RuntimeError(
                    "non-empty CSA selected KL state is missing its ready event"
                )
            ctx.unit_gradient_ready_event = kl_state.ready_event
            saved_tensors.extend(kl_state.unit_gradients)
        ctx.save_for_backward(*saved_tensors)
        ctx.compressed_ki_route = compressed_ki_route
        ctx.cp_group = cp_group
        ctx.sparse_backward_stream = sparse_backward_stream
        ctx.attention_scale = float(attention_scale)
        ctx.compressed_ki_local_shape = compressed_ki_local.shape
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(sparse_lse)
        return output, kl_state.loss, sparse_lse

    @staticmethod
    def backward(
        ctx,
        dout: torch.Tensor | None,
        dkl: torch.Tensor | None,
        dsparse_lse: torch.Tensor | None,
    ):
        del dsparse_lse
        (
            q_attention,
            kv,
            output,
            sparse_lse,
            sink,
            attention_indices,
            attention_lengths,
            *unit_gradients,
        ) = ctx.saved_tensors
        caller_stream = torch.cuda.current_stream(q_attention.device)
        reverse_transfer = None
        sparse_done = None
        sparse_launched = False

        grad_q_indexer = None
        grad_weights = None
        grad_compressed_ki_local = None
        sparse_result = None
        try:
            if dkl is not None:
                if ctx.kl_empty:
                    q_shape, weights_shape, consumer_ki_shape = ctx.indexer_shapes
                    q_dtype, weights_dtype, consumer_ki_dtype = ctx.indexer_dtypes
                    grad_q_indexer = torch.zeros(
                        q_shape, dtype=q_dtype, device=ctx.indexer_device
                    )
                    grad_weights = torch.zeros(
                        weights_shape,
                        dtype=weights_dtype,
                        device=ctx.indexer_device,
                    )
                    grad_consumer_ki = torch.zeros(
                        consumer_ki_shape,
                        dtype=consumer_ki_dtype,
                        device=ctx.indexer_device,
                    )
                else:
                    if len(unit_gradients) != 3:
                        raise RuntimeError(
                            "CSA selected KL saved an invalid unit-gradient state"
                        )
                    with dsa_nvtx_range(
                        "attention::csa::backward_overlap::"
                        "unit_gradient_ready_event_wait"
                    ):
                        caller_stream.wait_event(ctx.unit_gradient_ready_event)
                    with dsa_nvtx_range(
                        "attention::csa::backward_overlap::"
                        "unit_gradient_backward_lifetime"
                    ):
                        for backward_input in (*unit_gradients, dkl):
                            backward_input.record_stream(caller_stream)
                    from .kernels.triton.gradients import (
                        fused_dsa_scale_indexer_gradients,
                    )

                    with dsa_nvtx_range(
                        "attention::csa::backward_overlap::scale_indexer_gradients"
                    ):
                        (
                            grad_q_indexer,
                            grad_weights,
                            grad_consumer_ki,
                        ) = fused_dsa_scale_indexer_gradients(
                            unit_gradients[0],
                            unit_gradients[1],
                            unit_gradients[2],
                            dkl,
                        )

                from .comm import start_dsa_reverse_route

                with dsa_nvtx_range(
                    "attention::csa::backward_overlap::compressed_ki_reverse_start"
                ):
                    reverse_transfer = start_dsa_reverse_route(
                        grad_consumer_ki.contiguous(),
                        ctx.compressed_ki_route,
                        ctx.cp_group,
                        attention_mode="csa",
                    )

            if dout is not None:
                if q_attention.shape[0] == 0:
                    sparse_result = {
                        "dq": torch.zeros_like(q_attention),
                        "dkv": torch.zeros_like(kv),
                        "d_sink": torch.zeros_like(sink),
                    }
                else:
                    sparse_stream = ctx.sparse_backward_stream
                    sparse_ready = torch.cuda.Event()
                    sparse_ready.record(caller_stream)
                    sparse_done = torch.cuda.Event()
                    with dsa_nvtx_range(
                        "attention::csa::backward_overlap::sparse_backward_launch"
                    ):
                        with torch.cuda.stream(sparse_stream):
                            sparse_stream.wait_event(sparse_ready)
                            sparse_result = _run_dsa_sparse_attention_backward(
                                q_attention,
                                kv,
                                output,
                                dout,
                                sparse_lse,
                                sink,
                                attention_indices,
                                attention_lengths,
                                softmax_scale=ctx.attention_scale,
                                use_sentinel_width=True,
                                attention_mode="csa",
                                stream=_current_cu_stream(),
                            )
                            sparse_done.record(sparse_stream)
                    sparse_launched = True

            if reverse_transfer is not None:
                from .comm import finish_dsa_reverse_route

                with dsa_nvtx_range(
                    "attention::csa::backward_overlap::compressed_ki_reverse_finish"
                ):
                    grad_compressed_ki_local = finish_dsa_reverse_route(
                        reverse_transfer
                    )

            if sparse_launched:
                assert sparse_done is not None
                assert sparse_result is not None
                with dsa_nvtx_range(
                    "attention::csa::backward_overlap::sparse_backward_finish"
                ):
                    caller_stream.wait_event(sparse_done)
                    for sparse_gradient in sparse_result.values():
                        sparse_gradient.record_stream(caller_stream)
        except BaseException:
            if reverse_transfer is not None:
                reverse_transfer.wait()
            if sparse_launched:
                caller_stream.wait_stream(ctx.sparse_backward_stream)
            raise

        if (
            grad_compressed_ki_local is not None
            and grad_compressed_ki_local.shape != ctx.compressed_ki_local_shape
        ):
            raise RuntimeError("COMPRESSED_KI reverse returned an invalid local shape")

        return (
            None if sparse_result is None else sparse_result["dq"],
            None if sparse_result is None else sparse_result["dkv"],
            None if sparse_result is None else sparse_result["d_sink"],
            None,
            None,
            grad_q_indexer,
            grad_weights,
            grad_compressed_ki_local,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _DsaSelectedKlFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_indexer: torch.Tensor,
        weights: torch.Tensor,
        selected_k_indexer: torch.Tensor,
        q_attention: torch.Tensor,
        k_attention: torch.Tensor,
        compressed_lse: torch.Tensor,
        selection: DsaIndexerSelection,
        indexer_indices: torch.Tensor,
        attention_indices: torch.Tensor,
        loss_coeff: float,
        attention_scale: float,
        indexer_scale: float,
    ) -> torch.Tensor:
        state = _run_dsa_selected_kl_forward(
            q_indexer,
            weights,
            selected_k_indexer,
            q_attention,
            k_attention,
            compressed_lse,
            selection,
            indexer_indices,
            attention_indices,
            loss_coeff=loss_coeff,
            attention_scale=attention_scale,
            indexer_scale=indexer_scale,
        )
        ctx.empty = state.unit_gradients is None
        if state.unit_gradients is None:
            ctx.shapes = (
                q_indexer.shape,
                weights.shape,
                selected_k_indexer.shape,
            )
            ctx.device = q_indexer.device
            ctx.dtypes = (
                q_indexer.dtype,
                weights.dtype,
                selected_k_indexer.dtype,
            )
            return state.loss

        if state.ready_event is None:
            raise RuntimeError("non-empty selected KL state is missing its ready event")
        ctx.unit_gradient_ready_event = state.ready_event
        ctx.save_for_backward(*state.unit_gradients)
        ctx.set_materialize_grads(False)
        return state.loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor | None):
        if grad_loss is None:
            return (None,) * 12
        if ctx.empty:
            q_shape, w_shape, k_shape = ctx.shapes
            q_dtype, w_dtype, k_dtype = ctx.dtypes
            return (
                torch.zeros(q_shape, dtype=q_dtype, device=ctx.device),
                torch.zeros(w_shape, dtype=w_dtype, device=ctx.device),
                torch.zeros(k_shape, dtype=k_dtype, device=ctx.device),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        saved_grad_q, saved_grad_weights, saved_grad_k = ctx.saved_tensors
        backward_stream = torch.cuda.current_stream(saved_grad_q.device)
        with dsa_nvtx_range("selected_kl::unit_gradient_ready_event_wait"):
            backward_stream.wait_event(ctx.unit_gradient_ready_event)
        with dsa_nvtx_range("selected_kl::unit_gradient_backward_lifetime"):
            for backward_input in (
                saved_grad_q,
                saved_grad_weights,
                saved_grad_k,
                grad_loss,
            ):
                backward_input.record_stream(backward_stream)
        from .kernels.triton.gradients import fused_dsa_scale_indexer_gradients

        grad_q, grad_weights, grad_k = fused_dsa_scale_indexer_gradients(
            saved_grad_q,
            saved_grad_weights,
            saved_grad_k,
            grad_loss,
        )
        return (
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _validate_selected_kl_inputs(
    q_indexer: torch.Tensor,
    compressed_lse: torch.Tensor,
    q_attention: torch.Tensor,
    selection: DsaIndexerSelection,
) -> None:
    if selection.lengths.shape != (q_indexer.shape[0],):
        raise ValueError("Top-K lengths have an invalid shape")
    if selection.lse.shape != (q_indexer.shape[0],):
        raise ValueError("Indexer LSE has an invalid shape")
    if compressed_lse.shape != q_attention.shape[:2]:
        raise ValueError("compressed-prefix attention LSE has an invalid shape")
    if selection.lse.requires_grad or compressed_lse.requires_grad:
        raise ValueError("KL LSE inputs must be detached saved state")


def dsa_csa_attention_kl(
    q_attention: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    attention_indices: torch.Tensor,
    attention_lengths: torch.Tensor,
    q_indexer: torch.Tensor,
    weights: torch.Tensor,
    compressed_ki_local: torch.Tensor,
    selected_k_indexer: torch.Tensor,
    teacher_k_attention: torch.Tensor,
    selection: DsaIndexerSelection,
    indexer_indices: torch.Tensor,
    teacher_attention_indices: torch.Tensor,
    compressed_ki_route: DsaDeviceRoutePlan,
    cp_group: dist.ProcessGroup | None,
    sparse_backward_stream: torch.cuda.Stream,
    *,
    loss_coeff: float,
    config: MagiDSAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CSA forward and overlap KI reverse with sparse attention backward."""

    _validate_release_backend(config, q_attention.device)
    if config.ratio != 4:
        raise ValueError("the combined CSA scheduler requires ratio=4")
    if q_attention.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("CSA attention Q/KV must use BF16")
    if (
        sink.dtype != torch.float32
        or attention_indices.dtype != torch.int32
        or attention_lengths.dtype != torch.int32
    ):
        raise TypeError("CSA sink/indices/lengths have invalid dtypes")
    if q_attention.shape != (
        attention_indices.shape[0],
        config.num_query_heads,
        config.head_dim,
    ):
        raise ValueError("CSA attention Q has an invalid shape")
    if kv.ndim != 2 or kv.shape[1] != config.head_dim:
        raise ValueError("CSA attention KV bank has an invalid shape")
    if attention_indices.ndim != 2 or attention_lengths.shape != (
        q_attention.shape[0],
    ):
        raise ValueError("CSA sparse indices or lengths have an invalid shape")
    if attention_indices.shape[1] != config.indexer_topk + config.window_size:
        raise ValueError("CSA FlashMLA indices must use fixed compressed+window width")
    if compressed_ki_route.name != "COMPRESSED_KI":
        raise ValueError("the combined CSA scheduler requires COMPRESSED_KI metadata")
    if compressed_ki_local.shape != (
        compressed_ki_route.producer_row_count,
        config.indexer_head_dim,
    ):
        raise ValueError("owner-local compressed KI has an invalid shape")
    if selected_k_indexer.shape != (
        compressed_ki_route.consumer_row_count,
        config.indexer_head_dim,
    ):
        raise ValueError("consumer compressed KI has an invalid shape")
    if indexer_indices.shape != selection.global_ids.shape:
        raise ValueError("Indexer teacher indices have an invalid shape")
    if teacher_attention_indices.shape != selection.global_ids.shape:
        raise ValueError("attention teacher indices have an invalid shape")
    if indexer_indices.dtype != torch.int32:
        raise TypeError("Indexer teacher indices must use int32")
    if teacher_attention_indices.dtype != torch.int32:
        raise TypeError("attention teacher indices must use int32")
    if q_indexer.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
        raise TypeError("CSA Indexer Q/weights must use BF16")

    with dsa_nvtx_range("attention::csa::combined_forward"):
        return _DsaCsaAttentionKlFunction.apply(
            q_attention,
            kv,
            sink,
            attention_indices,
            attention_lengths,
            q_indexer,
            weights,
            compressed_ki_local,
            selected_k_indexer,
            teacher_k_attention,
            selection,
            indexer_indices,
            teacher_attention_indices,
            compressed_ki_route,
            cp_group,
            sparse_backward_stream,
            float(loss_coeff),
            config.head_dim**-0.5,
            config.indexer_head_dim**-0.5,
            config.indexer_topk,
        )


def dsa_selected_kl(
    q_indexer: torch.Tensor,
    weights: torch.Tensor,
    selected_k_indexer: torch.Tensor,
    q_attention: torch.Tensor,
    k_attention: torch.Tensor,
    compressed_lse: torch.Tensor,
    selection: DsaIndexerSelection,
    indexer_indices: torch.Tensor,
    attention_indices: torch.Tensor,
    *,
    loss_coeff: float,
    config: MagiDSAConfig,
) -> torch.Tensor:
    """Selected-only teacher KL paired with sparse Indexer backward."""

    _validate_release_backend(config, q_indexer.device)
    _validate_selected_kl_inputs(q_indexer, compressed_lse, q_attention, selection)
    with dsa_nvtx_range("selected_kl::forward"):
        return _DsaSelectedKlFunction.apply(
            q_indexer,
            weights,
            selected_k_indexer,
            q_attention,
            k_attention,
            compressed_lse,
            selection,
            indexer_indices,
            attention_indices,
            float(loss_coeff),
            config.head_dim**-0.5,
            config.indexer_head_dim**-0.5,
        )


__all__ = [
    "DsaIndexerSelection",
    "dsa_csa_attention_kl",
    "dsa_selected_kl",
    "dsa_sparse_attention",
    "run_grouped_dsa_indexer",
]
