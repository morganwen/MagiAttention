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

"""Transport-only CP=2 tests for Magi_DSA step 3.

No attention kernel runs here.  These tests isolate the static reference maps,
the four typed GroupCast routes and their symmetric FP32 GroupReduce routes.
"""

import inspect

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention.functional.dsa_comm as dsa_comm
from magi_attention.api import DsaPackedMeta, MagiDSAInput, calc_dsa
from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr
from magi_attention.comm.primitive.grpcoll.utils import (
    sanity_check_for_group_cast_meta_args_per_rank,
)
from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr
from magi_attention.experimental.dsa_v4 import MagiDSAV4Config
from magi_attention.functional.dsa_comm import (
    DsaPayloadKind,
    DsaTypedPayload,
    build_dsa_comm_plan,
    materialize_dsa_comm_plan,
    start_dsa_group_cast,
    start_dsa_group_reduce,
)
from magi_attention.meta.collection.dsa_meta import DsaFragmentSpec
from magi_attention.meta.solver.dsa_dispatch import build_dsa_dispatch_plan
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.utils import switch_envvar_context


def _noncontiguous_cp2_plan():
    return build_dsa_dispatch_plan(
        [1024],
        [
            [DsaFragmentSpec(0, 0, 256), DsaFragmentSpec(0, 768, 1024)],
            [DsaFragmentSpec(0, 256, 768)],
        ],
        compress_ratio=4,
        policy="balanced",
    )


def _nonadjacent_cp4_plan():
    return build_dsa_dispatch_plan(
        [512],
        [
            [DsaFragmentSpec(0, 0, 128)],
            [DsaFragmentSpec(0, 384, 512)],
            [DsaFragmentSpec(0, 128, 256)],
            [DsaFragmentSpec(0, 256, 384)],
        ],
        compress_ratio=4,
        policy="nonadjacent",
    )


class TestDsaCommMetadata:
    def test_four_payloads_have_independent_collective_args_and_unique_rows(self):
        plan = _noncontiguous_cp2_plan()
        comm_plan = build_dsa_comm_plan(plan, 0, object())  # type: ignore[arg-type]

        assert tuple(meta.kind for meta in comm_plan) == tuple(DsaPayloadKind)
        assert len({id(meta.collective_arg) for meta in comm_plan}) == 4
        for meta in comm_plan:
            assert len(meta.send_row_ids) == len(set(meta.send_row_ids))
            assert len(meta.receive_row_ids) == len(set(meta.receive_row_ids))
            assert sum(meta.collective_arg.input_split_size_list) == meta.send_row_count
            assert (
                sum(meta.collective_arg.output_split_size_list)
                == meta.receive_row_count
            )

        assert comm_plan.window_kv.send_row_count == 127
        assert comm_plan.overlap_x.send_row_count == 4
        assert comm_plan.compressed_kv.send_row_ids == tuple(range(64)) + tuple(
            range(192, 256)
        )
        assert comm_plan.compressed_ki.send_row_ids == (
            comm_plan.compressed_kv.send_row_ids
        )

    def test_nonadjacent_routes_and_zero_length_metadata_are_symmetric(self):
        plan = _nonadjacent_cp4_plan()
        assert any(
            abs(route.destination_rank - route.source_rank) > 1
            for route in plan.window_transfers
        )
        plans = [
            build_dsa_comm_plan(plan, rank, object())  # type: ignore[arg-type]
            for rank in range(plan.cp_size)
        ]

        for kind in DsaPayloadKind:
            metas = [comm_plan.for_kind(kind) for comm_plan in plans]
            sanity_check_for_group_cast_meta_args_per_rank(
                input_split_size_list_per_rank=[
                    meta.collective_arg.input_split_size_list for meta in metas
                ],
                output_split_size_list_per_rank=[
                    meta.collective_arg.output_split_size_list for meta in metas
                ],
                dst_indices_list_per_rank=[
                    meta.collective_arg.dst_indices_list for meta in metas
                ],
                src_index_list_per_rank=[
                    meta.collective_arg.src_index_list for meta in metas
                ],
                world_size=plan.cp_size,
                check_nccl_send_recv=True,
            )

        ratio0 = build_dsa_dispatch_plan(
            [128],
            [[DsaFragmentSpec(0, 0, 128)], []],
            compress_ratio=0,
            policy="empty-rank",
        )
        for rank in range(2):
            comm_plan = build_dsa_comm_plan(
                ratio0, rank, object()  # type: ignore[arg-type]
            )
            for meta in comm_plan:
                assert sum(meta.collective_arg.input_split_size_list) == 0
                assert sum(meta.collective_arg.output_split_size_list) == 0
                assert len(meta.collective_arg.input_split_size_list) >= 1

    def test_non_indexer_layers_keep_compressed_ki_route_empty(self):
        ratio128 = build_dsa_dispatch_plan(
            [256],
            [
                [DsaFragmentSpec(0, 0, 128)],
                [DsaFragmentSpec(0, 128, 256)],
            ],
            compress_ratio=128,
            policy="sequential",
        )
        for rank in range(2):
            comm_plan = build_dsa_comm_plan(
                ratio128, rank, object()  # type: ignore[arg-type]
            )
            assert comm_plan.compressed_kv.local_row_count == 1
            assert comm_plan.compressed_kv.receive_row_count == 1
            assert comm_plan.compressed_ki.local_row_count == 0
            assert comm_plan.compressed_ki.receive_row_count == 0

    def test_production_transport_only_calls_group_collective_primitives(self):
        source = inspect.getsource(dsa_comm)
        forbidden = (
            "dist.all_gather(",
            "dist.all_reduce(",
            "dist.all_to_all(",
            "all2all_v(",
            "batch_isend_irecv(",
        )
        assert all(token not in source for token in forbidden)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA buffers")
    def test_native_handle_and_buffer_contract_is_payload_private(self, monkeypatch):
        class ImmediateWork:
            def __init__(self, result):
                self.result = result

            def wait_post_process(self, _buffer):
                return self.result

        cast_calls = []
        reduce_calls = []

        def fake_group_cast(**kwargs):
            handle = object()
            handle_dict = kwargs["native_grpcoll_handle_dict"]
            handle_dict["group_cast"] = handle
            handle_dict["group_reduce"] = handle
            cast_calls.append(kwargs)
            return ImmediateWork(kwargs["output"])

        def fake_group_reduce(**kwargs):
            handle_dict = kwargs["native_grpcoll_handle_dict"]
            assert handle_dict["group_reduce"] is handle_dict["group_cast"]
            reduce_calls.append(kwargs)
            return ImmediateWork(kwargs["output"])

        monkeypatch.setattr(dsa_comm, "group_cast", fake_group_cast)
        monkeypatch.setattr(dsa_comm, "group_reduce", fake_group_reduce)

        comm_plan = build_dsa_comm_plan(
            _noncontiguous_cp2_plan(), 0, object()  # type: ignore[arg-type]
        )
        widths = {
            DsaPayloadKind.WINDOW_KV: 512,
            DsaPayloadKind.OVERLAP_X: 256,
            DsaPayloadKind.COMPRESSED_KV: 512,
            DsaPayloadKind.COMPRESSED_KI: 128,
        }
        cast_works = []
        logical_receive_widths = []
        device_plan = materialize_dsa_comm_plan(comm_plan, torch.device("cuda"))
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=True):
            for meta, device_map in zip(comm_plan, device_plan):
                local = DsaTypedPayload(
                    meta.kind,
                    torch.zeros(
                        (meta.local_row_count, widths[meta.kind]),
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                )
                cast_work = start_dsa_group_cast(
                    local,
                    meta,
                    device_map=device_map,
                    async_op=True,
                )
                remote = cast_work.wait()
                logical_receive_widths.append(remote.tensor.size(1))
                reduce_work = start_dsa_group_reduce(
                    DsaTypedPayload(meta.kind, remote.tensor.float()),
                    DsaTypedPayload(
                        meta.kind,
                        torch.ones(
                            (meta.local_row_count, widths[meta.kind]),
                            dtype=torch.float32,
                            device="cuda",
                        ),
                    ),
                    cast_work,
                    async_op=True,
                )
                reduce_work.wait()
                cast_works.append(cast_work)

        expected_names = [kind.value for kind in DsaPayloadKind]
        assert [call["buffer_name"] for call in cast_calls] == expected_names
        assert [call["buffer_name"] for call in reduce_calls] == expected_names
        assert [call["input"].size(1) for call in cast_calls] == [512, 256, 512, 256]
        assert logical_receive_widths == [512, 256, 512, 128]
        assert len({id(work.native_handle_dict) for work in cast_works}) == 4
        assert all(
            work.device_map is device_map
            for work, device_map in zip(cast_works, device_plan)
        )
        for meta, cast_call, reduce_call, cast_work in zip(
            comm_plan, cast_calls, reduce_calls, cast_works
        ):
            assert (
                cast_call["native_grpcoll_handle_dict"]
                is reduce_call["native_grpcoll_handle_dict"]
                is cast_work.native_handle_dict
            )
            assert cast_call["input_split_sizes"] == (
                meta.collective_arg.input_split_size_list
            )
            assert reduce_call["output_split_sizes"] == (
                meta.collective_arg.input_split_size_list
            )
            assert reduce_call["input"].dtype == torch.float32
            assert reduce_call["output"].dtype == torch.float32


class TestDsaCommTransport(DistTestBase):
    @property
    def timeout(self):
        # cuDNN's first sparse-backward call compiles CUTLASS DSL kernels.
        # Every communication-only and warm-cache test keeps the default 60s.
        if self._testMethodName == "test_cp2_full_kernel_backward_matches_cp1":
            return 600
        return 60

    @property
    def world_size(self) -> int:
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @staticmethod
    def _width(kind: DsaPayloadKind) -> int:
        return {
            DsaPayloadKind.WINDOW_KV: 512,
            DsaPayloadKind.OVERLAP_X: 256,
            DsaPayloadKind.COMPRESSED_KV: 512,
            DsaPayloadKind.COMPRESSED_KI: 128,
        }[kind]

    @staticmethod
    def _row_value(kind: DsaPayloadKind, row_id: int) -> int:
        return list(DsaPayloadKind).index(kind) * 32 + row_id % 31

    def _local_payload(self, meta):
        values = torch.tensor(
            [self._row_value(meta.kind, row_id) for row_id in meta.local_row_ids],
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        tensor = values[:, None].expand(-1, self._width(meta.kind)).contiguous()
        return DsaTypedPayload(meta.kind, tensor)

    def _run_transport(self, *, use_native: bool, empty_routes: bool = False):
        if empty_routes:
            dispatch_plan = build_dsa_dispatch_plan(
                [128],
                [[DsaFragmentSpec(0, 0, 128)], []],
                compress_ratio=0,
                policy="empty-rank",
            )
        else:
            dispatch_plan = _noncontiguous_cp2_plan()
        comm_plan = build_dsa_comm_plan(dispatch_plan, self.rank, self.process_group)
        device_plan = materialize_dsa_comm_plan(comm_plan, torch.cuda.current_device())

        cast_works = []
        for meta, device_map in zip(comm_plan, device_plan):
            cast_works.append(
                start_dsa_group_cast(
                    self._local_payload(meta),
                    meta,
                    device_map=device_map,
                    async_op=True,
                )
            )
        assert len({id(work.buffers) for work in cast_works}) == 4
        assert len({id(work.native_handle_dict) for work in cast_works}) == 4
        assert all(
            work.device_map is device_map
            for work, device_map in zip(cast_works, device_plan)
        )

        received = [work.wait() for work in cast_works]
        for meta, payload in zip(comm_plan, received):
            expected_values = torch.tensor(
                [self._row_value(meta.kind, row_id) for row_id in meta.receive_row_ids],
                dtype=torch.bfloat16,
                device=payload.tensor.device,
            )
            expected = expected_values[:, None].expand_as(payload.tensor)
            torch.testing.assert_close(payload.tensor, expected, rtol=0, atol=0)

        reduce_works = []
        remote_value = float(self.rank + 2)
        for meta, payload, cast_work in zip(comm_plan, received, cast_works):
            remote_gradient = DsaTypedPayload(
                meta.kind,
                torch.full_like(payload.tensor, remote_value, dtype=torch.float32),
            )
            local_gradient = DsaTypedPayload(
                meta.kind,
                torch.ones(
                    (meta.local_row_count, self._width(meta.kind)),
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
            reduce_works.append(
                start_dsa_group_reduce(
                    remote_gradient,
                    local_gradient,
                    cast_work,
                    async_op=True,
                )
            )

        peer = 1 - self.rank
        for meta, reduce_work in zip(comm_plan, reduce_works):
            owner_gradient = reduce_work.wait().tensor
            expected = torch.ones_like(owner_gradient)
            if meta.send_row_indices:
                indices = torch.tensor(
                    meta.send_row_indices,
                    dtype=torch.int64,
                    device=owner_gradient.device,
                )
                expected.index_fill_(0, indices, float(1 + peer + 2))
            torch.testing.assert_close(owner_gradient, expected, rtol=0, atol=0)

        if use_native:
            grpcoll_buffer_mgr.release_group(self.process_group)

    @staticmethod
    def _forward_config(ratio: int, backend: str) -> MagiDSAV4Config:
        return MagiDSAV4Config(
            compress_ratio=ratio,
            hidden_size=64,
            q_lora_rank=64,
            softmax_scale=512**-0.5,
            backend=backend,
        )

    @staticmethod
    def _global_forward_input(config: MagiDSAV4Config) -> MagiDSAInput:
        lengths = (256, 256)
        total = sum(lengths)
        torch.manual_seed(20260710 + config.compress_ratio)

        def make(*shape):
            return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

        return MagiDSAInput(
            x=make(total, config.hidden_size),
            qr=make(total, config.q_lora_rank),
            q=make(total, config.num_heads, config.kv_dim),
            latent_kv=make(total, config.kv_dim),
            sink=torch.randn(config.num_heads, dtype=torch.float32, device="cuda"),
            packed_meta=DsaPackedMeta(torch.tensor([0, 256, 512], dtype=torch.int32)),
        )

    @staticmethod
    def _local_forward_input(global_input, dispatch_plan, rank):
        offsets = (0, *torch.tensor(dispatch_plan.sample_lengths).cumsum(0).tolist())
        row_ids = [
            offsets[fragment.sample_id] + position
            for fragment in dispatch_plan.ranks[rank].fragments
            for position in range(fragment.q_begin, fragment.q_end)
        ]
        index = torch.tensor(row_ids, dtype=torch.int64, device="cuda")
        return (
            MagiDSAInput(
                x=global_input.x.index_select(0, index).contiguous(),
                qr=global_input.qr.index_select(0, index).contiguous(),
                q=global_input.q.index_select(0, index).contiguous(),
                latent_kv=global_input.latent_kv.index_select(0, index).contiguous(),
                sink=global_input.sink.clone(),
                packed_meta=global_input.packed_meta,
            ),
            index,
        )

    def _run_full_forward(self, *, ratio: int, policy: str, backend: str):
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        config = self._forward_config(ratio, backend)
        torch.manual_seed(1103 + ratio)
        runtime = MagiDSARuntimeMgr(
            config,
            cp_group=self.process_group,
            dispatch_policy=policy,
        ).cuda()
        torch.manual_seed(1103 + ratio)
        reference_runtime = MagiDSARuntimeMgr(config).cuda()
        reference_runtime.load_state_dict(runtime.state_dict())
        runtime.train()
        reference_runtime.train()

        global_input = self._global_forward_input(config)
        dispatch_plan = runtime.get_dispatch_plan(global_input.packed_meta)
        local_input, local_rows = self._local_forward_input(
            global_input, dispatch_plan, self.rank
        )

        local_output, local_kl = calc_dsa(local_input, runtime)
        reference_output, reference_kl = calc_dsa(global_input, reference_runtime)
        expected_output = reference_output.index_select(0, local_rows)
        torch.testing.assert_close(
            local_output,
            expected_output,
            rtol=2e-2,
            atol=2e-3,
        )
        assert local_output.shape == (
            dispatch_plan.ranks[self.rank].token_count,
            config.num_heads,
            config.kv_dim,
        )
        assert local_kl.shape == torch.Size([])
        assert local_kl.dtype == torch.float32

        global_kl = local_kl.detach().clone()
        dist.all_reduce(global_kl, group=self.process_group)
        torch.testing.assert_close(global_kl, reference_kl, rtol=2e-2, atol=2e-5)
        assert runtime.forward_plan_cache_size == 1
        assert dispatch_plan.policy == policy

        del (
            local_output,
            reference_output,
            expected_output,
            local_kl,
            reference_kl,
            local_input,
            global_input,
            runtime,
            reference_runtime,
        )
        torch.cuda.empty_cache()

    def _run_full_backward(self, *, ratio: int, backend: str):
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        config = self._forward_config(ratio, backend)
        torch.manual_seed(1403 + ratio)
        runtime = MagiDSARuntimeMgr(
            config,
            cp_group=self.process_group,
            dispatch_policy="balanced",
        ).cuda()
        torch.manual_seed(1403 + ratio)
        reference_runtime = MagiDSARuntimeMgr(config).cuda()
        reference_runtime.load_state_dict(runtime.state_dict())
        runtime.train()
        reference_runtime.train()

        global_input = self._global_forward_input(config)
        for tensor in (
            global_input.x,
            global_input.qr,
            global_input.q,
            global_input.latent_kv,
            global_input.sink,
        ):
            tensor.requires_grad_(True)
        dispatch_plan = runtime.get_dispatch_plan(global_input.packed_meta)
        local_input, local_rows = self._local_forward_input(
            global_input, dispatch_plan, self.rank
        )
        local_input = MagiDSAInput(
            x=local_input.x.detach().requires_grad_(True),
            qr=local_input.qr.detach().requires_grad_(True),
            q=local_input.q.detach().requires_grad_(True),
            latent_kv=local_input.latent_kv.detach().requires_grad_(True),
            sink=local_input.sink.detach().requires_grad_(True),
            packed_meta=local_input.packed_meta,
        )
        torch.manual_seed(8675309 + ratio)
        output_grad = torch.randn(
            global_input.q.shape,
            dtype=torch.float32,
            device=global_input.q.device,
        ) / global_input.q.size(0)

        saved = []

        def pack(tensor):
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            local_output, local_kl = calc_dsa(local_input, runtime)
        assert len(saved) == 9
        for actual, expected in zip(
            saved[:5],
            (
                local_input.x,
                local_input.qr,
                local_input.q,
                local_input.latent_kv,
                local_input.sink,
            ),
        ):
            assert actual.data_ptr() == expected.data_ptr()
        assert saved[5].data_ptr() == local_output.data_ptr()
        assert saved[6].dtype == torch.float32
        assert saved[7].dtype == torch.int32
        assert saved[8].dtype == torch.int32 and saved[8].numel() == 1
        compressed_rows = sum(
            length // max(ratio, 1) for length in dispatch_plan.sample_lengths
        )
        forbidden_shapes = {
            (compressed_rows, config.kv_dim),
            (compressed_rows, config.indexer_dim),
        }
        assert all(tuple(tensor.shape) not in forbidden_shapes for tensor in saved)
        local_loss = (
            local_output.float() * output_grad.index_select(0, local_rows)
        ).sum() + local_kl
        # No invocation-owned forward work or temporary communication buffer
        # may be needed by backward.
        torch.cuda.empty_cache()
        local_loss.backward()

        reference_output, reference_kl = calc_dsa(global_input, reference_runtime)
        (reference_output.float().mul(output_grad).sum() + reference_kl).backward()

        input_pairs = (
            ("x", local_input.x, global_input.x),
            ("qr", local_input.qr, global_input.qr),
            ("q", local_input.q, global_input.q),
            ("latent_kv", local_input.latent_kv, global_input.latent_kv),
        )
        for name, local_tensor, global_tensor in input_pairs:
            expected = (
                None
                if global_tensor.grad is None
                else global_tensor.grad.index_select(0, local_rows)
            )
            if expected is None:
                assert local_tensor.grad is None, name
            else:
                assert local_tensor.grad is not None, name
                torch.testing.assert_close(
                    local_tensor.grad,
                    expected,
                    rtol=3e-2,
                    atol=3e-3,
                )
        torch.testing.assert_close(
            local_input.sink.grad,
            global_input.sink.grad,
            rtol=3e-2,
            atol=3e-3,
        )
        assert getattr(local_input.sink, "_magi_dsa_cp_reduced", False)
        for (name, parameter), (reference_name, reference_parameter) in zip(
            runtime.dsa_module.named_parameters(),
            reference_runtime.dsa_module.named_parameters(),
        ):
            assert name == reference_name
            assert parameter.grad is not None, name
            assert reference_parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter.grad,
                rtol=3e-2,
                atol=3e-3,
                msg=f"ratio={ratio} parameter={name}",
            )
            assert getattr(parameter, "_magi_dsa_cp_reduced", False), name
        if ratio == 4:
            assert local_input.qr.grad is None

        del (
            local_output,
            reference_output,
            local_loss,
            local_kl,
            reference_kl,
            local_input,
            global_input,
            runtime,
            reference_runtime,
        )
        torch.cuda.empty_cache()

    @staticmethod
    def _kernel_dependencies_available() -> bool:
        try:
            from cudnn import DSA  # noqa: F401
            from flash_mla import flash_mla_sparse_fwd  # noqa: F401
        except ImportError:
            return False
        return True

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_a2av_reference_pack_forward_and_reverse(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            self._run_transport(use_native=False)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_zero_length_routes_enter_all_four_collectives(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            self._run_transport(use_native=False, empty_routes=True)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_native_grpcoll_matches_reference_semantics(self):
        try:
            grpcoll_buffer_mgr.initialize(
                group=self.process_group,
                config=GrpCollConfig(
                    num_sms=20,
                    nvl_chunk_size=8,
                    nvl_buffer_size=256,
                    rdma_chunk_size=8,
                    rdma_buffer_size=256,
                    num_nvl_bytes=64 * 1024 * 1024,
                    num_rdma_bytes=0,
                ),
            )
        except (AssertionError, ImportError) as error:
            self.skipTest(f"native grpcoll extension is unavailable: {error}")
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=True):
            self._run_transport(use_native=True)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_cp2_full_reference_forward_matches_cp1(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for policy in ("sequential", "balanced"):
                for ratio in (0, 4, 128):
                    self._run_full_forward(
                        ratio=ratio,
                        policy=policy,
                        backend="reference",
                    )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_cp2_full_kernel_forward_matches_cp1(self):
        if not self._kernel_dependencies_available():
            self.skipTest("FlashMLA and cudnn-frontend DSA packages are required")
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for ratio in (0, 4, 128):
                self._run_full_forward(
                    ratio=ratio,
                    policy="balanced",
                    backend="kernel",
                )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_cp2_full_reference_backward_matches_cp1(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for ratio in (0, 4, 128):
                self._run_full_backward(ratio=ratio, backend="reference")

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_cp2_full_kernel_backward_matches_cp1(self):
        if not self._kernel_dependencies_available():
            self.skipTest("FlashMLA and cudnn-frontend DSA packages are required")
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for ratio in (0, 4, 128):
                self._run_full_backward(ratio=ratio, backend="kernel")


if __name__ == "__main__":
    run_tests()
