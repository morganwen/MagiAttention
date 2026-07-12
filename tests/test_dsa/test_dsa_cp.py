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

"""CP=8 transport, numerical, overlap and concurrency tests for Magi_DSA."""

import inspect
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention.functional.dsa_comm as dsa_comm
from magi_attention.api import (
    DsaOverlapConfig,
    DsaPackedMeta,
    MagiDSAConfig,
    MagiDSAInput,
    MagiDSARuntimeMgr,
    calc_dsa,
)
from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr
from magi_attention.comm.primitive.grpcoll.utils import (
    sanity_check_for_group_cast_meta_args_per_rank,
)
from magi_attention.functional.dsa_comm import (
    DsaPayloadKind,
    DsaTypedPayload,
    DsaWorkTracker,
    build_dsa_comm_plan,
    materialize_dsa_comm_plan,
    start_dsa_group_cast,
    start_dsa_group_reduce,
)
from magi_attention.meta.collection.dsa_meta import DsaFragmentSpec
from magi_attention.meta.solver.dsa_dispatch import build_dsa_dispatch_plan
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.utils import switch_envvar_context

_CP8_NATIVE_NVL_BYTES = 1 << 30


def _noncontiguous_cp8_plan():
    return build_dsa_dispatch_plan(
        [2048],
        [
            [DsaFragmentSpec(0, 0, 128), DsaFragmentSpec(0, 1920, 2048)],
            [DsaFragmentSpec(0, 256, 384), DsaFragmentSpec(0, 1664, 1792)],
            [DsaFragmentSpec(0, 512, 640), DsaFragmentSpec(0, 1408, 1536)],
            [DsaFragmentSpec(0, 768, 896), DsaFragmentSpec(0, 1152, 1280)],
            [DsaFragmentSpec(0, 128, 256), DsaFragmentSpec(0, 1792, 1920)],
            [DsaFragmentSpec(0, 384, 512), DsaFragmentSpec(0, 1536, 1664)],
            [DsaFragmentSpec(0, 640, 768), DsaFragmentSpec(0, 1280, 1408)],
            [DsaFragmentSpec(0, 896, 1152)],
        ],
        compress_ratio=4,
        policy="balanced",
    )


class TestDsaCommMetadata:
    def test_cp8_native_buffer_covers_7168_wide_fp32_reverse_route(self):
        required = GrpCollConfig.get_min_num_bytes_intranode(
            num_sms=20,
            num_ranks=8,
            hidden_size=7168,
            nvl_buffer_size=256,
            dtype=torch.float32,
        )
        assert required == 587_286_144
        assert _CP8_NATIVE_NVL_BYTES >= required * 6 // 5

    @staticmethod
    def _fake_work(events, name, error=None):
        # DsaWorkTracker deliberately accepts only production DSA work types.
        # Bypass the large CUDA constructor while retaining that type contract.
        work = object.__new__(dsa_comm.DsaGroupCastWork)
        work._collective_failed = False

        def wait():
            events.append(name)
            if error is not None:
                raise error

        work.wait = wait
        return work

    def test_work_tracker_early_exit_drains_nested_scopes_independently(self):
        events = []
        outer_work = self._fake_work(events, "outer")
        inner_work = self._fake_work(events, "inner")

        def leave_scope_early():
            with DsaWorkTracker(None):
                assert dsa_comm._track_current_dsa_work(outer_work) is outer_work
                with DsaWorkTracker(None):
                    assert dsa_comm._track_current_dsa_work(inner_work) is inner_work
                assert events == ["inner"]
                return "early"

        assert leave_scope_early() == "early"
        assert events == ["inner", "outer"]

        # Both context-variable tokens were restored. Work created outside a
        # scope must remain caller-owned rather than leaking into either list.
        untracked = self._fake_work(events, "untracked")
        assert dsa_comm._track_current_dsa_work(untracked) is untracked
        assert events == ["inner", "outer"]

    def test_work_tracker_drains_on_exception_and_recovers_for_next_scope(self):
        events = []
        pending = self._fake_work(events, "pending")
        with pytest.raises(RuntimeError, match="kernel failed"):
            with DsaWorkTracker(None):
                dsa_comm._track_current_dsa_work(pending)
                raise RuntimeError("kernel failed")
        assert events == ["pending"]

        drain_error = ValueError("post-process failed")
        failing = self._fake_work(events, "failing", drain_error)
        after_failure = self._fake_work(events, "after-failure")
        with pytest.raises(ValueError, match="post-process failed"):
            with DsaWorkTracker(None):
                dsa_comm._track_current_dsa_work(failing)
                dsa_comm._track_current_dsa_work(after_failure)

        recovered = self._fake_work(events, "recovered")
        with DsaWorkTracker(None):
            dsa_comm._track_current_dsa_work(recovered)
        assert events == ["pending", "failing", "after-failure", "recovered"]

    def test_work_tracker_aborts_after_collective_wait_failure(self, monkeypatch):
        events = []
        failure = RuntimeError("collective wait failed")
        failed_work = self._fake_work(events, "failed", failure)
        failed_work._collective_failed = True
        group = object()
        aborted = []
        monkeypatch.setattr(
            dsa_comm,
            "_abort_failed_process_group",
            lambda actual_group: aborted.append(actual_group),
        )

        with pytest.raises(RuntimeError, match="collective wait failed"):
            with DsaWorkTracker(group):  # type: ignore[arg-type]
                dsa_comm._track_current_dsa_work(failed_work)

        assert events == ["failed"]
        assert aborted == [group]

    def test_work_tracker_aborts_after_rank_local_compute_failure(self, monkeypatch):
        events = []
        pending = self._fake_work(events, "pending")
        group = object()
        aborted = []
        monkeypatch.setattr(
            dsa_comm,
            "_abort_failed_process_group",
            lambda actual_group: aborted.append(actual_group),
        )

        with pytest.raises(RuntimeError, match="kernel failed"):
            with DsaWorkTracker(group):  # type: ignore[arg-type]
                dsa_comm._track_current_dsa_work(pending)
                raise RuntimeError("kernel failed")

        assert events == ["pending"]
        assert aborted == [group]

    def test_four_payloads_have_independent_collective_args_and_unique_rows(self):
        plan = _noncontiguous_cp8_plan()
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

        assert comm_plan.window_kv.send_row_count > 0
        assert comm_plan.overlap_x.send_row_count > 0
        assert comm_plan.compressed_kv.send_row_ids == tuple(range(32)) + tuple(
            range(480, 512)
        )
        assert comm_plan.compressed_ki.send_row_ids == (
            comm_plan.compressed_kv.send_row_ids
        )
        assert comm_plan.compressed_kv.collective_arg.dst_indices_list == [
            list(range(1, 8))
        ]

    def test_nonadjacent_routes_and_zero_length_metadata_are_symmetric(self):
        plan = _noncontiguous_cp8_plan()
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
            [[DsaFragmentSpec(0, 0, 128)], [], [], [], [], [], [], []],
            compress_ratio=0,
            policy="empty-rank",
        )
        for rank in range(8):
            comm_plan = build_dsa_comm_plan(
                ratio0, rank, object()  # type: ignore[arg-type]
            )
            for meta in comm_plan:
                assert sum(meta.collective_arg.input_split_size_list) == 0
                assert sum(meta.collective_arg.output_split_size_list) == 0
                assert len(meta.collective_arg.input_split_size_list) >= 1

    def test_non_indexer_layers_keep_compressed_ki_route_empty(self):
        ratio128 = build_dsa_dispatch_plan(
            [1024],
            [
                [DsaFragmentSpec(0, 0, 128)],
                [DsaFragmentSpec(0, 128, 256)],
                [DsaFragmentSpec(0, 256, 384)],
                [DsaFragmentSpec(0, 384, 512)],
                [DsaFragmentSpec(0, 512, 640)],
                [DsaFragmentSpec(0, 640, 768)],
                [DsaFragmentSpec(0, 768, 896)],
                [DsaFragmentSpec(0, 896, 1024)],
            ],
            compress_ratio=128,
            policy="sequential",
        )
        for rank in range(8):
            comm_plan = build_dsa_comm_plan(
                ratio128, rank, object()  # type: ignore[arg-type]
            )
            assert comm_plan.compressed_kv.local_row_count == 1
            assert comm_plan.compressed_kv.receive_row_count == 7
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
            _noncontiguous_cp8_plan(), 0, object()  # type: ignore[arg-type]
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
                if not cast_works:
                    wrong_width = widths[meta.kind] + 1
                    with pytest.raises(ValueError, match="forward payload row shape"):
                        start_dsa_group_reduce(
                            DsaTypedPayload(
                                meta.kind,
                                torch.zeros(
                                    (meta.receive_row_count, wrong_width),
                                    dtype=torch.float32,
                                    device="cuda",
                                ),
                            ),
                            DsaTypedPayload(
                                meta.kind,
                                torch.zeros(
                                    (meta.local_row_count, wrong_width),
                                    dtype=torch.float32,
                                    device="cuda",
                                ),
                            ),
                            cast_work,
                            async_op=True,
                        )
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
        # CP=8 starts eight workers and initializes one full-node process group.
        # cuDNN's first sparse-backward call also compiles CUTLASS DSL kernels
        # independently on every device, so cold kernel tests need a wider bound.
        name = self._testMethodName
        if "full_kernel_backward" in name:
            return 1200
        if any(
            marker in name
            for marker in (
                "overlap_matrix",
                "two_inflight",
                "retain_graph",
                "coordinated_forward_exception",
            )
        ):
            return 900
        if "full_" in name or "short_sample" in name:
            return 600
        return 180

    @property
    def world_size(self) -> int:
        return 8

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @staticmethod
    def _width(kind: DsaPayloadKind) -> int:
        return {
            DsaPayloadKind.WINDOW_KV: 512,
            # Representative production trunk width; this also proves the
            # frozen 1 GiB native buffer covers the widest FP32 reverse route.
            DsaPayloadKind.OVERLAP_X: 7168,
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
                [[DsaFragmentSpec(0, 0, 128)], [], [], [], [], [], [], []],
                compress_ratio=0,
                policy="empty-rank",
            )
        else:
            dispatch_plan = _noncontiguous_cp8_plan()
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
        assert all(
            (work.native_handle_dict["group_cast"] is not None) is use_native
            for work in cast_works
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

        for meta, reduce_work in zip(comm_plan, reduce_works):
            owner_gradient = reduce_work.wait().tensor
            expected = torch.ones_like(owner_gradient)
            packed_begin = 0
            for split_size, destinations in zip(
                meta.collective_arg.input_split_size_list,
                meta.collective_arg.dst_indices_list,
            ):
                contribution = float(
                    sum(destination + 2 for destination in destinations)
                )
                for packed_row in range(packed_begin, packed_begin + split_size):
                    expected[meta.send_row_indices[packed_row]].add_(contribution)
                packed_begin += split_size
            assert packed_begin == len(meta.send_row_indices)
            torch.testing.assert_close(owner_gradient, expected, rtol=0, atol=0)

    @staticmethod
    def _forward_config(ratio: int, backend: str) -> MagiDSAConfig:
        return MagiDSAConfig(
            compress_ratio=ratio,
            hidden_size=64,
            q_lora_rank=64,
            softmax_scale=512**-0.5,
            backend=backend,
        )

    @staticmethod
    def _global_forward_input(
        config: MagiDSAConfig,
        *,
        lengths=(1024, 1024),
        seed=None,
    ) -> MagiDSAInput:
        total = sum(lengths)
        torch.manual_seed(20260710 + config.compress_ratio if seed is None else seed)

        def make(*shape):
            return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

        bounds = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        return MagiDSAInput(
            x=make(total, config.hidden_size),
            qr=make(total, config.q_lora_rank),
            q=make(total, config.num_heads, config.kv_dim),
            latent_kv=make(total, config.kv_dim),
            sink=torch.randn(config.num_heads, dtype=torch.float32, device="cuda"),
            packed_meta=DsaPackedMeta(torch.tensor(bounds, dtype=torch.int32)),
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

    @staticmethod
    def _require_input_gradients(dsa_input):
        for tensor in (
            dsa_input.x,
            dsa_input.qr,
            dsa_input.q,
            dsa_input.latent_kv,
            dsa_input.sink,
        ):
            tensor.requires_grad_(True)
        return dsa_input

    @staticmethod
    def _detach_trainable_input(dsa_input):
        return MagiDSAInput(
            x=dsa_input.x.detach().requires_grad_(True),
            qr=dsa_input.qr.detach().requires_grad_(True),
            q=dsa_input.q.detach().requires_grad_(True),
            latent_kv=dsa_input.latent_kv.detach().requires_grad_(True),
            sink=dsa_input.sink.detach().requires_grad_(True),
            packed_meta=dsa_input.packed_meta,
        )

    @staticmethod
    def _assert_local_input_gradients(local_input, global_input, local_rows):
        for name in ("x", "qr", "q", "latent_kv"):
            local_tensor = getattr(local_input, name)
            global_tensor = getattr(global_input, name)
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

    @staticmethod
    def _assert_parameter_gradients(
        runtime, reference_runtime, *, gradient_divisor: int = 1
    ):
        assert gradient_divisor >= 1
        for (name, parameter), (reference_name, reference_parameter) in zip(
            runtime.dsa_module.named_parameters(),
            reference_runtime.dsa_module.named_parameters(),
        ):
            assert name == reference_name
            assert parameter.grad is not None, name
            assert reference_parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad.float() / gradient_divisor,
                reference_parameter.grad.float() / gradient_divisor,
                rtol=3e-2,
                atol=3e-3,
                msg=f"parameter={name}",
            )
            assert getattr(parameter, "_magi_dsa_cp_reduced", False), name

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

        local_saved = []
        reference_saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: local_saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            local_output, local_kl = calc_dsa(local_input, runtime)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: reference_saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
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
        if local_output.grad_fn is None:
            # ratio=0 has no runtime parameters, and this forward-only helper
            # deliberately leaves every public input non-differentiable.
            assert reference_output.grad_fn is None
            assert not local_saved and not reference_saved
        else:
            assert len(local_saved) == len(reference_saved) == 9
            expected_topk = reference_saved[7].index_select(0, local_rows)
            expected_topk_length = reference_saved[8].index_select(0, local_rows)
            if backend == "reference":
                # Fragmented BF16 score GEMMs can reorder nearly equal valid
                # scores without changing the selected set. Each invocation's
                # own stable sort still enforces score order and id tie-break.
                torch.testing.assert_close(
                    local_saved[7].sort(dim=-1).values,
                    expected_topk.sort(dim=-1).values,
                    rtol=0,
                    atol=0,
                )
            else:
                torch.testing.assert_close(
                    local_saved[7], expected_topk, rtol=0, atol=0
                )
            torch.testing.assert_close(
                local_saved[8], expected_topk_length, rtol=0, atol=0
            )
            if backend == "kernel":
                assert local_saved[6].dtype == reference_saved[6].dtype == torch.float32
                torch.testing.assert_close(
                    local_saved[6],
                    reference_saved[6].index_select(0, local_rows),
                    rtol=2e-2,
                    atol=2e-4,
                )
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

    def _run_full_backward(
        self,
        *,
        ratio: int,
        backend: str,
        overlap_config: DsaOverlapConfig | None = None,
        lengths=(1024, 1024),
    ):
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        config = self._forward_config(ratio, backend)
        torch.manual_seed(1403 + ratio)
        runtime = MagiDSARuntimeMgr(
            config,
            cp_group=self.process_group,
            dispatch_policy="balanced",
            overlap_config=overlap_config,
        ).cuda()
        torch.manual_seed(1403 + ratio)
        reference_runtime = MagiDSARuntimeMgr(config).cuda()
        reference_runtime.load_state_dict(runtime.state_dict())
        runtime.train()
        reference_runtime.train()

        global_input = self._global_forward_input(config, lengths=lengths)
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
        expected_topk_width = config.topk if ratio == 4 else 0
        assert saved[7].shape == (local_rows.numel(), expected_topk_width)
        assert saved[8].dtype == torch.int32
        assert saved[8].shape == (local_rows.numel(),)
        torch.testing.assert_close(
            saved[8],
            (saved[7] >= 0).sum(dim=-1, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
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
        reference_output, reference_kl = calc_dsa(global_input, reference_runtime)
        torch.testing.assert_close(
            local_output,
            reference_output.index_select(0, local_rows),
            rtol=2e-2,
            atol=2e-3,
        )
        global_kl = local_kl.detach().clone()
        dist.all_reduce(global_kl, group=self.process_group)
        torch.testing.assert_close(global_kl, reference_kl, rtol=2e-2, atol=2e-5)

        local_loss.backward()
        (reference_output.float().mul(output_grad).sum() + reference_kl).backward()

        self._assert_local_input_gradients(local_input, global_input, local_rows)
        assert getattr(local_input.sink, "_magi_dsa_cp_reduced", False)
        self._assert_parameter_gradients(runtime, reference_runtime)
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

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_a2av_reference_pack_forward_and_reverse(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            self._run_transport(use_native=False)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_zero_length_routes_enter_all_four_collectives(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            self._run_transport(use_native=False, empty_routes=True)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_native_grpcoll_matches_reference_semantics(self):
        self._initialize_native_grpcoll_or_skip()
        try:
            with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=True):
                self._run_transport(use_native=True)
        finally:
            grpcoll_buffer_mgr.release_group(self.process_group)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_native_grpcoll_zero_length_routes_are_symmetric(self):
        self._initialize_native_grpcoll_or_skip()
        try:
            with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=True):
                self._run_transport(use_native=True, empty_routes=True)
        finally:
            grpcoll_buffer_mgr.release_group(self.process_group)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_native_grpcoll_full_reference_backward_matches_cp1(self):
        self._initialize_native_grpcoll_or_skip()
        try:
            with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=True):
                self._run_full_backward(
                    ratio=4,
                    backend="reference",
                    lengths=(512, 512),
                )
        finally:
            grpcoll_buffer_mgr.release_group(self.process_group)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_native_grpcoll_full_kernel_backward_matches_cp1(self):
        if not self._kernel_dependencies_available():
            self.skipTest("FlashMLA and cudnn-frontend DSA packages are required")
        self._initialize_native_grpcoll_or_skip()
        try:
            with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=True):
                self._run_full_backward(
                    ratio=4,
                    backend="kernel",
                    lengths=(512, 512),
                )
        finally:
            grpcoll_buffer_mgr.release_group(self.process_group)

    def _initialize_native_grpcoll_or_skip(self):
        try:
            grpcoll_buffer_mgr.initialize(
                group=self.process_group,
                config=GrpCollConfig(
                    num_sms=20,
                    nvl_chunk_size=8,
                    nvl_buffer_size=256,
                    rdma_chunk_size=8,
                    rdma_buffer_size=256,
                    # The same frozen CP8 configuration also covers the
                    # production 7168-wide FP32 reverse route (560.1 MiB
                    # exact minimum at 20 SMs) with ample metadata margin.
                    num_nvl_bytes=_CP8_NATIVE_NVL_BYTES,
                    num_rdma_bytes=0,
                ),
            )
        except (AssertionError, ImportError) as error:
            self.skipTest(f"native grpcoll extension is unavailable: {error}")

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_ratio4_reference_overlap_matrix_matches_cp1(self):
        import magi_attention.functional.dist_dsa as dist_dsa

        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for compressed_cast_indexer in (False, True):
                for dki_reduce_sparse_backward in (False, True):
                    events = []
                    project_queries = dist_dsa._project_ratio4_queries
                    cast_wait = dsa_comm.DsaGroupCastWork.wait
                    reduce_wait = dsa_comm.DsaGroupReduceWork.wait
                    attention_backward = dist_dsa._attention_backward

                    def traced_project(*args, **kwargs):
                        events.append("indexer_projection")
                        return project_queries(*args, **kwargs)

                    def traced_cast_wait(work):
                        if (
                            work.meta.kind is DsaPayloadKind.COMPRESSED_KV
                            and work._result is None
                        ):
                            events.append("compressed_cast_wait")
                        return cast_wait(work)

                    def traced_reduce_wait(work):
                        if (
                            work.meta.kind is DsaPayloadKind.COMPRESSED_KI
                            and work._result is None
                        ):
                            events.append("dki_reduce_wait")
                        return reduce_wait(work)

                    def traced_attention_backward(*args, **kwargs):
                        events.append("sparse_backward")
                        return attention_backward(*args, **kwargs)

                    with (
                        patch.object(
                            dist_dsa,
                            "_project_ratio4_queries",
                            side_effect=traced_project,
                        ),
                        patch.object(
                            dsa_comm.DsaGroupCastWork,
                            "wait",
                            new=traced_cast_wait,
                        ),
                        patch.object(
                            dsa_comm.DsaGroupReduceWork,
                            "wait",
                            new=traced_reduce_wait,
                        ),
                        patch.object(
                            dist_dsa,
                            "_attention_backward",
                            side_effect=traced_attention_backward,
                        ),
                    ):
                        self._run_full_backward(
                            ratio=4,
                            backend="reference",
                            lengths=(512, 512),
                            overlap_config=DsaOverlapConfig(
                                compressed_cast_indexer=compressed_cast_indexer,
                                dki_reduce_sparse_backward=(dki_reduce_sparse_backward),
                            ),
                        )

                    projection_position = events.index("indexer_projection")
                    cast_wait_position = events.index("compressed_cast_wait")
                    assert (projection_position < cast_wait_position) is (
                        compressed_cast_indexer
                    )
                    sparse_position = events.index("sparse_backward")
                    dki_wait_position = events.index("dki_reduce_wait")
                    assert (sparse_position < dki_wait_position) is (
                        dki_reduce_sparse_backward
                    )

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_coordinated_forward_exception_drains_real_collectives(self):
        import magi_attention.functional.dist_dsa as dist_dsa

        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            config = self._forward_config(4, "reference")
            torch.manual_seed(1700)
            runtime = MagiDSARuntimeMgr(
                config,
                cp_group=self.process_group,
                dispatch_policy="balanced",
            ).cuda()
            runtime.train()
            global_input = self._global_forward_input(
                config,
                lengths=(512, 512),
                seed=20260719,
            )
            dispatch_plan = runtime.get_dispatch_plan(global_input.packed_meta)
            local_input, _ = self._local_forward_input(
                global_input, dispatch_plan, self.rank
            )
            local_input = self._detach_trainable_input(local_input)
            abort_requests = []

            # The fault is injected symmetrically after all four real A2AV
            # works launch. Suppress physical fail-stop only so this test can
            # prove the common prefix drained and the group itself stayed
            # usable; separate unit tests require the abort request.
            with (
                patch.object(
                    dist_dsa,
                    "_project_ratio4_queries",
                    side_effect=RuntimeError("synthetic indexer kernel failure"),
                ),
                patch.object(
                    dsa_comm,
                    "_abort_failed_process_group",
                    side_effect=lambda group: abort_requests.append(group),
                ),
            ):
                with pytest.raises(
                    RuntimeError, match="synthetic indexer kernel failure"
                ):
                    calc_dsa(local_input, runtime)
            assert abort_requests == [self.process_group]

            dist.barrier(group=self.process_group)
            output, kl_loss = calc_dsa(local_input, runtime)
            (output.float().square().mean() + kl_loss).backward()
            assert torch.isfinite(output).all()
            assert runtime.forward_plan_cache_size == 1

            del output, kl_loss, local_input, global_input, runtime
            torch.cuda.empty_cache()

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_two_inflight_microbatches_reenter_and_accumulate_gradients(self):
        use_native = self._testMethodName.startswith("test_native_grpcoll_")
        if use_native:
            self._initialize_native_grpcoll_or_skip()
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=use_native):
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            config = self._forward_config(4, "reference")
            torch.manual_seed(1701)
            runtime = MagiDSARuntimeMgr(
                config,
                cp_group=self.process_group,
                dispatch_policy="balanced",
            ).cuda()
            reference_runtime = MagiDSARuntimeMgr(config).cuda()
            reference_runtime.load_state_dict(runtime.state_dict())
            sequential_runtime = MagiDSARuntimeMgr(
                config,
                cp_group=self.process_group,
                dispatch_policy="balanced",
            ).cuda()
            sequential_runtime.load_state_dict(runtime.state_dict())
            runtime.train()
            reference_runtime.train()
            sequential_runtime.train()

            records = []
            for microbatch in range(2):
                global_input = self._require_input_gradients(
                    self._global_forward_input(
                        config,
                        lengths=(512, 512),
                        seed=20260720 + microbatch,
                    )
                )
                dispatch_plan = runtime.get_dispatch_plan(global_input.packed_meta)
                local_input, local_rows = self._local_forward_input(
                    global_input, dispatch_plan, self.rank
                )
                local_input = self._detach_trainable_input(local_input)
                local_output, local_kl = calc_dsa(local_input, runtime)
                torch.manual_seed(8675400 + microbatch)
                output_grad = torch.randn(
                    global_input.q.shape,
                    dtype=torch.float32,
                    device=global_input.q.device,
                ) / global_input.q.size(0)
                local_loss = (
                    local_output.float() * output_grad.index_select(0, local_rows)
                ).sum() + local_kl
                records.append(
                    {
                        "global_input": global_input,
                        "local_input": local_input,
                        "local_rows": local_rows,
                        "local_output": local_output,
                        "local_kl": local_kl,
                        "local_loss": local_loss,
                        "output_grad": output_grad,
                    }
                )

            # Both autograd contexts coexist on one runtime and share only the
            # immutable cached plan. Their work/buffers remain invocation-local.
            assert records[0]["local_output"].grad_fn is not None
            assert records[1]["local_output"].grad_fn is not None
            assert (
                records[0]["local_output"].grad_fn
                is not records[1]["local_output"].grad_fn
            )
            assert runtime.forward_plan_cache_size == 1

            reference_losses = []
            for record in records:
                reference_output, reference_kl = calc_dsa(
                    record["global_input"], reference_runtime
                )
                torch.testing.assert_close(
                    record["local_output"],
                    reference_output.index_select(0, record["local_rows"]),
                    rtol=2e-2,
                    atol=2e-3,
                )
                global_kl = record["local_kl"].detach().clone()
                dist.all_reduce(global_kl, group=self.process_group)
                torch.testing.assert_close(
                    global_kl, reference_kl, rtol=2e-2, atol=2e-5
                )
                reference_losses.append(
                    reference_output.float().mul(record["output_grad"]).sum()
                    + reference_kl
                )

            sequential_records = []
            for record in records:
                sequential_input = self._detach_trainable_input(record["local_input"])
                sequential_output, sequential_kl = calc_dsa(
                    sequential_input, sequential_runtime
                )
                sequential_loss = (
                    sequential_output.float()
                    * record["output_grad"].index_select(0, record["local_rows"])
                ).sum() + sequential_kl
                sequential_records.append(
                    {
                        "input": sequential_input,
                        "output": sequential_output,
                        "loss": sequential_loss,
                    }
                )
                torch.testing.assert_close(
                    record["local_output"], sequential_output, rtol=0, atol=0
                )

            # Reenter microbatch 1 after microbatch 0 has launched its async
            # dKi GroupReduce but before the outer sparse backward/wait. This
            # nests two real DSA work-tracker scopes, not only two engines.
            import magi_attention.functional.dist_dsa as dist_dsa

            reentered = []
            attention_backward = dist_dsa._attention_backward
            reentrant_records = records

            def reentrant_attention_backward(*args, **kwargs):
                if not reentered:
                    reentered.append(True)
                    reentrant_records[1]["local_loss"].backward()
                return attention_backward(*args, **kwargs)

            with patch.object(
                dist_dsa,
                "_attention_backward",
                side_effect=reentrant_attention_backward,
            ):
                records[0]["local_loss"].backward()
            assert reentered == [True]

            # A separate CP=8 runtime executes the same two backwards in the
            # same inner-then-outer order. Reentrant and sequential CP=8 must
            # be bitwise identical before applying the CP=1 numerical budget.
            sequential_records[1]["loss"].backward()
            sequential_records[0]["loss"].backward()
            for record, sequential_record in zip(records, sequential_records):
                for name in ("x", "qr", "q", "latent_kv", "sink"):
                    actual = getattr(record["local_input"], name).grad
                    expected = getattr(sequential_record["input"], name).grad
                    assert (actual is None) is (expected is None), name
                    if actual is not None:
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0, msg=name
                        )
            for (name, parameter), (sequential_name, sequential_parameter) in zip(
                runtime.dsa_module.named_parameters(),
                sequential_runtime.dsa_module.named_parameters(),
            ):
                assert name == sequential_name
                torch.testing.assert_close(
                    parameter.grad,
                    sequential_parameter.grad,
                    rtol=0,
                    atol=0,
                    msg=f"reentrant parameter={name}",
                )

            # The CP runtime accumulates two already-CP-reduced parameter
            # contributions. Compare the averaged gradient so the original
            # single-microbatch absolute error budget remains unchanged.
            for reference_loss in reference_losses:
                reference_loss.backward()
            for record in records:
                self._assert_local_input_gradients(
                    record["local_input"],
                    record["global_input"],
                    record["local_rows"],
                )
            self._assert_parameter_gradients(
                runtime,
                reference_runtime,
                gradient_divisor=len(records),
            )

            del (
                records,
                sequential_records,
                reference_losses,
                runtime,
                sequential_runtime,
                reference_runtime,
            )
            torch.cuda.empty_cache()
        if use_native:
            grpcoll_buffer_mgr.release_group(self.process_group)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_native_grpcoll_two_inflight_reentrant_backward_is_isolated(self):
        implementation = (
            TestDsaCommTransport.test_cp8_two_inflight_microbatches_reenter_and_accumulate_gradients
        )
        while hasattr(implementation, "__wrapped__"):
            implementation = implementation.__wrapped__
        implementation(self)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_retain_graph_backward_is_repeatable(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            config = self._forward_config(4, "reference")
            torch.manual_seed(1702)
            runtime = MagiDSARuntimeMgr(config, cp_group=self.process_group).cuda()
            runtime.train()
            global_input = self._global_forward_input(
                config,
                lengths=(512, 512),
                seed=20260722,
            )
            dispatch_plan = runtime.get_dispatch_plan(global_input.packed_meta)
            local_input, local_rows = self._local_forward_input(
                global_input, dispatch_plan, self.rank
            )
            local_input = self._detach_trainable_input(local_input)
            local_output, local_kl = calc_dsa(local_input, runtime)
            torch.manual_seed(8675402)
            output_grad = torch.randn(
                global_input.q.shape,
                dtype=torch.float32,
                device=global_input.q.device,
            ) / global_input.q.size(0)
            loss = (
                local_output.float() * output_grad.index_select(0, local_rows)
            ).sum() + local_kl

            loss.backward(retain_graph=True)
            gradient_tensors = {
                "x": local_input.x,
                "qr": local_input.qr,
                "q": local_input.q,
                "latent_kv": local_input.latent_kv,
                "sink": local_input.sink,
                **{
                    f"parameter:{name}": parameter
                    for name, parameter in runtime.dsa_module.named_parameters()
                },
            }
            first_gradients = {
                name: None if tensor.grad is None else tensor.grad.detach().clone()
                for name, tensor in gradient_tensors.items()
            }

            loss.backward()
            for name, tensor in gradient_tensors.items():
                first = first_gradients[name]
                if first is None:
                    assert tensor.grad is None, name
                else:
                    assert tensor.grad is not None, name
                    torch.testing.assert_close(
                        tensor.grad,
                        first + first,
                        rtol=3e-2,
                        atol=3e-3,
                        msg=name,
                    )

            del local_output, local_kl, loss, local_input, global_input, runtime
            torch.cuda.empty_cache()

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_full_reference_empty_rank_matches_cp1(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            config = self._forward_config(4, "reference")
            torch.manual_seed(1703)
            runtime = MagiDSARuntimeMgr(
                config,
                cp_group=self.process_group,
                dispatch_policy="balanced",
            ).cuda()
            reference_runtime = MagiDSARuntimeMgr(config).cuda()
            reference_runtime.load_state_dict(runtime.state_dict())
            runtime.train()
            reference_runtime.train()

            global_input = self._require_input_gradients(
                self._global_forward_input(
                    config,
                    lengths=(7,),
                    seed=20260723,
                )
            )
            dispatch_plan = runtime.get_dispatch_plan(global_input.packed_meta)
            assert sorted(rank.token_count for rank in dispatch_plan.ranks) == [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                7,
            ]
            local_input, local_rows = self._local_forward_input(
                global_input, dispatch_plan, self.rank
            )
            local_input = self._detach_trainable_input(local_input)
            local_output, local_kl = calc_dsa(local_input, runtime)
            reference_output, reference_kl = calc_dsa(global_input, reference_runtime)
            assert local_output.shape[0] == dispatch_plan.ranks[self.rank].token_count
            torch.testing.assert_close(
                local_output,
                reference_output.index_select(0, local_rows),
                rtol=2e-2,
                atol=2e-3,
            )
            global_kl = local_kl.detach().clone()
            dist.all_reduce(global_kl, group=self.process_group)
            torch.testing.assert_close(global_kl, reference_kl, rtol=2e-2, atol=2e-5)

            torch.manual_seed(8675403)
            output_grad = torch.randn(
                global_input.q.shape,
                dtype=torch.float32,
                device=global_input.q.device,
            ) / global_input.q.size(0)
            local_loss = (
                local_output.float() * output_grad.index_select(0, local_rows)
            ).sum() + local_kl
            local_loss.backward()
            (reference_output.float().mul(output_grad).sum() + reference_kl).backward()
            self._assert_local_input_gradients(local_input, global_input, local_rows)
            self._assert_parameter_gradients(runtime, reference_runtime)

            del (
                local_output,
                reference_output,
                local_kl,
                reference_kl,
                local_loss,
                local_input,
                global_input,
                runtime,
                reference_runtime,
            )
            torch.cuda.empty_cache()

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_short_sample_without_blocks_has_zero_differentiable_kl(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            self._run_full_backward(
                ratio=4,
                backend="reference",
                lengths=(1, 128),
            )

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_full_reference_forward_matches_cp1(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for policy in ("sequential", "balanced"):
                for ratio in (0, 4, 128):
                    self._run_full_forward(
                        ratio=ratio,
                        policy=policy,
                        backend="reference",
                    )

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_full_kernel_forward_matches_cp1(self):
        if not self._kernel_dependencies_available():
            self.skipTest("FlashMLA and cudnn-frontend DSA packages are required")
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for ratio in (0, 4, 128):
                self._run_full_forward(
                    ratio=ratio,
                    policy="balanced",
                    backend="kernel",
                )

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_full_reference_backward_matches_cp1(self):
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for ratio in (0, 4, 128):
                self._run_full_backward(ratio=ratio, backend="reference")

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_cp8_full_kernel_backward_matches_cp1(self):
        if not self._kernel_dependencies_available():
            self.skipTest("FlashMLA and cudnn-frontend DSA packages are required")
        with switch_envvar_context("MAGI_ATTENTION_NATIVE_GRPCOLL", enable=False):
            for ratio in (0, 4, 128):
                self._run_full_backward(ratio=ratio, backend="kernel")


if __name__ == "__main__":
    run_tests()
