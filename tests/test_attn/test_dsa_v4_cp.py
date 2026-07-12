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

"""Eight-rank CP tests for the Magi_DSA V4 runtime (reference backend).

CP=8 against a CP=1 full-sequence run with identical weights, for the
single-sample and packed paths, all three layer forms. Tolerance
methodology: elementwise bf16-relative bounds for outputs and direct
gradients; cosine/relative-L2 similarity for gradients downstream of
top-k selection (borderline picks can flip when GEMM row counts differ
across the split — same acceptance style as the upstream Megatron CP
suite).
"""

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention.experimental.dsa_v4 import MagiDSAV4, MagiDSAV4Config
from magi_attention.experimental.dsa_v4.config import MagiDSAV4YarnConfig
from magi_attention.experimental.dsa_v4.cp import forward_cp, forward_cp_packed
from magi_attention.testing.dist_common import DistTestBase, with_comms

SQ_GLOBAL = 2048
HIDDEN = 256
# Ragged packed batch. The total gives every CP rank 256 rows, which is
# sufficient for the ratio-128 packed halo, and several samples cross a CP cut.
SEG_LENS = [37, 128, 200, 3, 96, 48, 511, 257, 333, 435]
assert sum(SEG_LENS) == SQ_GLOBAL


def _make_config(ratio: int) -> MagiDSAV4Config:
    return MagiDSAV4Config(
        compress_ratio=ratio,
        hidden_size=HIDDEN,
        q_lora_rank=64,
        softmax_scale=64**-0.5,
        num_heads=16,
        kv_dim=64,
        rope_dim=32,
        window_size=8,
        topk=8,
        indexer_heads=8,
        indexer_dim=64,
        yarn=MagiDSAV4YarnConfig(rotary_base=10000.0, scaling_factor=1.0),
    )


def _rel_ok(a: torch.Tensor, b: torch.Tensor, rtol=8e-3, atol=1e-5) -> bool:
    d = (a.float() - b.float()).abs().max().item()
    return d <= atol + rtol * b.float().abs().max().item()


def _sim_ok(a: torch.Tensor, b: torch.Tensor, thres=1e-3) -> bool:
    af, bf = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    rel_l2 = ((af - bf).norm() / bf.norm().clamp(min=1e-12)).item()
    return (1.0 - cos) <= thres and rel_l2 <= 3e-2


class TestDSAV4CP8(DistTestBase):
    @property
    def timeout(self) -> int:
        return 600

    @property
    def world_size(self) -> int:
        return 8

    @property
    def seed(self) -> int:
        return 42

    def _modules(self, ratio):
        torch.manual_seed(self.seed)
        module = MagiDSAV4(_make_config(ratio)).cuda().train()
        torch.manual_seed(self.seed)
        module_ref = MagiDSAV4(_make_config(ratio)).cuda().train()
        return module, module_ref

    def _inputs(self, ratio, flat: bool):
        torch.manual_seed(100 + ratio)
        shape_b = () if flat else (1,)
        x = torch.randn(
            SQ_GLOBAL, *shape_b, HIDDEN, dtype=torch.bfloat16, device="cuda"
        )
        qr = torch.randn(SQ_GLOBAL, *shape_b, 64, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(
            SQ_GLOBAL, *shape_b, 16, 64, dtype=torch.bfloat16, device="cuda"
        )
        kv = torch.randn(SQ_GLOBAL, *shape_b, 64, dtype=torch.bfloat16, device="cuda")
        sink = torch.randn(16, dtype=torch.float32, device="cuda")
        g = torch.randn(
            SQ_GLOBAL, *shape_b, 16 * 64, dtype=torch.bfloat16, device="cuda"
        )
        return x, qr, q, kv, sink, g

    def _run_case(self, ratio, packed: bool):
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        group = dist.group.WORLD
        sq_local = SQ_GLOBAL // self.world_size
        start = self.rank * sq_local
        seg = slice(start, start + sq_local)

        module, module_ref = self._modules(ratio)
        x_f, qr_f, q_f, kv_f, sink_f, g_f = self._inputs(ratio, flat=packed)

        loc = [t[seg].clone().requires_grad_(True) for t in (x_f, qr_f, q_f, kv_f)]
        sink = sink_f.clone().requires_grad_(True)
        if packed:
            cu = torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(SEG_LENS), 0)), dtype=torch.int32
            )
            out, kl = forward_cp_packed(module, *loc, sink, cu, group)
        else:
            out, kl = forward_cp(module, *loc, sink, group)
        ((out.float() * g_f[seg].float()).sum() + kl).backward()

        ref = [t.clone().requires_grad_(True) for t in (x_f, qr_f, q_f, kv_f)]
        sink_ref = sink_f.clone().requires_grad_(True)
        if packed:
            out_ref, kl_ref = module_ref.forward_packed(*ref, sink_ref, cu)
        else:
            out_ref, kl_ref = module_ref(*ref, sink_ref)
        ((out_ref.float() * g_f.float()).sum() + kl_ref).backward()

        assert _rel_ok(out, out_ref[seg]), f"ratio={ratio} packed={packed}: output"
        kl_sum = kl.detach().clone()
        dist.all_reduce(kl_sum)
        assert (
            abs(kl_sum.item() - kl_ref.item()) <= 5e-5
        ), f"ratio={ratio} packed={packed}: kl"
        assert _rel_ok(loc[2].grad, ref[2].grad[seg]), f"ratio={ratio}: dq"
        assert _rel_ok(loc[3].grad, ref[3].grad[seg]), f"ratio={ratio}: dkv"
        if ratio > 1:
            assert _sim_ok(loc[0].grad, ref[0].grad[seg]), f"ratio={ratio}: dx"

        sink_g = sink.grad.detach().clone()
        dist.all_reduce(sink_g)
        assert _rel_ok(
            sink_g, sink_ref.grad, rtol=1e-2, atol=1e-4
        ), f"ratio={ratio} packed={packed}: d_sink"
        for (n, p), (_, p_ref) in zip(
            module.named_parameters(), module_ref.named_parameters()
        ):
            grad = (
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            )
            dist.all_reduce(grad)
            ref_grad = p_ref.grad if p_ref.grad is not None else torch.zeros_like(p_ref)
            if ref_grad.abs().max() > 1e-6:
                assert _sim_ok(
                    grad, ref_grad, thres=2e-3
                ), f"ratio={ratio} packed={packed}: param {n}"

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_single_sample_cp8(self):
        for ratio in (0, 4, 128):
            self._run_case(ratio, packed=False)

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_packed_cp8(self):
        for ratio in (0, 4, 128):
            self._run_case(ratio, packed=True)


if __name__ == "__main__":
    run_tests()
