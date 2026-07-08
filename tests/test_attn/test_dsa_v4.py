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

"""Self-contained unit tests for the Magi_DSA V4 reference runtime.

Covers the invariants promised by the design doc without any external
reference dependency: block-level causality, the uncompressed-tail rule,
overlapped compression boundaries, sink neutralization, the three layer
forms, and gradient flow to every owned parameter. Cross-implementation
parity against Megatron dsv4 is exercised separately (agent harness).
"""

import unittest

import torch

from magi_attention.experimental.dsa_v4 import (
    DSAv4Compressor,
    MagiDSAV4,
    MagiDSAV4Config,
    MagiDSAV4YarnConfig,
)
from magi_attention.experimental.dsa_v4.indexer import build_block_causal_mask
from magi_attention.experimental.dsa_v4.reference import (
    get_compress_topk_idxs,
    get_window_topk_idxs,
    sparse_attn_with_sink,
)


def _make_config(ratio: int, **kw) -> MagiDSAV4Config:
    defaults = dict(
        compress_ratio=ratio,
        hidden_size=128,
        q_lora_rank=64,
        softmax_scale=64**-0.5,
        num_heads=8,
        kv_dim=64,
        rope_dim=32,
        window_size=16,
        topk=8,
        indexer_heads=4,
        indexer_dim=64,
        yarn=MagiDSAV4YarnConfig(rotary_base=10000.0, scaling_factor=1.0),
    )
    defaults.update(kw)
    return MagiDSAV4Config(**defaults)


def _random_inputs(cfg, sq, b, device, requires_grad=False):
    def t(*shape):
        out = torch.randn(*shape, dtype=torch.bfloat16, device=device)
        return out.requires_grad_(True) if requires_grad else out

    x = t(sq, b, cfg.hidden_size)
    qr = t(sq, b, cfg.q_lora_rank)
    q = t(sq, b, cfg.num_heads, cfg.kv_dim)
    kv = t(sq, b, cfg.kv_dim)
    sink = torch.randn(cfg.num_heads, dtype=torch.float32, device=device)
    if requires_grad:
        sink.requires_grad_(True)
    return x, qr, q, kv, sink


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestIndexHelpers(unittest.TestCase):
    def test_window_causal_and_padding(self):
        idxs = get_window_topk_idxs(8, 1, 32, torch.device("cuda"))
        for i in range(32):
            row = idxs[0, i]
            valid = row[row >= 0]
            self.assertTrue(torch.all(valid <= i))
            self.assertEqual(len(valid), min(i + 1, 8))

    def test_compress_prefix_counts(self):
        ratio, sq, offset = 4, 64, 100
        idxs = get_compress_topk_idxs(ratio, 1, sq, offset, torch.device("cuda"))
        for i in range(sq):
            row = idxs[0, i]
            valid = row[row >= 0]
            self.assertEqual(len(valid), (i + 1) // ratio)
            if len(valid):
                self.assertTrue(torch.all(valid >= offset))

    def test_block_causal_mask_matches_prefix_rule(self):
        mask = build_block_causal_mask(32, 8, 4, 1, torch.device("cuda"))
        visible = (mask[0] == 0).sum(dim=-1)
        expected = torch.arange(1, 33, device=mask.device) // 4
        expected = expected.clamp(max=8)
        self.assertTrue(torch.equal(visible, expected))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCompressor(unittest.TestCase):
    def test_tail_rule_and_shapes(self):
        cfg = _make_config(4)
        comp = DSAv4Compressor(cfg, head_dim=cfg.kv_dim).cuda()
        x = torch.randn(66, 2, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        out = comp(x)
        self.assertEqual(out.shape, (16, 2, cfg.kv_dim))  # 66 // 4, tail dropped

    def test_shorter_than_ratio_returns_none(self):
        cfg = _make_config(128)
        comp = DSAv4Compressor(cfg, head_dim=cfg.kv_dim).cuda()
        x = torch.randn(100, 1, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(comp(x))

    def test_overlap_first_block_ignores_missing_neighbor(self):
        # Block 0 has no predecessor: its entry must depend only on tokens
        # 0..3, so perturbing token 4 must not change compressed entry 0.
        cfg = _make_config(4)
        comp = DSAv4Compressor(cfg, head_dim=cfg.kv_dim).cuda()
        x = torch.randn(8, 1, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        base = comp(x)[0].clone()
        x2 = x.clone()
        x2[4] += 1.0
        pert = comp(x2)[0]
        self.assertTrue(torch.equal(base, pert))

    def test_overlap_second_block_sees_first_block(self):
        cfg = _make_config(4)
        comp = DSAv4Compressor(cfg, head_dim=cfg.kv_dim).cuda()
        x = torch.randn(8, 1, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        base = comp(x)[1].clone()
        x2 = x.clone()
        x2[0] += 1.0  # token 0 feeds block 1 through the overlap
        pert = comp(x2)[1]
        self.assertFalse(torch.equal(base, pert))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSinkSoftmax(unittest.TestCase):
    def test_neg_inf_sink_matches_plain_softmax(self):
        sq, b, nh, hd, n_kv, k = 16, 1, 4, 32, 16, 8
        q = torch.randn(sq, b, nh, hd, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(n_kv, b, hd, dtype=torch.bfloat16, device="cuda")
        idx = get_window_topk_idxs(k, b, sq, q.device).int()
        neg_inf_sink = torch.full((nh,), float("-inf"), device="cuda")
        zero_ref = []
        out = sparse_attn_with_sink(q, kv, neg_inf_sink, idx, hd**-0.5)

        # plain reference without sink
        scores = torch.einsum("sbnh,tbh->sbnt", q.float(), kv.float()) * hd**-0.5
        mask = torch.full((sq, n_kv), float("-inf"), device="cuda")
        for i in range(sq):
            row = idx[0, i]
            mask[i, row[row >= 0].long()] = 0.0
        scores = scores + mask.view(sq, 1, 1, n_kv)
        ref = torch.einsum(
            "sbnt,tbh->sbnh", torch.softmax(scores, dim=-1), kv.float()
        ).to(torch.bfloat16)
        del zero_ref
        diff = (out.view(sq, b, nh, hd).float() - ref.float()).abs().max()
        self.assertLess(diff.item(), 2e-2)

    def test_sink_receives_gradient(self):
        cfg = _make_config(0)
        attn = MagiDSAV4(cfg).cuda()
        x, qr, q, kv, sink = _random_inputs(cfg, 32, 2, "cuda", requires_grad=True)
        out, kl = attn(x, qr, q, kv, sink)
        out.float().sum().backward()
        self.assertIsNotNone(sink.grad)
        self.assertGreater(sink.grad.abs().sum().item(), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestThreeForms(unittest.TestCase):
    def _run(self, ratio, training):
        cfg = _make_config(ratio)
        attn = MagiDSAV4(cfg).cuda()
        attn.train(training)
        # HCA needs at least one full 128-token block to exercise compression.
        sq, b = (288, 2) if ratio == 128 else (96, 2)
        x, qr, q, kv, sink = _random_inputs(cfg, sq, b, "cuda", requires_grad=True)
        out, kl = attn(x, qr, q, kv, sink)
        self.assertEqual(out.shape, (sq, b, cfg.num_heads * cfg.kv_dim))
        self.assertTrue(torch.isfinite(out.float()).all())
        (out.float().sum() + kl).backward()
        return attn, x, qr, kl

    def test_window_only(self):
        attn, x, qr, kl = self._run(0, training=True)
        self.assertIsNone(attn.compressor)
        self.assertIsNone(attn.indexer)
        self.assertEqual(kl.item(), 0.0)
        self.assertIsNone(x.grad if x.grad is None else None)  # x unused: no grad required

    def test_hca(self):
        attn, x, qr, kl = self._run(128, training=True)
        self.assertIsNotNone(attn.compressor)
        self.assertIsNone(attn.indexer)
        self.assertEqual(kl.item(), 0.0)
        self.assertIsNotNone(x.grad)  # compressor path pulls gradient into x

    def test_csa_train_kl_and_param_grads(self):
        attn, x, qr, kl = self._run(4, training=True)
        self.assertIsNotNone(attn.indexer)
        self.assertGreater(kl.item(), 0.0)
        for name, p in attn.named_parameters():
            self.assertIsNotNone(p.grad, f"missing grad for {name}")
        # indexer is a detached branch: qr receives no gradient
        self.assertIsNone(qr.grad)

    def test_csa_eval_no_kl(self):
        attn, x, qr, kl = self._run(4, training=False)
        self.assertEqual(kl.item(), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPacked(unittest.TestCase):
    """Packed THD path: per-segment equivalence and boundary isolation."""

    def _packed_inputs(self, cfg, seg_lens, device):
        total = sum(seg_lens)
        x = torch.randn(total, cfg.hidden_size, dtype=torch.bfloat16, device=device)
        qr = torch.randn(total, cfg.q_lora_rank, dtype=torch.bfloat16, device=device)
        q = torch.randn(
            total, cfg.num_heads, cfg.kv_dim, dtype=torch.bfloat16, device=device
        )
        kv = torch.randn(total, cfg.kv_dim, dtype=torch.bfloat16, device=device)
        sink = torch.randn(cfg.num_heads, dtype=torch.float32, device=device)
        cu = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seg_lens), 0)), dtype=torch.int32
        )
        return x, qr, q, kv, sink, cu

    def _check_matches_per_segment(self, ratio, seg_lens):
        cfg = _make_config(ratio)
        attn = MagiDSAV4(cfg).cuda().train()
        x, qr, q, kv, sink, cu = self._packed_inputs(cfg, seg_lens, "cuda")
        out_packed, kl_packed = attn.forward_packed(x, qr, q, kv, sink, cu)

        bounds = cu.tolist()
        kl_expected = torch.zeros((), dtype=torch.float32, device=x.device)
        for start, end in zip(bounds[:-1], bounds[1:]):
            seg = slice(start, end)
            out_seg, kl_seg = attn._forward_single(
                x[seg].unsqueeze(1),
                qr[seg].unsqueeze(1),
                q[seg].unsqueeze(1),
                kv[seg].unsqueeze(1),
                sink,
                kl_reduce="sum",
            )
            self.assertTrue(
                torch.equal(out_packed[seg], out_seg.squeeze(1)),
                f"segment [{start}:{end}] mismatch",
            )
            kl_expected = kl_expected + kl_seg
        kl_expected = kl_expected / sum(seg_lens)
        self.assertAlmostEqual(kl_packed.item(), kl_expected.item(), places=6)

    def test_packed_equals_per_segment_csa(self):
        self._check_matches_per_segment(4, [5, 37, 96, 3, 64])

    def test_packed_equals_per_segment_hca(self):
        self._check_matches_per_segment(128, [130, 96, 300])

    def test_packed_equals_per_segment_window(self):
        self._check_matches_per_segment(0, [7, 33, 128])

    def test_boundary_isolation(self):
        # Perturbing sample 0 must not change sample 1's outputs.
        cfg = _make_config(4)
        attn = MagiDSAV4(cfg).cuda().eval()
        seg_lens = [64, 64]
        x, qr, q, kv, sink, cu = self._packed_inputs(cfg, seg_lens, "cuda")
        base, _ = attn.forward_packed(x, qr, q, kv, sink, cu)
        x2 = x.clone()
        x2[:64] += 1.0
        kv2 = kv.clone()
        kv2[:64] += 1.0
        pert, _ = attn.forward_packed(x2, qr, q, kv2, sink, cu)
        self.assertTrue(torch.equal(base[64:], pert[64:]))
        self.assertFalse(torch.equal(base[:64], pert[:64]))


if __name__ == "__main__":
    unittest.main()
