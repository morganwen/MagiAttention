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

import json
import math
from importlib import metadata

import flash_mla
import torch


def main() -> None:
    torch.manual_seed(0)
    assert torch.cuda.get_device_capability() == (10, 3)
    version = metadata.version("flash-mla")
    assert version.endswith("+9241ae3"), version

    seqlen_q, seqlen_k, heads, dim, top_k = 64, 1024, 64, 512, 512
    q = torch.randn(seqlen_q, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seqlen_k, 1, dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    row = torch.arange(seqlen_q, device="cuda", dtype=torch.int32).view(-1, 1, 1)
    column = torch.arange(top_k, device="cuda", dtype=torch.int32).view(1, 1, -1)
    indices = (row * 7 + column).remainder(seqlen_k).contiguous()
    topk_length = 256 + torch.arange(seqlen_q, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(dim)

    out, max_logits, lse = flash_mla.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale=scale,
        d_v=512,
        attn_sink=sink,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()
    print("flashmla_sparse_fwd: executed", flush=True)

    flat_indices = indices[:, 0].clone()
    position = torch.arange(top_k, device="cuda").view(1, -1)
    invalid = position >= topk_length.view(-1, 1)
    flat_indices[invalid] = 0
    selected_kv = kv[:, 0].index_select(0, flat_indices.flatten().long())
    selected_kv = selected_kv.view(seqlen_q, top_k, dim).float()
    score = torch.einsum("qhd,qkd->qhk", q.float(), selected_kv) * scale
    score = score.masked_fill(invalid.unsqueeze(1), float("-inf"))
    reference_max = score.max(dim=-1).values
    reference_lse = torch.logsumexp(score, dim=-1)
    denominator = torch.logaddexp(reference_lse, sink.view(1, heads))
    probability = torch.exp(score - denominator.unsqueeze(-1))
    reference_out = torch.einsum("qhk,qkd->qhd", probability, selected_kv)

    torch.testing.assert_close(out.float(), reference_out, atol=5e-3, rtol=2e-2)
    torch.testing.assert_close(max_logits, reference_max, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse, reference_lse, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(out).all()
    print("flashmla_sparse_fwd: reference matched", flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "flashmla": version,
        "lse_shape": list(lse.shape),
        "max_logits_shape": list(max_logits.shape),
        "out_shape": list(out.shape),
        "topk_length_max": int(topk_length.max().item()),
        "topk_length_min": int(topk_length.min().item()),
    }
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
