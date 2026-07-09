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

"""Magi_DSA V4: DeepSeek V4 hybrid sparse attention runtime.

API v1 frozen 2026-07-09 after full-layer drop-in parity inside Megatron's
DSv4HybridSelfAttention (all three layer forms, outputs and every shared
parameter gradient matching). Inputs: hidden x, query latent qr, RoPE-applied
main query and latent KV, learnable sink; indexer projections and compressed
entries are runtime-internal.
"""

__api_version__ = "1.0.0"

from .attention import MagiDSAV4
from .compressor import DSAv4Compressor
from .config import MagiDSAV4Config, MagiDSAV4YarnConfig
from .indexer import DSAv4Indexer

__all__ = [
    "MagiDSAV4",
    "MagiDSAV4Config",
    "MagiDSAV4YarnConfig",
    "DSAv4Compressor",
    "DSAv4Indexer",
]
