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

"""Compute kernels backing the Magi-DSA runtime.

Only fused elementwise and index kernels live here. Row gathers and reductions
are MagiAttention Core range ops, and the collectives are Core group
collectives, so this package carries no packing or communication kernel.

The Triton wrappers import lazily so CPU plan and reference paths do not
require the Triton toolchain.
"""
