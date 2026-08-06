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

COPY_SPECS = (
    ("bf16", 8),
    ("bf16", 16),
    ("bf16", 32),
    ("bf16", 128),
    ("bf16", 512),
    ("bf16", 1024),
    ("bf16", 1536),
    ("bf16", 4096),
    ("bf16", 7168),
    ("bf16", 32768),
    ("bf16", 65536),
    ("f32", 4),
    ("f32", 64),
    ("f32", 128),
    ("i32", 4),
    ("i32", 512),
    ("i32", 1024),
)

REDUCE_SPECS = (
    ("bf16", 8),
    ("bf16", 16),
    ("bf16", 32),
    ("bf16", 128),
    ("bf16", 512),
    ("bf16", 4096),
    ("bf16", 7168),
)


def required_object_names() -> tuple[str, ...]:
    copy = tuple(f"copy_sm103_{dtype}_w{width}.o" for dtype, width in COPY_SPECS)
    reduce = tuple(f"reduce_sm103_{dtype}_w{width}.o" for dtype, width in REDUCE_SPECS)
    return copy + reduce


if __name__ == "__main__":
    print("\n".join(required_object_names()))
