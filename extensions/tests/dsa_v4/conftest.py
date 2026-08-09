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

"""Shared helpers for the Magi-DSA test suite."""

from __future__ import annotations

from pathlib import Path

_ROOT_MARKERS = ("pyproject.toml", "extensions/setup.py")


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up from ``start``.

    The suite used to hardcode ``Path(__file__).resolve().parents[2]``, which
    pointed at ``extensions/`` once these tests moved one level deeper. Looking
    for the marker files instead keeps the resolver correct wherever the suite
    is mounted, including the read-only container mounts used by the multi-GPU
    and release scripts.
    """

    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if not candidate.is_dir():
            continue
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise RuntimeError(
        f"could not locate the repository root above {current}: "
        f"none of its parents contain {' and '.join(_ROOT_MARKERS)}"
    )


REPO_ROOT = find_repo_root()
