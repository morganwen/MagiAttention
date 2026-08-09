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

import json
import os
import time
from types import TracebackType
from typing import Literal

import torch

_LOG_PHASES = os.environ.get("MAGI_DSA_PHASE_LOG", "0") == "1"


def _phase_record(name: str, event: str, error: str | None = None) -> None:
    payload: dict[str, object] = {
        "event": event,
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "monotonic_ns": time.monotonic_ns(),
        "name": name,
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", "-1")),
        "record_type": "magi_dsa_phase",
        "wall_time_ns": time.time_ns(),
    }
    if error is not None:
        payload["error"] = error
    print(f"MAGI_DSA_PHASE {json.dumps(payload, sort_keys=True)}", flush=True)


class dsa_phase:
    """Emit an NVTX range and optional synchronized per-rank boundary records."""

    def __init__(self, name: str):
        self.name = name
        self._range_pushed = False

    def __enter__(self) -> dsa_phase:
        if _LOG_PHASES:
            _phase_record(self.name, "begin")
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"magi_dsa::phase::{self.name}")
        self._range_pushed = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del traceback
        sync_error: BaseException | None = None
        if _LOG_PHASES and exc_type is None:
            try:
                torch.cuda.synchronize()
            except (
                RuntimeError
            ) as error:  # pragma: no cover - exercised only on a CUDA failure
                sync_error = error
        if self._range_pushed:
            torch.cuda.nvtx.range_pop()
            self._range_pushed = False
        if _LOG_PHASES:
            error_value = exc_value if exc_value is not None else sync_error
            _phase_record(
                self.name,
                "end" if error_value is None else "error",
                None if error_value is None else repr(error_value),
            )
        if sync_error is not None:
            raise sync_error
        return False


__all__ = ["dsa_phase"]
