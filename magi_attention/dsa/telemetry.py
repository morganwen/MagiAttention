# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Invocation-local CUDA phase telemetry for Magi_DSA benchmarks.

The runtime never stores events on a shared module. A benchmark may activate
``DsaTelemetry`` around one forward/backward invocation; production calls pay
only the NVTX range cost needed for timeline profiling.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

import torch

from magi_attention.utils import nvtx


@dataclass(frozen=True)
class _PhaseEvents:
    name: str
    start: torch.cuda.Event
    end: torch.cuda.Event


_ACTIVE_TELEMETRY: ContextVar["DsaTelemetry | None"] = ContextVar(
    "magi_dsa_active_telemetry", default=None
)


class DsaTelemetry:
    """Collect CUDA-event durations for one invocation on the current rank."""

    def __init__(self) -> None:
        self._records: list[_PhaseEvents] = []
        self._token: Token[DsaTelemetry | None] | None = None

    def __enter__(self) -> "DsaTelemetry":
        if self._token is not None:
            raise RuntimeError("DsaTelemetry cannot be entered twice")
        self._token = _ACTIVE_TELEMETRY.set(self)
        return self

    def __exit__(self, *excinfo) -> None:
        assert self._token is not None
        _ACTIVE_TELEMETRY.reset(self._token)
        self._token = None

    def _begin(self, name: str) -> _PhaseEvents:
        record = _PhaseEvents(
            name=name,
            start=torch.cuda.Event(enable_timing=True),
            end=torch.cuda.Event(enable_timing=True),
        )
        record.start.record()
        self._records.append(record)
        return record

    @staticmethod
    def _end(record: _PhaseEvents) -> None:
        record.end.record()

    def durations_ms(self, *, synchronize: bool = True) -> dict[str, float]:
        """Return summed phase durations, preserving repeated fragment calls."""

        if self._token is not None:
            raise RuntimeError("exit DsaTelemetry before reading durations")
        if synchronize:
            torch.cuda.synchronize()
        totals: dict[str, float] = {}
        for record in self._records:
            totals[record.name] = totals.get(record.name, 0.0) + float(
                record.start.elapsed_time(record.end)
            )
        return totals


class dsa_phase:
    """Emit an NVTX range and optionally collect CUDA events for one phase."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._nvtx = nvtx.add_nvtx_event(f"magi_dsa::{name}")
        self._telemetry: DsaTelemetry | None = None
        self._record: _PhaseEvents | None = None

    def __enter__(self) -> "dsa_phase":
        self._nvtx.__enter__()
        self._telemetry = _ACTIVE_TELEMETRY.get()
        if self._telemetry is not None:
            self._record = self._telemetry._begin(self.name)
        return self

    def __exit__(self, *excinfo) -> None:
        try:
            if self._telemetry is not None and self._record is not None:
                self._telemetry._end(self._record)
        finally:
            self._nvtx.__exit__(*excinfo)


__all__ = ["DsaTelemetry", "dsa_phase"]
