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

"""Stable implementation package for DeepSeek V4 hybrid sparse attention.

Application code should use :mod:`magi_attention.api`; this package contains
the reusable implementation behind that public surface.
"""

__api_version__ = "1.0.0"

from .attention import MagiDSAV4
from .compressor import DSAv4Compressor
from .config import (
    MagiDSAConfig,
    MagiDSAV4Config,
    MagiDSAV4YarnConfig,
    MagiDSAYarnConfig,
)
from .indexer import DSAv4Indexer
from .telemetry import DsaTelemetry

__all__ = [
    "MagiDSAV4",
    "MagiDSAConfig",
    "MagiDSAYarnConfig",
    "MagiDSAV4Config",
    "MagiDSAV4YarnConfig",
    "DSAv4Compressor",
    "DSAv4Indexer",
    "DsaTelemetry",
]
