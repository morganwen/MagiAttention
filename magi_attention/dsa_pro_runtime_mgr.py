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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import torch
import torch.distributed as dist

from magi_attention.dsa_config import (
    DsaRatio,
    DsaStructuralLayoutConfig,
    MagiDSAProModelSpec,
)
from magi_attention.dsa_runtime_mgr import (
    DsaExecutionHandle,
    DsaRuntimeCounters,
    MagiDSARuntimeMgr,
)
from magi_attention.dsa_types import (
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)

if TYPE_CHECKING:
    from magi_attention.dsa_model_adapter import MagiDSAProjector


@dataclass(frozen=True, eq=False)
class MagiDSAProExecutionBundle:
    """Ratio-specific handles backed by one shared final Query layout."""

    csa: DsaExecutionHandle
    hca: DsaExecutionHandle
    pro_runtime_identity: int = field(repr=False)

    def __post_init__(self) -> None:
        if self.csa.plan.ratio != 4 or self.hca.plan.ratio != 128:
            raise ValueError("Pro execution bundle requires CSA and HCA handles")
        if self.csa.plan.query_layout_hash != self.hca.plan.query_layout_hash:
            raise ValueError("Pro execution bundle handles have different layouts")

    @property
    def query_layout_hash(self) -> str:
        return self.csa.plan.query_layout_hash

    @property
    def local_token_count(self) -> int:
        return self.csa.device_plan.local_token_count


class MagiDSAProRuntimeMgr:
    """Parameter-free runtime pair for the 30-CSA/31-HCA Pro main stack."""

    def __init__(
        self,
        cp_group: dist.ProcessGroup | None = None,
        *,
        model_spec: MagiDSAProModelSpec | None = None,
        structural_layout_config: DsaStructuralLayoutConfig | None = None,
        max_cached_handles: int = 4,
    ) -> None:
        self._identity = id(self)
        self.model_spec = model_spec or MagiDSAProModelSpec()
        self.structural_layout_config = (
            structural_layout_config or DsaStructuralLayoutConfig()
        )
        self.cp_group = cp_group
        self.csa_runtime = MagiDSARuntimeMgr(
            self.model_spec.make_layer_config(self.model_spec.csa_layer_ids[0]),
            cp_group,
            policy="structural_balanced",
            structural_layout_config=self.structural_layout_config,
            max_cached_handles=max_cached_handles,
        )
        self.hca_runtime = MagiDSARuntimeMgr(
            self.model_spec.make_layer_config(self.model_spec.hca_layer_ids[0]),
            cp_group,
            policy="structural_balanced",
            structural_layout_config=self.structural_layout_config,
            max_cached_handles=max_cached_handles,
        )

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        del recurse
        return iter(())

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        del prefix, recurse
        return iter(())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    @property
    def counters(self) -> dict[str, DsaRuntimeCounters]:
        return {
            "csa": self.csa_runtime.counters,
            "hca": self.hca_runtime.counters,
        }

    def clear_handle_cache(self) -> None:
        self.csa_runtime.clear_handle_cache()
        self.hca_runtime.clear_handle_cache()

    def prepare_execution(
        self,
        packed_meta: MagiDSAPackedMeta,
        device: torch.device | str,
        *,
        local_token_capacity: int,
        health_check: bool = True,
    ) -> MagiDSAProExecutionBundle:
        csa = self.csa_runtime.prepare_execution(
            packed_meta,
            device,
            local_token_capacity=local_token_capacity,
            health_check=health_check,
        )
        hca = self.hca_runtime.prepare_execution(
            packed_meta,
            device,
            local_token_capacity=local_token_capacity,
            health_check=health_check,
        )
        self._validate_shared_layout(csa, hca)
        return MagiDSAProExecutionBundle(
            csa=csa,
            hca=hca,
            pro_runtime_identity=self._identity,
        )

    def _validate_bundle(self, bundle: MagiDSAProExecutionBundle) -> None:
        if not isinstance(bundle, MagiDSAProExecutionBundle):
            raise TypeError("bundle must be a MagiDSAProExecutionBundle")
        if bundle.pro_runtime_identity != self._identity:
            raise ValueError("execution bundle belongs to a different Pro runtime")
        if bundle.csa.runtime_identity != self.csa_runtime.runtime_identity:
            raise ValueError("CSA handle belongs to a different runtime")
        if bundle.hca.runtime_identity != self.hca_runtime.runtime_identity:
            raise ValueError("HCA handle belongs to a different runtime")
        if bundle.csa.plan.query_layout_hash != bundle.hca.plan.query_layout_hash:
            raise ValueError("execution bundle no longer has one shared layout")

    @staticmethod
    def _validate_shared_layout(
        csa: DsaExecutionHandle,
        hca: DsaExecutionHandle,
    ) -> None:
        if csa.plan.ratio != 4 or hca.plan.ratio != 128:
            raise ValueError("Pro execution bundle requires one CSA and one HCA plan")
        if (
            csa.plan.cu_seqlens != hca.plan.cu_seqlens
            or csa.plan.source_token_counts != hca.plan.source_token_counts
            or csa.plan.query_token_counts != hca.plan.query_token_counts
            or csa.plan.query_layout_hash != hca.plan.query_layout_hash
        ):
            raise RuntimeError("CSA and HCA plans do not share one Query layout")
        csa_rank = csa.plan.rank_plans[csa.rank]
        hca_rank = hca.plan.rank_plans[hca.rank]
        if (
            csa_rank.query_fragments != hca_rank.query_fragments
            or csa_rank.local_query_global_rows != hca_rank.local_query_global_rows
            or csa_rank.local_q_sample_ids != hca_rank.local_q_sample_ids
            or csa_rank.local_q_positions != hca_rank.local_q_positions
            or csa_rank.token_layout_route != hca_rank.token_layout_route
        ):
            raise RuntimeError("CSA and HCA rank metadata disagree on Query layout")

    def runtime_for_ratio(self, ratio: DsaRatio) -> MagiDSARuntimeMgr:
        if ratio == 4:
            return self.csa_runtime
        if ratio == 128:
            return self.hca_runtime
        raise ValueError("the Pro main stack contains no window-only layer")

    def handle_for_layer(
        self,
        layer_id: int,
        bundle: MagiDSAProExecutionBundle,
    ) -> DsaExecutionHandle:
        if not 0 <= layer_id < self.model_spec.main_layer_count:
            raise IndexError(
                f"main layer_id must be in [0, {self.model_spec.main_layer_count})"
            )
        self._validate_bundle(bundle)
        ratio = self.model_spec.main_compress_ratios[layer_id]
        return bundle.csa if ratio == 4 else bundle.hca

    def layout_source_hidden_once(
        self,
        source_x: torch.Tensor,
        bundle: MagiDSAProExecutionBundle,
    ) -> torch.Tensor:
        """Apply TOKEN_LAYOUT exactly once before the 61-layer main stack."""

        self._validate_bundle(bundle)
        return self.csa_runtime.layout_hidden(source_x, bundle.csa)

    def get_position_ids(
        self,
        bundle: MagiDSAProExecutionBundle,
    ) -> torch.Tensor:
        self._validate_bundle(bundle)
        return self.csa_runtime.get_position_ids(bundle.csa)

    def project_local_input(
        self,
        layer_id: int,
        local_x: torch.Tensor,
        sink: torch.Tensor,
        packed_meta: MagiDSAPackedMeta,
        bundle: MagiDSAProExecutionBundle,
        projector: MagiDSAProjector,
        *,
        detach_indexer_trunk: bool = False,
    ) -> MagiDSAInput:
        """Build one layer input without repeating the outer TOKEN_LAYOUT."""

        runtime = self.runtime_for_ratio(
            self.model_spec.make_layer_config(layer_id).ratio
        )
        handle = self.handle_for_layer(layer_id, bundle)
        from magi_attention.dsa_model_adapter import project_local_dsa_input

        return project_local_dsa_input(
            local_x,
            sink,
            packed_meta,
            runtime,
            handle,
            projector,
            detach_indexer_trunk=detach_indexer_trunk,
        )

    def calc_layer(
        self,
        layer_id: int,
        layer: torch.nn.Module,
        dsa_input: MagiDSAInput,
        bundle: MagiDSAProExecutionBundle,
    ) -> MagiDSAForwardResult:
        from magi_attention.dsa_layer import MagiDSALayer

        if not isinstance(layer, MagiDSALayer):
            raise TypeError("layer must be a MagiDSALayer")
        expected = self.model_spec.make_layer_config(layer_id)
        if layer.config != expected:
            raise ValueError("layer config does not match the official Pro schedule")
        if layer.layer_id != layer_id:
            raise ValueError("layer_id does not match the bound Pro parameter owner")
        runtime = self.runtime_for_ratio(expected.ratio)
        handle = self.handle_for_layer(layer_id, bundle)
        return runtime.calc_dsa(layer, dsa_input, handle)


__all__ = ["MagiDSAProExecutionBundle", "MagiDSAProRuntimeMgr"]
