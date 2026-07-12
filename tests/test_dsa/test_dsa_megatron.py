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

"""Forward parity against the frozen Megatron DSA V4 implementation."""

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from magi_attention.api import (
    MagiDSAConfig,
    MagiDSAYarnConfig,
)
from magi_attention.dsa import DSAv4Compressor

_MEGATRON_REVISION = "c6449f0b23be397449f21c0967c5fc90785e55ea"
_DEFAULT_MEGATRON_PATHS = (
    Path("/home/scratch.wewen_gpu/megatron-lm"),
    Path("/ws/megatron-lm"),
)


def _megatron_path() -> Path:
    configured = os.environ.get("MAGI_DSA_MEGATRON_PATH")
    candidates = (
        (Path(configured).expanduser(),) if configured else _DEFAULT_MEGATRON_PATHS
    )
    path = next(
        (
            candidate.resolve()
            for candidate in candidates
            if (candidate / "megatron" / "core" / "__init__.py").is_file()
        ),
        None,
    )
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        pytest.skip(
            "Megatron DSA source is unavailable; set MAGI_DSA_MEGATRON_PATH "
            f"to the frozen checkout (searched: {searched})"
        )

    try:
        revision = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        pytest.skip(f"cannot determine Megatron revision at {path}: {error}")
    if revision != _MEGATRON_REVISION:
        pytest.skip(
            f"Megatron revision mismatch at {path}: expected "
            f"{_MEGATRON_REVISION}, found {revision or '<empty>'}"
        )
    return path


def _import_megatron_dependencies():
    path = _megatron_path()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

    try:
        import transformer_engine.pytorch  # noqa: F401
        from megatron.core import parallel_state
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.models.common.embeddings import RotaryEmbedding
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.experimental_attention_variant.csa import (
            Compressor,
            CompressorSubmodules,
        )
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.transformer.transformer_config import MLATransformerConfig
    except (ImportError, OSError) as error:
        pytest.skip(f"Megatron DSA dependency import failed: {error}")

    return {
        "parallel_state": parallel_state,
        "TELinear": TELinear,
        "TENorm": TENorm,
        "RotaryEmbedding": RotaryEmbedding,
        "ProcessGroupCollection": ProcessGroupCollection,
        "model_parallel_cuda_manual_seed": model_parallel_cuda_manual_seed,
        "Compressor": Compressor,
        "CompressorSubmodules": CompressorSubmodules,
        "ModuleSpec": ModuleSpec,
        "MLATransformerConfig": MLATransformerConfig,
    }


@contextmanager
def _single_rank_megatron_context(megatron):
    if not torch.cuda.is_available():
        pytest.skip("Megatron DSA forward parity requires a CUDA GPU")
    if torch.distributed.is_initialized():
        pytest.skip("Megatron DSA parity requires an uninitialized process group")

    store_path = tempfile.mktemp(prefix="magi-dsa-megatron-")
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{store_path}",
        rank=0,
        world_size=1,
    )
    try:
        megatron["parallel_state"].initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        megatron["model_parallel_cuda_manual_seed"](123)
        process_groups = megatron["ProcessGroupCollection"].use_mpu_process_groups(
            required_pgs=["tp", "cp"]
        )
        rotary = megatron["RotaryEmbedding"](
            32,
            rotary_percent=1.0,
            rotary_base=10000,
            cp_group=process_groups.cp,
        )
        yield process_groups, rotary
    finally:
        megatron["parallel_state"].destroy_model_parallel()
        torch.distributed.destroy_process_group()
        try:
            Path(store_path).unlink()
        except FileNotFoundError:
            pass


def _make_megatron_config(megatron):
    return megatron["MLATransformerConfig"](
        num_layers=2,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        csa_compress_ratios=[4, 4],
        csa_window_size=8,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=8,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
    )


def _make_magi_config():
    return MagiDSAConfig(
        compress_ratio=4,
        hidden_size=256,
        q_lora_rank=64,
        softmax_scale=64**-0.5,
        num_heads=16,
        kv_dim=64,
        rope_dim=32,
        window_size=8,
        topk=8,
        indexer_heads=8,
        indexer_dim=64,
        norm_eps=1e-5,
        yarn=MagiDSAYarnConfig(rotary_base=10000.0, scaling_factor=1.0),
    )


def _make_megatron_compressor(megatron, config, process_groups, rotary):
    submodules = megatron["CompressorSubmodules"](
        linear_wkv=megatron["ModuleSpec"](module=megatron["TELinear"]),
        linear_wgate=megatron["ModuleSpec"](module=megatron["TELinear"]),
        norm=megatron["ModuleSpec"](module=megatron["TENorm"]),
    )
    return megatron["Compressor"](
        config=config,
        submodules=submodules,
        compress_ratio=4,
        head_dim=64,
        rotate=False,
        rotary_pos_emb=rotary,
        pg_collection=process_groups,
    ).cuda()


def test_ratio4_compressor_forward_matches_frozen_megatron():
    """Cover ratio-4 overlap and the dropped, non-divisible sequence tail."""
    megatron = _import_megatron_dependencies()
    with _single_rank_megatron_context(megatron) as (process_groups, rotary):
        megatron_compressor = _make_megatron_compressor(
            megatron,
            _make_megatron_config(megatron),
            process_groups,
            rotary,
        )
        magi_compressor = DSAv4Compressor(
            _make_magi_config(), head_dim=64, rotate=False
        ).cuda()

        with torch.no_grad():
            magi_compressor.linear_wkv.weight.copy_(
                megatron_compressor.linear_wkv.weight
            )
            magi_compressor.linear_wgate.weight.copy_(
                megatron_compressor.linear_wgate.weight
            )
            magi_compressor.ape.copy_(megatron_compressor.ape)
            magi_compressor.norm.weight.copy_(megatron_compressor.norm.weight)

        torch.manual_seed(20260712)
        hidden_states = torch.randn(10, 2, 256, device="cuda", dtype=torch.bfloat16)
        expected = megatron_compressor(hidden_states)
        actual = magi_compressor(hidden_states)

        assert expected.shape == actual.shape == (2, 2, 64)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5)
