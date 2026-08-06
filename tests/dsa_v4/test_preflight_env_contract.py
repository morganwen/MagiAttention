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

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_preflight_records_the_required_cudnn_indexer_forward_abi() -> None:
    source = (_repository_root() / "scripts/test/preflight_env.py").read_text(
        encoding="utf-8"
    )

    assert "inspect.signature(DSA.indexer_forward_wrapper)" in source
    assert '_CUDNN_INDEXER_FORWARD_REQUIRED_KEYWORDS = ("q_causal_offsets",)' in source
    assert '"cudnn_dsa_indexer_forward_abi": indexer_forward_abi' in source
    assert "inspect.signature(DSA.indexer_top_k_wrapper)" in source
    assert '_CUDNN_INDEXER_TOPK_REQUIRED_KEYWORDS = ("return_val",)' in source
    assert '"cudnn_dsa_indexer_topk_abi": indexer_topk_abi' in source
    assert "inspect.Parameter.POSITIONAL_OR_KEYWORD" in source
    assert "inspect.Parameter.KEYWORD_ONLY" in source


def test_release_image_hard_checks_the_cudnn_indexer_forward_abi() -> None:
    source = (_repository_root() / "docker/Dockerfile.dsa-v4").read_text(
        encoding="utf-8"
    )
    compact = " ".join(source.replace("\\\n", " ").split())

    assert "inspect.signature(DSA.indexer_forward_wrapper).parameters" in compact
    assert "assert 'q_causal_offsets' in indexer_forward_parameters" in compact
    assert "inspect.signature(DSA.indexer_top_k_wrapper).parameters" in compact
    assert "assert 'return_val' in indexer_topk_parameters" in compact
    assert "inspect.Parameter.POSITIONAL_OR_KEYWORD" in compact
    assert "inspect.Parameter.KEYWORD_ONLY" in compact
    assert "python3 -m pip uninstall --yes nvidia-cutlass-dsl" in compact
    assert '"nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}"' in compact
    assert "from cutlass import Int32" in source


def test_release_image_pins_cp1_pytest_without_packaging_drift() -> None:
    repo_root = _repository_root()
    dockerfile = (repo_root / "docker/Dockerfile.dsa-v4").read_text(encoding="utf-8")
    compact = " ".join(dockerfile.replace("\\\n", " ").split())
    build = (repo_root / "scripts/image/build.sh").read_text(encoding="utf-8")
    cp1 = (repo_root / "scripts/test/run_cp1.sh").read_text(encoding="utf-8")

    assert "ARG PACKAGING_VERSION=25.0" in dockerfile
    assert "ARG PYTEST_VERSION=8.4.2" in dockerfile
    assert '"packaging==${PACKAGING_VERSION}"' in compact
    assert '"pytest==${PYTEST_VERSION}"' in compact
    assert "metadata.version('packaging') == '25.0'" in dockerfile
    assert "metadata.version('pytest') == '8.4.2'" in dockerfile
    assert 'metadata.version("packaging") == "25.0"' in build
    assert 'metadata.version("pytest") == "8.4.2"' in build
    assert 'packaging_version="25.0"' in cp1
    assert 'pytest_version="8.4.2"' in cp1
    assert 'metadata.version("packaging")' in cp1
    assert 'metadata.version("pytest")' in cp1
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cp1
    assert "PYTHONSAFEPATH=1" in cp1
    assert "--env HOME=" not in cp1
    assert '"$cache_dir/home"' not in cp1


def test_cudnn_preflight_uses_an_explicit_nonzero_torch_stream() -> None:
    source = (_repository_root() / "scripts/test/preflight_cudnn_dsa.py").read_text(
        encoding="utf-8"
    )

    assert "torch_stream = torch.cuda.Stream()" in source
    assert "if torch_stream.cuda_stream == 0:" in source
    assert "with torch.cuda.stream(torch_stream):" in source
    assert '"cuda_stream_nonzero": int(stream) != 0' in source
