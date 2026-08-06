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

import inspect
import json
import os
import subprocess
from importlib import metadata

import cudnn
import flash_mla
import quack
import torch
from cudnn import DSA
from cutlass import Int32

_FLASHMLA_BASE_REVISION = "9241ae3ef9bac614dd25e45e507e089f888280e0"
_FLASHMLA_DUAL_LSE_PATCH_REVISION = "13d173ac48abd8ec88a4e742bcaaa59c5ccf4ece"
_FLASHMLA_PRO_H128_PATCH_REVISION = "b7643bd54521f563b839b98289b5cd048c062ba2"
_PRO_FLASHMLA_ABI = {
    "head_dim": 512,
    "indexer_topk": 1024,
    "num_query_heads": 128,
    "total_topk": 1152,
    "window_topk": 128,
}
_CUDNN_INDEXER_FORWARD_REQUIRED_KEYWORDS = ("q_causal_offsets",)
_CUDNN_INDEXER_TOPK_REQUIRED_KEYWORDS = ("return_val",)


def _command_version(*command: str) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (result.stdout or result.stderr).strip()


def main() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 8
    devices = []
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        assert capability == (10, 3), (index, capability)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(capability),
            }
        )

    assert cudnn.__version__ == "1.26.0"
    assert cudnn.backend_version() == 92400
    assert metadata.version("nvidia-cudnn-cu13") == "9.24.0.43"
    assert metadata.version("nvidia-cutlass-dsl") == "4.5.0"
    assert metadata.version("nvidia-cutlass-dsl-libs-base") == "4.5.0"
    assert metadata.version("nvidia-cutlass-dsl-libs-cu13") == "4.5.0"
    assert Int32
    assert metadata.version("apache-tvm-ffi") == "0.1.8.post0"
    assert metadata.version("quack-kernels") == "0.4.1"
    flashmla_version = metadata.version("flash-mla")
    assert flashmla_version.endswith("+9241ae3"), flashmla_version
    assert os.environ["MAGI_DSA_FLASHMLA_BASE_REVISION"] == _FLASHMLA_BASE_REVISION
    assert (
        os.environ["MAGI_DSA_FLASHMLA_DUAL_LSE_PATCH_REVISION"]
        == _FLASHMLA_DUAL_LSE_PATCH_REVISION
    )
    assert (
        os.environ["MAGI_DSA_FLASHMLA_PRO_H128_PATCH_REVISION"]
        == _FLASHMLA_PRO_H128_PATCH_REVISION
    )
    assert DSA.indexer_forward_wrapper
    assert DSA.indexer_top_k_wrapper
    assert DSA.indexer_backward_wrapper
    assert DSA.sparse_attention_backward_wrapper
    indexer_forward_signature = inspect.signature(DSA.indexer_forward_wrapper)
    indexer_forward_parameters = indexer_forward_signature.parameters
    for parameter_name in _CUDNN_INDEXER_FORWARD_REQUIRED_KEYWORDS:
        assert parameter_name in indexer_forward_parameters, (
            parameter_name,
            indexer_forward_signature,
        )
        assert indexer_forward_parameters[parameter_name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    indexer_forward_abi = {
        "parameters": list(indexer_forward_parameters),
        "required_keywords": list(_CUDNN_INDEXER_FORWARD_REQUIRED_KEYWORDS),
        "signature": str(indexer_forward_signature),
    }
    indexer_topk_signature = inspect.signature(DSA.indexer_top_k_wrapper)
    indexer_topk_parameters = indexer_topk_signature.parameters
    for parameter_name in _CUDNN_INDEXER_TOPK_REQUIRED_KEYWORDS:
        assert parameter_name in indexer_topk_parameters, (
            parameter_name,
            indexer_topk_signature,
        )
        assert indexer_topk_parameters[parameter_name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    indexer_topk_abi = {
        "parameters": list(indexer_topk_parameters),
        "required_keywords": list(_CUDNN_INDEXER_TOPK_REQUIRED_KEYWORDS),
        "signature": str(indexer_topk_signature),
    }
    assert flash_mla.flash_mla_sparse_fwd
    flashmla_signature = inspect.signature(flash_mla.flash_mla_sparse_fwd)
    assert "indexer_topk" in flashmla_signature.parameters
    assert flashmla_signature.parameters["indexer_topk"].default == 0
    flashmla_doc = inspect.getdoc(flash_mla.flash_mla_sparse_fwd) or ""
    assert "0/512/1024/2048" in flashmla_doc
    assert quack.__file__ is not None

    report = {
        "cuda": torch.version.cuda,
        "cudnn_backend": cudnn.backend_version(),
        "cudnn_backend_package": metadata.version("nvidia-cudnn-cu13"),
        "cudnn_dsa_indexer_forward_abi": indexer_forward_abi,
        "cudnn_dsa_indexer_topk_abi": indexer_topk_abi,
        "cudnn_frontend": cudnn.__version__,
        "cutlass_dsl": metadata.version("nvidia-cutlass-dsl"),
        "devices": devices,
        "flashmla": flashmla_version,
        "flashmla_base_revision": _FLASHMLA_BASE_REVISION,
        "flashmla_dual_lse_patch_revision": _FLASHMLA_DUAL_LSE_PATCH_REVISION,
        "flashmla_pro_h128_patch_revision": _FLASHMLA_PRO_H128_PATCH_REVISION,
        "flashmla_pro_required_abi": _PRO_FLASHMLA_ABI,
        "nccl": torch.cuda.nccl.version(),
        "nsight_systems": _command_version("nsys", "--version"),
        "nvcc": _command_version("nvcc", "--version"),
        "quack": metadata.version("quack-kernels"),
        "torch": torch.__version__,
        "tvm_ffi": metadata.version("apache-tvm-ffi"),
    }
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
