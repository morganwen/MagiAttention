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

import json
import subprocess
from importlib import metadata

import cudnn
import flash_mla
import quack
import torch
from cudnn import DSA


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
    assert metadata.version("apache-tvm-ffi") == "0.1.8.post0"
    assert metadata.version("quack-kernels") == "0.4.1"
    flashmla_version = metadata.version("flash-mla")
    assert flashmla_version.endswith("+9241ae3"), flashmla_version
    assert DSA.indexer_forward_wrapper
    assert DSA.indexer_top_k_wrapper
    assert DSA.sparse_attention_backward_wrapper
    assert flash_mla.flash_mla_sparse_fwd
    assert quack.__file__ is not None

    report = {
        "cuda": torch.version.cuda,
        "cudnn_backend": cudnn.backend_version(),
        "cudnn_backend_package": metadata.version("nvidia-cudnn-cu13"),
        "cudnn_frontend": cudnn.__version__,
        "cutlass_dsl": metadata.version("nvidia-cutlass-dsl"),
        "devices": devices,
        "flashmla": flashmla_version,
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
