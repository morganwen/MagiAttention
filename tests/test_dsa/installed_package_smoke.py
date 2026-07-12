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

"""Smoke-test an installed MagiAttention package from outside its checkout.

Examples::

    cd /tmp
    python /path/to/checkout/tests/test_dsa/installed_package_smoke.py

    python -m pip install --no-deps --target /tmp/magi-wheel package.whl
    cd /tmp
    python /path/to/checkout/tests/test_dsa/installed_package_smoke.py \
        --target /tmp/magi-wheel --cuda
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

PUBLIC_DSA_SYMBOLS = (
    "calc_dsa",
    "DsaPackedMeta",
    "MagiDSAInput",
    "MagiDSARuntimeMgr",
    "DsaOverlapConfig",
    "MagiDSAConfig",
    "MagiDSAYarnConfig",
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        help="Site-packages/--target directory containing the installed package.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Also run CP=1 reference forward/backward for ratios 0, 4, and 128.",
    )
    return parser.parse_args()


def _prepare_import_path(args: argparse.Namespace, checkout: Path) -> None:
    cwd = Path.cwd().resolve()
    if _is_relative_to(cwd, checkout):
        raise RuntimeError(
            f"run this smoke test outside the source checkout: cwd={cwd}"
        )
    if args.target is not None:
        target = args.target.expanduser().resolve()
        if not target.is_dir():
            raise RuntimeError(f"--target is not a directory: {target}")
        if _is_relative_to(target, checkout):
            raise RuntimeError(f"--target must be outside the checkout: {target}")
        sys.path.insert(0, str(target))


def _load_public_api(checkout: Path):
    package = importlib.import_module("magi_attention")
    api = importlib.import_module("magi_attention.api")
    package_file = Path(package.__file__).resolve()
    if _is_relative_to(package_file, checkout):
        raise RuntimeError(
            "import resolved to the source checkout instead of an installed "
            f"package: {package_file}"
        )

    missing = [name for name in PUBLIC_DSA_SYMBOLS if not hasattr(api, name)]
    if missing:
        raise RuntimeError(
            f"magi_attention.api is missing public DSA symbols: {missing}"
        )
    missing_from_all = [name for name in PUBLIC_DSA_SYMBOLS if name not in api.__all__]
    if missing_from_all:
        raise RuntimeError(
            "public DSA symbols missing from magi_attention.api.__all__: "
            f"{missing_from_all}"
        )

    experimental_spec = importlib.util.find_spec("magi_attention.experimental")
    if experimental_spec is not None:
        raise RuntimeError(
            "installed package still exposes magi_attention.experimental: "
            f"{experimental_spec}"
        )
    experimental_dir = package_file.parent / "experimental"
    if experimental_dir.exists():
        raise RuntimeError(
            "installed package still contains an experimental directory: "
            f"{experimental_dir}"
        )
    return package, api


def _make_config(api, ratio: int):
    return api.MagiDSAConfig(
        compress_ratio=ratio,
        hidden_size=8,
        q_lora_rank=8,
        softmax_scale=512**-0.5,
        backend="reference",
    )


def _check_cpu_construction(api) -> None:
    yarn = api.MagiDSAYarnConfig()
    if yarn.rotary_base != 160000.0 or yarn.scaling_factor != 16.0:
        raise RuntimeError(
            "public MagiDSAYarnConfig defaults do not match the contract"
        )
    api.DsaOverlapConfig()

    for ratio in (0, 4, 128):
        config = _make_config(api, ratio)
        runtime = api.MagiDSARuntimeMgr(config)
        if runtime.plan.cp_size != 1 or runtime.plan.cp_rank != 0:
            raise RuntimeError(f"ratio={ratio}: default runtime is not CP=1")
        if runtime.plan.compress_ratio != ratio:
            raise RuntimeError(
                f"ratio={ratio}: runtime plan stored {runtime.plan.compress_ratio}"
            )


def _make_cuda_input(api, torch, config, length: int):
    def tensor(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device="cuda").requires_grad_(True)

    return api.MagiDSAInput(
        x=tensor(length, config.hidden_size),
        qr=tensor(length, config.q_lora_rank),
        q=tensor(length, config.num_heads, config.kv_dim),
        latent_kv=tensor(length, config.kv_dim),
        sink=tensor(config.num_heads, dtype=torch.float32),
        packed_meta=api.DsaPackedMeta(torch.tensor([0, length], dtype=torch.int32)),
    )


def _check_cuda_forward_backward(api) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--cuda requested, but torch.cuda.is_available() is false")

    torch.cuda.set_device(0)
    for ratio, length in ((0, 1), (4, 4), (128, 128)):
        torch.manual_seed(20260712 + ratio)
        config = _make_config(api, ratio)
        runtime = api.MagiDSARuntimeMgr(config).cuda().train()
        dsa_input = _make_cuda_input(api, torch, config, length)

        output, kl_loss = api.calc_dsa(dsa_input, runtime)
        expected_shape = (length, config.num_heads, config.kv_dim)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"ratio={ratio}: output shape {tuple(output.shape)} != {expected_shape}"
            )
        if output.dtype != torch.bfloat16 or not torch.isfinite(output.float()).all():
            raise RuntimeError(f"ratio={ratio}: output is not finite BF16")
        if kl_loss.shape != torch.Size([]) or kl_loss.dtype != torch.float32:
            raise RuntimeError(
                f"ratio={ratio}: KL must be scalar FP32, got "
                f"{kl_loss.shape}/{kl_loss.dtype}"
            )
        if not torch.isfinite(kl_loss):
            raise RuntimeError(f"ratio={ratio}: KL is not finite")

        (output.float().square().mean() + kl_loss).backward()
        required_grads = {
            "q": dsa_input.q.grad,
            "latent_kv": dsa_input.latent_kv.grad,
            "sink": dsa_input.sink.grad,
        }
        missing_grads = [name for name, grad in required_grads.items() if grad is None]
        if missing_grads:
            raise RuntimeError(f"ratio={ratio}: missing gradients for {missing_grads}")
        if ratio > 1 and dsa_input.x.grad is None:
            raise RuntimeError(f"ratio={ratio}: compressor did not produce dx")

        del output, kl_loss, dsa_input, runtime
        torch.cuda.empty_cache()


def main() -> int:
    args = _parse_args()
    checkout = Path(__file__).resolve().parents[2]
    _prepare_import_path(args, checkout)
    package, api = _load_public_api(checkout)
    _check_cpu_construction(api)
    if args.cuda:
        _check_cuda_forward_backward(api)

    print(
        json.dumps(
            {
                "cuda_forward_backward": bool(args.cuda),
                "package": str(Path(package.__file__).resolve()),
                "public_dsa_symbols": list(PUBLIC_DSA_SYMBOLS),
                "ratios": [0, 4, 128],
                "status": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
