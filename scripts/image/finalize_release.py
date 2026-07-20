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

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a Magi-DSA v4 release artifact"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--correctness-artifact", type=Path, required=True)
    parser.add_argument("--profile-artifact", type=Path, required=True)
    parser.add_argument("--cp1-artifact", type=Path, required=True)
    parser.add_argument("--cp2-artifact", type=Path, required=True)
    return parser.parse_args()


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 60,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with status {result.returncode}: {' '.join(command)}\n{output}"
        )
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite release artifact: {path}")
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite release artifact: {destination}")
    shutil.copy2(source, destination)


def _validate_manifest(artifact: Path) -> str:
    manifest = artifact / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing SHA-256 manifest: {manifest}")
    return _run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=artifact,
        timeout_seconds=60,
    )


def _validate_correctness(path: Path, revision: str) -> dict[str, Any]:
    summary = _read_json(path / "SUMMARY.json")
    if summary.get("result") != "PASS":
        raise ValueError("installed-wheel CP8 summary did not pass")
    reports = sorted(path.glob("result_rank*.json"))
    if len(reports) != 8:
        raise ValueError(f"installed-wheel CP8 produced {len(reports)} rank reports")
    expected_version = f"1.1.1+g{revision}"
    for rank, report_path in enumerate(reports):
        report = _read_json(report_path)
        installed = report.get("installed_wheel")
        if not isinstance(installed, dict):
            raise ValueError(f"rank {rank} did not report installed-wheel provenance")
        if installed.get("source_revision") != revision:
            raise ValueError(f"rank {rank} installed-wheel revision mismatch")
        if installed.get("package_version") != expected_version:
            raise ValueError(f"rank {rank} installed-wheel version mismatch")
        if "/site-packages/" not in str(installed.get("package_path")):
            raise ValueError(f"rank {rank} did not import Magi-DSA from site-packages")
    _validate_manifest(path)
    return summary


def _validate_profile(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _read_json(path / "SUMMARY.json")
    if (
        summary.get("result") != "PASS"
        or summary.get("correctness") != "PASS"
        or summary.get("balanced_5pct_gate") is not True
        or int(summary.get("profile_records", 0)) != 160
    ):
        raise ValueError("formal five-step profile summary did not pass")
    ranges = json.loads((path / "rank_ranges.json").read_text(encoding="utf-8"))
    if not isinstance(ranges, list) or len(ranges) != 20:
        raise ValueError("formal profile rank-range grid is incomplete")
    balanced = [record for record in ranges if record.get("plan") == "balanced"]
    if len(balanced) != 10 or not all(
        record.get("gate_pass") is True for record in balanced
    ):
        raise ValueError("balanced profile does not pass all ten per-step gates")
    for plan in ("balanced", "sequential"):
        report = path / plan / f"{plan}_5steps.nsys-rep"
        sqlite = path / plan / f"{plan}_5steps.sqlite"
        if not report.is_file() or report.stat().st_size == 0:
            raise FileNotFoundError(f"missing aggregate Nsight report: {report}")
        if not sqlite.is_file() or sqlite.stat().st_size == 0:
            raise FileNotFoundError(f"missing aggregate Nsight SQLite export: {sqlite}")
    _validate_manifest(path)
    return summary, ranges


def _profile_rows(ranges: list[dict[str, Any]]) -> list[str]:
    by_key = {
        (str(record["plan"]), int(record["step"]), str(record["phase"])): record
        for record in ranges
    }
    lines = [
        "| Step | Phase | Sequential relative | Balanced relative | Balanced range ms | Gate |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for step in range(5):
        for phase in ("indexer_score", "indexer_topk"):
            sequential = by_key[("sequential", step, phase)]
            balanced = by_key[("balanced", step, phase)]
            lines.append(
                "| {step} | {phase} | {sequential:.6f} | {balanced:.6f} | "
                "{rank_range:.6f} | PASS |".format(
                    step=step,
                    phase=phase,
                    sequential=float(sequential["relative_rank_range"]),
                    balanced=float(balanced["relative_rank_range"]),
                    rank_range=float(balanced["rank_range_ms"]),
                )
            )
    return lines


def _report(
    *,
    revision: str,
    image: str,
    image_id: str,
    correctness: Path,
    profile: Path,
    cp1: Path,
    cp2: Path,
    ranges: list[dict[str, Any]],
) -> str:
    balanced = [record for record in ranges if record["plan"] == "balanced"]
    worst_score = max(
        record["relative_rank_range"]
        for record in balanced
        if record["phase"] == "indexer_score"
    )
    worst_topk = max(
        record["relative_rank_range"]
        for record in balanced
        if record["phase"] == "indexer_topk"
    )
    lines = [
        "# Magi-DSA v4 Release Report",
        "",
        "## 结论",
        "",
        "- 批准范围内的 Magi-DSA v4 implementation、CP8 correctness、正式五步 profile、",
        "  installed-wheel 镜像与追溯证据均已完成。",
        f"- Clean revision：`{revision}`。",
        f"- Release image：`{image}`（`{image_id}`）。",
        "- 本次未重复用户取消的 smoke；沿用已封存 smoke 证据，release 镜像改用 CP8 natural",
        "  forward/backward 验证 installed wheel。",
        "",
        "## 实现摘要",
        "",
        "- parameter-free `MagiDSARuntimeMgr` 与模型侧 `MagiDSALayer` 参数归属。",
        "- ratio 0/4/128、owner-local API、static plan、All2AllV/CSR、grouped Indexer、",
        "  selected-KL-owner 与 natural forward/backward。",
        "- Q15 exact raw-score tie 使用 canonical global compressed-block ID 升序，",
        "  near-equal 非 tie 行为与 backend ABI 不变。",
        "- 正式 NVTX 层级包含 8 ranks × 5 steps，并在每个 rank/step 下提供",
        "  `magi_dsa::indexer_score` 与 `magi_dsa::indexer_topk`。",
        "",
        "## 实际验证与结果路径",
        "",
        f"- CP1 历史通过证据：`{cp1}`。",
        f"- CP2 natural backward 通过证据：`{cp2}`。",
        f"- Installed-wheel CP8 natural backward：`{correctness}`。",
        f"- 正式 sequential/balanced profile：`{profile}`。",
        f"- Balanced Nsight：`{profile / 'balanced' / 'balanced_5steps.nsys-rep'}`。",
        f"- Sequential Nsight：`{profile / 'sequential' / 'sequential_5steps.nsys-rep'}`。",
        "- 完整命令见本目录 `COMMAND.txt`，构建输出见 `BUILD.log`，CP8 输出见 `CP8.log`。",
        "",
        "## 性能表",
        "",
        *_profile_rows(ranges),
        "",
        f"- Balanced score 最差 relative rank range：`{worst_score:.6f}`。",
        f"- Balanced top-k 最差 relative rank range：`{worst_topk:.6f}`。",
        "- 10/10 balanced step/phase 均满足 `<=0.05`。",
        "",
        "## 已知风险",
        "",
        "- Release wheel 为 DSA Python runtime 范围，不编译 Magi 的 legacy CUDA extension；",
        "  DSA 使用的 torch NCCL All2AllV、CuTe pack、cuDNN DSA 与 FlashMLA 路径已由",
        "  installed-wheel CP8 验证。",
        "- Nsight 参考报告由 2026.3 生成，冻结镜像为 2026.2；通过 MSA 源码和可恢复 NVTX",
        "  字符串对齐层级，未修改只读参考报告。",
        "",
        "## 未完成项",
        "",
        "- Magi-DSA v4 已批准范围内：无。",
        "- decode/cache、TP、FP8/FP4、CUDA Graph 与 selected-KV routing 属于明确排除范围。",
        "",
    ]
    return "\n".join(lines)


def _write_manifest(artifact_dir: Path) -> None:
    entries: list[str] = []
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(artifact_dir)}")
    _write_text(artifact_dir / "SHA256SUMS", "\n".join(entries) + "\n")


def main() -> None:
    args = _parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise ValueError("--revision must be an exact 40-character commit")
    artifact_dir = args.artifact_dir.resolve()
    correctness = args.correctness_artifact.resolve()
    profile = args.profile_artifact.resolve()
    cp1 = args.cp1_artifact.resolve()
    cp2 = args.cp2_artifact.resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"release artifact directory does not exist: {artifact_dir}"
        )

    repo_root = Path(
        _run(["git", "rev-parse", "--show-toplevel"], timeout_seconds=30).strip()
    )
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout_seconds=30).strip()
    if head != args.revision:
        raise ValueError(f"HEAD {head} does not match release revision {args.revision}")
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        timeout_seconds=30,
    )
    if dirty:
        raise ValueError("release finalization requires a clean worktree")

    correctness_summary = _validate_correctness(correctness, args.revision)
    profile_summary, ranges = _validate_profile(profile)

    inspect_output = _run(
        ["docker", "image", "inspect", args.image],
        cwd=repo_root,
        timeout_seconds=30,
    )
    inspect = json.loads(inspect_output)
    if not isinstance(inspect, list) or len(inspect) != 1:
        raise ValueError("docker image inspect returned an unexpected payload")
    image_record = inspect[0]
    labels = image_record.get("Config", {}).get("Labels", {})
    if labels.get("org.magi-dsa.magi-attention-revision") != args.revision:
        raise ValueError("release image revision label does not match source revision")
    if labels.get("org.magi-dsa.install-mode") != "python-wheel":
        raise ValueError("release image is not labeled as an installed-wheel image")

    environment = _run(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=5s",
            "60s",
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "bash",
            "--workdir",
            "/tmp",
            args.image,
            "-lc",
            "python3 -VV; python3 -m pip freeze; nsys --version; "
            "sha256sum /opt/magi-wheels/magi_attention-*.whl; "
            "python3 -c 'from importlib import metadata; import magi_attention; "
            'print(metadata.version("magi-attention")); print(magi_attention.__file__)\'',
        ],
        cwd=repo_root,
        timeout_seconds=70,
    )

    _write_text(artifact_dir / "SOURCE_REVISION.txt", args.revision + "\n")
    _write_text(
        artifact_dir / "SOURCE_REMOTE.txt",
        _run(["git", "remote", "get-url", "origin"], cwd=repo_root, timeout_seconds=30),
    )
    _write_text(
        artifact_dir / "SUBMODULES.txt",
        _run(
            ["git", "submodule", "status", "--recursive"],
            cwd=repo_root,
            timeout_seconds=30,
        ),
    )
    _write_text(artifact_dir / "DIRTY_STATUS.txt", dirty)
    _write_text(artifact_dir / "ENVIRONMENT.txt", environment)
    _write_text(
        artifact_dir / "HARDWARE.txt",
        _run(["nvidia-smi", "-q"], cwd=repo_root, timeout_seconds=30),
    )
    _write_json(artifact_dir / "IMAGE.json", image_record)

    references = {
        "correctness_artifact": str(correctness),
        "correctness_manifest_sha256": _sha256(correctness / "SHA256SUMS"),
        "cp1_artifact": str(cp1),
        "cp2_artifact": str(cp2),
        "profile_artifact": str(profile),
        "profile_manifest_sha256": _sha256(profile / "SHA256SUMS"),
        "profile_report_sha256": _sha256(profile / "REPORT.md"),
    }
    _write_json(artifact_dir / "REFERENCES.json", references)
    _copy(correctness / "SUMMARY.json", artifact_dir / "CP8_SUMMARY.json")
    _copy(correctness / "PHASE_AUDIT.json", artifact_dir / "CP8_PHASE_AUDIT.json")
    _copy(correctness / "SEEDS.json", artifact_dir / "CP8_SEEDS.json")
    _copy(profile / "SUMMARY.json", artifact_dir / "PROFILE_SUMMARY.json")
    _copy(profile / "WORKLOAD.json", artifact_dir / "PROFILE_WORKLOAD.json")
    _copy(profile / "rank_ranges.json", artifact_dir / "PROFILE_RANK_RANGES.json")
    _copy(profile / "REPORT.md", artifact_dir / "PROFILE_REPORT.md")
    _write_text(
        artifact_dir / "REPORT.md",
        _report(
            revision=args.revision,
            image=args.image,
            image_id=str(image_record["Id"]),
            correctness=correctness,
            profile=profile,
            cp1=cp1,
            cp2=cp2,
            ranges=ranges,
        ),
    )
    summary = {
        "balanced_5pct_gate": profile_summary["balanced_5pct_gate"],
        "correctness": correctness_summary["result"],
        "image": args.image,
        "image_id": image_record["Id"],
        "installed_wheel_cp8": "PASS",
        "profile": profile_summary["result"],
        "result": "PASS",
        "revision": args.revision,
    }
    _write_json(artifact_dir / "SUMMARY.json", summary)
    _write_manifest(artifact_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
