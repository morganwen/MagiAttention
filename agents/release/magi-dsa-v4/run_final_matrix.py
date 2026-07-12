#!/usr/bin/env python3
"""Run the ordered Magi_DSA V4 release matrix and seal its evidence.

The runner intentionally uses argv arrays rather than a shell, starts every case
in its own process session, enforces a TERM/KILL timeout, parses pytest's built-in
JUnit XML, and rejects every skip.  It is designed to run *inside* the immutable
production image built by ``build_image.sh``; the repository and tests are baked
into that image, while only the result directory may be bind-mounted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class MatrixError(RuntimeError):
    """An acceptance-contract violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MatrixError(f"{path} must contain a JSON object")
    return value


def expand(value: str, context: Mapping[str, str]) -> str:
    try:
        return value.format_map(context)
    except KeyError as error:
        raise MatrixError(f"unknown template key {error.args[0]!r} in {value!r}") from error


def validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != SCHEMA_VERSION:
        raise MatrixError(
            f"unsupported matrix schema {matrix.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    requirements = matrix.get("requirements")
    if not isinstance(requirements, dict):
        raise MatrixError("matrix.requirements must be an object")
    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        raise MatrixError("matrix.cases must be a non-empty array")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise MatrixError(f"matrix.cases[{index}] must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise MatrixError(f"invalid case id at index {index}: {case_id!r}")
        if case_id in seen:
            raise MatrixError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        command = case.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise MatrixError(f"case {case_id!r} command must be a non-empty argv array")
        timeout = case.get("timeout_seconds")
        if not isinstance(timeout, int) or timeout <= 0:
            raise MatrixError(f"case {case_id!r} timeout_seconds must be positive")
        expected = case.get("expected_exit_codes", [0])
        if not isinstance(expected, list) or not expected or not all(
            isinstance(code, int) for code in expected
        ):
            raise MatrixError(f"case {case_id!r} expected_exit_codes must be integers")
        if case.get("junit") and (
            not isinstance(case.get("min_tests"), int) or case["min_tests"] <= 0
        ):
            raise MatrixError(f"JUnit case {case_id!r} must set a positive min_tests")

    checks = matrix.get("sha256_checks", [])
    if not isinstance(checks, list):
        raise MatrixError("matrix.sha256_checks must be an array")
    for check in checks:
        if not isinstance(check, dict):
            raise MatrixError("each SHA256 check must be an object")
        if not isinstance(check.get("path"), str) or not SHA256_RE.fullmatch(
            str(check.get("sha256", ""))
        ):
            raise MatrixError(f"invalid SHA256 check: {check!r}")


def verify_sha256_manifest(manifest: Path) -> dict[str, str]:
    """Verify a GNU sha256sum-style manifest with paths relative to its directory."""

    verified: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MatrixError(f"cannot read SHA256 manifest {manifest}: {error}") from error
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise MatrixError(f"malformed {manifest}:{line_number}: {raw_line!r}")
        expected, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise MatrixError(f"unsafe manifest path {relative!r}")
        target = manifest.parent / relative_path
        if not target.is_file():
            raise MatrixError(f"manifest target is missing: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise MatrixError(
                f"SHA256 mismatch for {target}: expected {expected}, got {actual}"
            )
        verified[relative] = actual
    if not verified:
        raise MatrixError(f"SHA256 manifest {manifest} contains no entries")
    return verified


def terminate_process_group(
    process: subprocess.Popen[Any], *, grace_seconds: float = 5.0
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"term_sent_at": None, "kill_sent_at": None}
    if process.poll() is not None:
        return evidence
    try:
        os.killpg(process.pid, signal.SIGTERM)
        evidence["term_sent_at"] = utc_now()
    except ProcessLookupError:
        return evidence
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            evidence["kill_sent_at"] = utc_now()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=max(grace_seconds, 1.0))
    except subprocess.TimeoutExpired:
        pass
    return evidence


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    log_path: Path,
) -> dict[str, Any]:
    if not cwd.is_dir():
        raise MatrixError(f"command working directory does not exist: {cwd}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = utc_now()
    started = time.monotonic()
    timed_out = False
    termination: dict[str, Any] = {}
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination = terminate_process_group(process)
            return_code = process.poll()
            if return_code is None:
                return_code = -signal.SIGKILL
    return {
        "argv": argv,
        "cwd": str(cwd),
        "pid": process.pid,
        "started_at": started_wall,
        "finished_at": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "return_code": return_code,
        "termination": termination,
        "log": str(log_path),
    }


def parse_junit(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise MatrixError(f"JUnit report was not produced: {path}")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise MatrixError(f"invalid JUnit XML {path}: {error}") from error

    def attributes(element: ET.Element) -> dict[str, int]:
        result: dict[str, int] = {}
        for name in ("tests", "failures", "errors", "skipped"):
            raw = element.attrib.get(name, "0")
            try:
                result[name] = int(raw)
            except ValueError as error:
                raise MatrixError(f"JUnit {path} has invalid {name}={raw!r}") from error
        return result

    if "tests" in root.attrib:
        counts = attributes(root)
    else:
        suites = [child for child in root if child.tag.rsplit("}", 1)[-1] == "testsuite"]
        if not suites and root.tag.rsplit("}", 1)[-1] == "testsuite":
            suites = [root]
        if not suites:
            raise MatrixError(f"JUnit {path} contains no testsuite")
        counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for name, value in attributes(suite).items():
                counts[name] += value
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    return counts


def topology_is_all_nvlink(topology: str, gpu_count: int, prefix: str) -> bool:
    rows: list[list[str]] = []
    for line in topology.splitlines():
        fields = line.split()
        # Exclude the header, which also starts with GPU0/GPU1.  A data row's
        # first topology cell is X/NV*/PIX/PXB/PHB/NODE/SYS, never another GPU*.
        if (
            len(fields) >= 2
            and re.fullmatch(r"GPU\d+", fields[0])
            and re.fullmatch(r"X|NV\d+|PIX|PXB|PHB|NODE|SYS", fields[1])
        ):
            rows.append(fields)
    if len(rows) != gpu_count:
        return False
    for row_index, fields in enumerate(rows):
        if len(fields) < gpu_count + 1:
            return False
        links = fields[1 : gpu_count + 1]
        for column, link in enumerate(links):
            if column == row_index:
                if link != "X":
                    return False
            elif not link.startswith(prefix):
                return False
    return True


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise MatrixError(f"cannot determine Git revision at {path}: {error}") from error


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise MatrixError(f"required distribution is not installed: {name}") from error


def run_preflight(
    matrix: dict[str, Any],
    *,
    context: Mapping[str, str],
    expected_revision: str | None,
    image_id: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    requirements = matrix["requirements"]
    evidence: dict[str, Any] = {
        "started_at": utc_now(),
        "image_id": image_id,
        "expected_revision": expected_revision,
        "environment": {},
    }

    if image_id is None or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise MatrixError(
            "--image-id (or MAGI_DSA_IMAGE_ID) must be an immutable sha256 image ID"
        )
    if requirements.get("nvshmem_symmetric_size") == "unset" and (
        "NVSHMEM_SYMMETRIC_SIZE" in os.environ
    ):
        raise MatrixError("NVSHMEM_SYMMETRIC_SIZE must be absent, not merely empty")

    repo_root = Path(context["repo_root"]).resolve()
    release_dir = Path(context["release_dir"]).resolve()
    megatron_path = Path(context["megatron_path"]).resolve()
    if not (repo_root / "tests/test_dsa/test_dsa_cp.py").is_file():
        raise MatrixError(f"baked acceptance source is incomplete: {repo_root}")
    if not release_dir.is_dir():
        raise MatrixError(f"release directory is missing: {release_dir}")
    if requirements.get("forbid_source_bind") and os.path.ismount(repo_root):
        raise MatrixError(f"source checkout is a mount point, expected an image layer: {repo_root}")

    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.is_file():
        shutil.copyfile(mountinfo, output_dir / "mountinfo.txt")

    try:
        import torch
    except ImportError as error:
        raise MatrixError(f"PyTorch import failed: {error}") from error
    gpu_count = torch.cuda.device_count()
    if gpu_count != requirements["gpu_count"]:
        raise MatrixError(f"expected exactly 8 visible GPUs, found {gpu_count}")
    gpu_evidence = []
    expected_capability = tuple(requirements["compute_capability"])
    for index in range(gpu_count):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        if requirements["gpu_name_contains"] not in name:
            raise MatrixError(f"GPU {index} is not a B300 acceptance device: {name}")
        if capability != expected_capability:
            raise MatrixError(
                f"GPU {index} capability is {capability}, expected {expected_capability}"
            )
        gpu_evidence.append(
            {"index": index, "name": name, "compute_capability": list(capability)}
        )
    evidence["gpus"] = gpu_evidence

    try:
        topology = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise MatrixError(f"cannot inspect GPU topology: {error}") from error
    (output_dir / "nvidia-smi-topo.txt").write_text(topology, encoding="utf-8")
    if not topology_is_all_nvlink(
        topology, gpu_count, requirements["all_gpu_links_prefix"]
    ):
        raise MatrixError("GPU0..7 are not one all-NVLink group")

    required_public_names = (
        "DsaOverlapConfig",
        "DsaPackedMeta",
        "MagiDSAConfig",
        "MagiDSAInput",
        "MagiDSARuntimeMgr",
        "MagiDSAYarnConfig",
        "calc_dsa",
    )
    api = importlib.import_module("magi_attention.api")
    missing_public = [name for name in required_public_names if not hasattr(api, name)]
    if missing_public:
        raise MatrixError(f"installed public API is incomplete: {missing_public}")
    package = importlib.import_module("magi_attention")
    package_file = Path(str(package.__file__)).resolve()
    if package_file.is_relative_to(repo_root):
        raise MatrixError(
            f"magi_attention is source-shadowed by {package_file}; expected installed wheel"
        )
    evidence["installed_package"] = {
        "path": str(package_file),
        "version": getattr(package, "__version__", None),
    }

    native_modules = (
        "nvidia.nvshmem",
        "magi_attention.magi_attn_ext",
        "magi_attention.magi_attn_comm",
        "flash_mla",
        "cudnn",
    )
    module_evidence: dict[str, Any] = {}
    for module_name in native_modules:
        module = importlib.import_module(module_name)
        module_path_raw = getattr(module, "__file__", None)
        module_path = Path(module_path_raw).resolve() if module_path_raw else None
        module_evidence[module_name] = {
            "path": None if module_path is None else str(module_path),
            "sha256": (
                None if module_path is None or not module_path.is_file() else sha256_file(module_path)
            ),
        }
    evidence["native_modules"] = module_evidence

    distribution_names = {
        "nvidia_nvshmem_cu13": "nvidia-nvshmem-cu13",
        "nvidia_cutlass_dsl": "nvidia-cutlass-dsl",
        "pytest": "pytest",
    }
    versions: dict[str, str] = {}
    for key, expected in requirements.get("dependency_versions", {}).items():
        distribution = distribution_names.get(key, key.replace("_", "-"))
        actual = _distribution_version(distribution)
        if actual != expected:
            raise MatrixError(
                f"distribution {distribution} is {actual}, expected frozen {expected}"
            )
        versions[distribution] = actual
    evidence["distribution_versions"] = versions

    manifest_path = Path(expand(requirements["build_manifest"], context)).resolve()
    build_manifest = load_json(manifest_path)
    if build_manifest.get("schema_version") != 1:
        raise MatrixError(f"invalid build manifest schema in {manifest_path}")
    revision = build_manifest.get("magi_attention_revision")
    if not isinstance(revision, str) or not GIT_SHA_RE.fullmatch(revision):
        raise MatrixError("build manifest has no full MagiAttention revision")
    if expected_revision is not None and revision != expected_revision:
        raise MatrixError(
            f"image revision {revision} does not match expected {expected_revision}"
        )
    if build_manifest.get("base_image_digest") != requirements["base_image_digest"]:
        raise MatrixError("build manifest base image digest does not match the matrix")
    for name, expected in requirements.get("dependency_commits", {}).items():
        actual = build_manifest.get("dependency_commits", {}).get(name)
        if actual != expected:
            raise MatrixError(f"build dependency {name} is {actual}, expected {expected}")
    for path, expected in requirements.get("submodules", {}).items():
        actual = build_manifest.get("submodules", {}).get(path)
        if actual != expected:
            raise MatrixError(f"submodule {path} is {actual}, expected {expected}")
    verified_build_artifacts: dict[str, str] = {}
    for raw_path, expected in build_manifest.get("artifacts", {}).items():
        artifact = Path(raw_path)
        if not artifact.is_file():
            raise MatrixError(f"build artifact from manifest is missing: {artifact}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise MatrixError(
                f"build artifact SHA256 mismatch for {artifact}: {actual} != {expected}"
            )
        verified_build_artifacts[str(artifact)] = actual
    if not verified_build_artifacts:
        raise MatrixError("build manifest contains no verifiable artifacts")
    evidence["build_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "revision": revision,
        "verified_artifacts": verified_build_artifacts,
    }

    megatron_revision = git_revision(megatron_path)
    if megatron_revision != requirements["megatron_revision"]:
        raise MatrixError(
            f"Megatron revision is {megatron_revision}, expected "
            f"{requirements['megatron_revision']}"
        )
    evidence["megatron"] = {
        "path": str(megatron_path),
        "revision": megatron_revision,
    }

    sha_checks: list[dict[str, str]] = []
    for check in matrix.get("sha256_checks", []):
        path = Path(expand(check["path"], context)).resolve()
        if not path.is_file():
            raise MatrixError(f"SHA256 contract file is missing: {path}")
        actual = sha256_file(path)
        if actual != check["sha256"]:
            raise MatrixError(
                f"SHA256 contract mismatch for {path}: {actual} != {check['sha256']}"
            )
        sha_checks.append({"path": str(path), "sha256": actual})
    evidence["sha256_checks"] = sha_checks
    evidence["environment"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "MAGI_ATTENTION_NATIVE_GRPCOLL": os.environ.get(
            "MAGI_ATTENTION_NATIVE_GRPCOLL"
        ),
        "NVSHMEM_SYMMETRIC_SIZE_present": "NVSHMEM_SYMMETRIC_SIZE" in os.environ,
        "python": sys.version,
        "executable": sys.executable,
    }
    evidence["finished_at"] = utc_now()
    write_json(output_dir / "preflight.json", evidence)
    return evidence


def build_artifact_manifest(run_dir: Path) -> Path:
    manifest = run_dir / "artifact_manifest.sha256"
    entries: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        relative = path.relative_to(run_dir)
        entries.append(f"{sha256_file(path)}  {relative.as_posix()}")
    if not entries:
        raise MatrixError("cannot seal an empty artifact directory")
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest


def selected_cases(
    matrix: dict[str, Any], requested: Iterable[str]
) -> list[dict[str, Any]]:
    requested_set = set(requested)
    cases = list(matrix["cases"])
    if not requested_set:
        return cases
    known = {case["id"] for case in cases}
    unknown = requested_set - known
    if unknown:
        raise MatrixError(f"unknown requested cases: {sorted(unknown)}")
    return [case for case in cases if case["id"] in requested_set]


def main(argv: list[str] | None = None) -> int:
    release_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path, default=release_dir / "final_matrix.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("MAGI_DSA_FINAL_OUTPUT", "/artifacts")),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--expected-revision", default=os.environ.get("MAGI_DSA_EXPECTED_REVISION")
    )
    parser.add_argument("--image-id", default=os.environ.get("MAGI_DSA_IMAGE_ID"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_manifest is not None:
        verified = verify_sha256_manifest(args.verify_manifest.resolve())
        print(f"verified {len(verified)} SHA256 artifacts in {args.verify_manifest}")
    elif args.verify_only:
        parser.error("--verify-only requires --verify-manifest")
    if args.verify_only:
        return 0

    matrix_path = args.matrix.resolve()
    matrix = load_json(matrix_path)
    validate_matrix(matrix)
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    )
    if not RUN_ID_RE.fullmatch(run_id) or run_id in (".", ".."):
        raise MatrixError(f"unsafe run id {run_id!r}")
    run_dir = args.output_dir.resolve() / run_id
    if run_dir.exists():
        raise MatrixError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    shutil.copyfile(matrix_path, run_dir / "matrix.snapshot.json")

    repo_root = Path(
        os.environ.get("MAGI_DSA_REPO_ROOT", str(release_dir.parents[2]))
    ).resolve()
    megatron_path = Path(
        os.environ.get("MAGI_DSA_MEGATRON_PATH", "/opt/megatron-lm")
    ).resolve()
    base_context = {
        "python": sys.executable,
        "repo_root": str(repo_root),
        "release_dir": str(release_dir),
        "output_dir": str(run_dir),
        "megatron_path": str(megatron_path),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at": utc_now(),
        "status": "running",
        "preflight": None,
        "cases": [],
        "error": None,
    }
    exit_code = 0
    try:
        print(f"[preflight] image={args.image_id} revision={args.expected_revision}")
        result["preflight"] = run_preflight(
            matrix,
            context=base_context,
            expected_revision=args.expected_revision,
            image_id=args.image_id,
            output_dir=run_dir,
        )
        if not args.preflight_only:
            for case in selected_cases(matrix, args.case):
                case_id = case["id"]
                case_dir = run_dir / "cases" / case_id
                case_dir.mkdir(parents=True)
                context = {**base_context, "case_dir": str(case_dir)}
                command = [expand(item, context) for item in case["command"]]
                junit_path = case_dir / "junit.xml"
                if case.get("junit"):
                    command.append(f"--junitxml={junit_path}")
                environment = os.environ.copy()
                for name in case.get("unset_env", []):
                    environment.pop(name, None)
                for name, value in case.get("env", {}).items():
                    environment[name] = expand(value, context)
                cwd = Path(expand(case.get("cwd", "/tmp"), context)).resolve()
                log_path = case_dir / "case.log"
                print(f"[case:{case_id}] starting (timeout={case['timeout_seconds']}s)")
                execution = run_command(
                    command,
                    cwd=cwd,
                    env=environment,
                    timeout_seconds=case["timeout_seconds"],
                    log_path=log_path,
                )
                case_result: dict[str, Any] = {
                    "id": case_id,
                    "description": case.get("description"),
                    "execution": execution,
                    "junit": None,
                    "status": "passed",
                    "violations": [],
                }
                if execution["timed_out"]:
                    case_result["violations"].append("case timed out")
                if execution["return_code"] not in case.get("expected_exit_codes", [0]):
                    case_result["violations"].append(
                        f"unexpected return code {execution['return_code']}"
                    )
                if case.get("junit"):
                    try:
                        counts = parse_junit(junit_path)
                        case_result["junit"] = counts
                        if counts["tests"] < case["min_tests"]:
                            case_result["violations"].append(
                                f"collected {counts['tests']} tests, minimum is {case['min_tests']}"
                            )
                        if counts["skipped"] != 0:
                            case_result["violations"].append(
                                f"zero-skip contract violated: {counts['skipped']} skipped"
                            )
                        if counts["failures"] or counts["errors"]:
                            case_result["violations"].append(
                                f"JUnit failures={counts['failures']} errors={counts['errors']}"
                            )
                    except MatrixError as error:
                        case_result["violations"].append(str(error))
                if case_result["violations"]:
                    case_result["status"] = "failed"
                    exit_code = 1
                write_json(case_dir / "case_result.json", case_result)
                result["cases"].append(case_result)
                print(
                    f"[case:{case_id}] {case_result['status']} "
                    f"({execution['duration_seconds']:.3f}s)"
                )
                if case_result["status"] != "passed" and not args.continue_on_failure:
                    break
        result["status"] = "passed" if exit_code == 0 else "failed"
    except BaseException as error:
        exit_code = 1
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr)
    finally:
        result["finished_at"] = utc_now()
        result_path = run_dir / "final_result.json"
        write_json(result_path, result)
        manifest = build_artifact_manifest(run_dir)
        # Verify immediately so a filesystem/write bug cannot produce a false pass.
        verify_sha256_manifest(manifest)
        print(f"result: {result_path}")
        print(f"artifact manifest: {manifest}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
