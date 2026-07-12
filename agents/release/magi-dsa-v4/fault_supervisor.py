#!/usr/bin/env python3
"""External 8-worker fail-stop launcher for the Magi_DSA V4 release gate.

The supervisor, not the poisoned process group, owns teardown.  It waits for a
healthy native round on every worker, triggers one rank-local hard exit, gives
the peers time to enter the next native collective, then sends TERM/KILL to all
surviving worker process sessions.  Success requires complete reclamation in at
most 60 seconds and a subsequent native health round in eight fresh processes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_FAULT_EXIT = 86
EXPECTED_HANDLE = "GrpCollIntraHandle"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def proc_start_time(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except (OSError, UnicodeError):
        return None
    return fields[21] if len(fields) > 21 else None


def spawn_cohort(
    *,
    worker: Path,
    output_dir: Path,
    mode: str,
    world_size: int,
    fault_rank: int,
    master_port: int,
    run_id: str,
    pg_timeout_seconds: float,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for rank in range(world_size):
        stdout_path = output_dir / f"rank-{rank}.stdout.log"
        stderr_path = output_dir / f"rank-{rank}.stderr.log"
        environment = os.environ.copy()
        environment.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "PYTHONUNBUFFERED": "1",
                "MAGI_ATTENTION_NATIVE_GRPCOLL": "1",
                "MAGI_DSA_FAULT_RUN_ID": run_id,
            }
        )
        environment.pop("NVSHMEM_SYMMETRIC_SIZE", None)
        command = [
            sys.executable,
            str(worker),
            "--mode",
            mode,
            "--coord-dir",
            str(output_dir / "coord"),
            "--rank",
            str(rank),
            "--local-rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--fault-rank",
            str(fault_rank),
            "--pg-timeout-seconds",
            str(pg_timeout_seconds),
            "--coord-timeout-seconds",
            str(max(pg_timeout_seconds, 60.0)),
        ]
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                env=environment,
                cwd="/tmp",
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
        records.append(
            {
                "rank": rank,
                "process": process,
                "pid": process.pid,
                "proc_start_time": proc_start_time(process.pid),
                "command": command,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "spawned_at": utc_now(),
                "term_sent_at": None,
                "kill_sent_at": None,
            }
        )
    return records


def alive(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["process"].poll() is None]


def signal_records(records: Iterable[dict[str, Any]], sig: signal.Signals) -> None:
    timestamp_key = "term_sent_at" if sig == signal.SIGTERM else "kill_sent_at"
    timestamp = utc_now()
    for record in records:
        process: subprocess.Popen[Any] = record["process"]
        if process.poll() is not None:
            continue
        try:
            # Workers inherit the supervisor's process group.  Direct signals
            # let the supervisor record every worker, while an outer matrix
            # timeout can still kill the supervisor group and all descendants.
            process.send_signal(sig)
            record[timestamp_key] = timestamp
        except ProcessLookupError:
            pass


def reap_until(records: list[dict[str, Any]], deadline: float) -> None:
    while alive(records) and time.monotonic() < deadline:
        time.sleep(0.03)
    for record in records:
        process: subprocess.Popen[Any] = record["process"]
        if process.poll() is not None:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass


def terminate_all(
    records: list[dict[str, Any]], *, grace_seconds: float, deadline: float
) -> None:
    signal_records(alive(records), signal.SIGTERM)
    term_deadline = min(deadline, time.monotonic() + grace_seconds)
    reap_until(records, term_deadline)
    if alive(records):
        signal_records(alive(records), signal.SIGKILL)
    reap_until(records, deadline)


def scan_run_id_survivors(run_id: str) -> list[int]:
    marker = f"MAGI_DSA_FAULT_RUN_ID={run_id}".encode()
    survivors: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environment = (entry / "environ").read_bytes()
        except OSError:
            continue
        if marker in environment.split(b"\0"):
            survivors.append(int(entry.name))
    return sorted(survivors)


def serializable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        process: subprocess.Popen[Any] = record["process"]
        result.append(
            {
                key: value
                for key, value in record.items()
                if key != "process"
            }
            | {
                "return_code": process.poll(),
                "finished_at": utc_now() if process.poll() is not None else None,
            }
        )
    return result


def validate_native_evidence(paths: list[Path], world_size: int) -> list[dict[str, Any]]:
    if len(paths) != world_size:
        raise RuntimeError(
            f"expected {world_size} native evidence files, found {len(paths)}"
        )
    evidence = [load_json(path) for path in sorted(paths)]
    ranks = sorted(int(item.get("rank", -1)) for item in evidence)
    if ranks != list(range(world_size)):
        raise RuntimeError(f"native evidence ranks are incomplete: {ranks}")
    for item in evidence:
        if item.get("native_handle_type") != EXPECTED_HANDLE:
            raise RuntimeError(f"non-native handle evidence: {item}")
        if item.get("num_rdma_ranks") != 1:
            raise RuntimeError(f"unexpected RDMA topology evidence: {item}")
        if item.get("num_sms") != 20:
            raise RuntimeError(f"unexpected native SM configuration: {item}")
        if item.get("num_nvl_bytes") != 1 << 30:
            raise RuntimeError(f"unexpected native NVL allocation: {item}")
        if item.get("num_rdma_bytes") != 0:
            raise RuntimeError(f"unexpected native RDMA allocation: {item}")
        if item.get("reverse_value") != world_size:
            raise RuntimeError(f"native reverse reduction did not cover all ranks: {item}")
    return evidence


def wait_for_ready(
    records: list[dict[str, Any]], coord_dir: Path, world_size: int, timeout: float
) -> list[Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready = list(coord_dir.glob("ready.*.json"))
        if len(ready) == world_size:
            return ready
        failed = [
            (record["rank"], record["process"].poll())
            for record in records
            if record["process"].poll() is not None
        ]
        if failed:
            raise RuntimeError(f"workers exited before fault readiness: {failed}")
        time.sleep(0.05)
    raise TimeoutError(f"workers did not become ready within {timeout} seconds")


def wait_for_health(
    records: list[dict[str, Any]], coord_dir: Path, world_size: int, timeout: float
) -> list[Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(record["process"].poll() is not None for record in records):
            break
        time.sleep(0.05)
    if alive(records):
        terminate_all(records, grace_seconds=5.0, deadline=time.monotonic() + 15.0)
        raise TimeoutError(f"fresh native health cohort exceeded {timeout} seconds")
    return_codes = [record["process"].poll() for record in records]
    if return_codes != [0] * world_size:
        raise RuntimeError(f"fresh native health workers failed: {return_codes}")
    paths = list(coord_dir.glob("health.*.json"))
    if len(paths) != world_size:
        raise RuntimeError(
            f"fresh health produced {len(paths)} evidence files, expected {world_size}"
        )
    return paths


def run_fault_phase(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    phase_dir = args.output_dir / "fault-phase"
    coord_dir = phase_dir / "coord"
    records = spawn_cohort(
        worker=args.worker,
        output_dir=phase_dir,
        mode="fault",
        world_size=args.world_size,
        fault_rank=args.fault_rank,
        master_port=free_local_port(),
        run_id=run_id,
        pg_timeout_seconds=args.initialization_timeout_seconds,
    )
    triggered = False
    trigger_monotonic: float | None = None
    trigger_wall: str | None = None
    try:
        ready_paths = wait_for_ready(
            records,
            coord_dir,
            args.world_size,
            args.initialization_timeout_seconds,
        )
        ready_evidence = validate_native_evidence(ready_paths, args.world_size)
        (coord_dir / "trigger").touch()
        triggered = True
        trigger_monotonic = time.monotonic()
        trigger_wall = utc_now()
        fault_deadline = trigger_monotonic + args.fault_timeout_seconds

        first_failure: dict[str, Any] | None = None
        while time.monotonic() < fault_deadline:
            failed = [
                record
                for record in records
                if record["process"].poll() is not None
            ]
            if failed:
                first = min(failed, key=lambda item: item["rank"])
                first_failure = {
                    "rank": first["rank"],
                    "return_code": first["process"].poll(),
                    "detected_at": utc_now(),
                    "seconds_after_trigger": round(
                        time.monotonic() - trigger_monotonic, 6
                    ),
                }
                break
            time.sleep(0.02)
        if first_failure is None:
            raise TimeoutError("no rank failure was observed before the 60-second deadline")

        # The fault rank exits only after all seven peers have armed.  Give those
        # peers a bounded window to record entry into the poisoned native round.
        entered_deadline = min(fault_deadline, time.monotonic() + 5.0)
        while (
            len(list(coord_dir.glob("entered.*.json"))) < args.world_size - 1
            and time.monotonic() < entered_deadline
        ):
            time.sleep(0.02)
        entered_paths = list(coord_dir.glob("entered.*.json"))
        terminate_all(
            records,
            grace_seconds=args.term_grace_seconds,
            deadline=fault_deadline,
        )
        reclaimed_seconds = time.monotonic() - trigger_monotonic
        survivors = scan_run_id_survivors(run_id)
        return_codes = {
            record["rank"]: record["process"].poll() for record in records
        }
        violations: list[str] = []
        if return_codes.get(args.fault_rank) != EXPECTED_FAULT_EXIT:
            violations.append(
                f"fault rank returned {return_codes.get(args.fault_rank)}, "
                f"expected {EXPECTED_FAULT_EXIT}"
            )
        if len(entered_paths) != args.world_size - 1:
            violations.append(
                f"only {len(entered_paths)}/{args.world_size - 1} peers entered "
                "the poisoned native round"
            )
        if alive(records):
            violations.append(
                f"launcher still owns live workers: {[item['pid'] for item in alive(records)]}"
            )
        if survivors:
            violations.append(f"run-id process survivors remain: {survivors}")
        if reclaimed_seconds > args.fault_timeout_seconds:
            violations.append(
                f"teardown took {reclaimed_seconds:.3f}s, exceeding "
                f"{args.fault_timeout_seconds}s"
            )
        if any(
            code == 92 for rank, code in return_codes.items() if rank != args.fault_rank
        ):
            violations.append("a peer unexpectedly completed the poisoned native round")
        return {
            "status": "passed" if not violations else "failed",
            "triggered": triggered,
            "triggered_at": trigger_wall,
            "first_failure": first_failure,
            "reclaimed_seconds": round(reclaimed_seconds, 6),
            "deadline_seconds": args.fault_timeout_seconds,
            "entered_peer_count": len(entered_paths),
            "ready_evidence": ready_evidence,
            "return_codes": return_codes,
            "survivors": survivors,
            "violations": violations,
            "workers": serializable_records(records),
        }
    except BaseException:
        deadline = (
            (trigger_monotonic + args.fault_timeout_seconds)
            if trigger_monotonic is not None
            else time.monotonic() + min(args.fault_timeout_seconds, 30.0)
        )
        terminate_all(
            records,
            grace_seconds=args.term_grace_seconds,
            deadline=deadline,
        )
        raise


def run_health_phase(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    phase_dir = args.output_dir / "fresh-health-phase"
    coord_dir = phase_dir / "coord"
    health_run_id = f"{run_id}-fresh-health"
    records = spawn_cohort(
        worker=args.worker,
        output_dir=phase_dir,
        mode="health",
        world_size=args.world_size,
        fault_rank=args.fault_rank,
        master_port=free_local_port(),
        run_id=health_run_id,
        pg_timeout_seconds=args.health_timeout_seconds,
    )
    try:
        paths = wait_for_health(
            records, coord_dir, args.world_size, args.health_timeout_seconds
        )
        evidence = validate_native_evidence(paths, args.world_size)
        survivors = scan_run_id_survivors(health_run_id)
        violations = [] if not survivors else [f"fresh health survivors remain: {survivors}"]
        return {
            "status": "passed" if not violations else "failed",
            "evidence": evidence,
            "survivors": survivors,
            "violations": violations,
            "workers": serializable_records(records),
        }
    except BaseException:
        terminate_all(
            records,
            grace_seconds=args.term_grace_seconds,
            deadline=time.monotonic() + min(args.health_timeout_seconds, 30.0),
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--fault-rank", type=int, default=3)
    parser.add_argument("--fault-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--term-grace-seconds", type=float, default=5.0)
    parser.add_argument("--initialization-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--health-timeout-seconds", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.worker = args.worker.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.world_size != 8:
        raise RuntimeError(f"release fault matrix requires exactly 8 workers, got {args.world_size}")
    if not 0 <= args.fault_rank < args.world_size:
        raise RuntimeError(f"invalid fault rank {args.fault_rank}")
    if args.fault_timeout_seconds != 60:
        raise RuntimeError("the frozen fail-stop watchdog is exactly 60 seconds")
    if not 0 < args.term_grace_seconds < args.fault_timeout_seconds:
        raise RuntimeError("TERM grace must be positive and shorter than the fault watchdog")
    if not args.worker.is_file():
        raise RuntimeError(f"fault worker does not exist: {args.worker}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing stale non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "NVSHMEM_SYMMETRIC_SIZE" in os.environ:
        raise RuntimeError("NVSHMEM_SYMMETRIC_SIZE must be absent")

    import torch

    if torch.cuda.device_count() != args.world_size:
        raise RuntimeError(
            f"supervisor sees {torch.cuda.device_count()} GPUs, expected exactly 8"
        )

    run_id = f"magi-dsa-fault-{uuid.uuid4().hex}"
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": utc_now(),
        "world_size": args.world_size,
        "fault_rank": args.fault_rank,
        "fault_timeout_seconds": args.fault_timeout_seconds,
        "term_grace_seconds": args.term_grace_seconds,
        "fault_phase": None,
        "fresh_health_phase": None,
        "status": "running",
        "error": None,
    }
    exit_code = 1
    try:
        result["fault_phase"] = run_fault_phase(args, run_id)
        # A fresh process group, rather than the poisoned group, proves that the
        # external teardown released all GPU/native resources needed by the next job.
        time.sleep(1.0)
        result["fresh_health_phase"] = run_health_phase(args, run_id)
        phases_passed = all(
            result[name] is not None and result[name]["status"] == "passed"
            for name in ("fault_phase", "fresh_health_phase")
        )
        result["status"] = "passed" if phases_passed else "failed"
        exit_code = 0 if phases_passed else 1
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        print(result["error"], file=sys.stderr)
    finally:
        result["finished_at"] = utc_now()
        write_json(args.output_dir / "fault_result.json", result)
        print(json.dumps({"status": result["status"], "result": str(args.output_dir / 'fault_result.json')}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
