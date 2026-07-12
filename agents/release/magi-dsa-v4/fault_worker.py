#!/usr/bin/env python3
"""One worker used by the isolated Magi_DSA fail-stop acceptance probe.

Every worker first completes a real native GroupCast/GroupReduce round and
records the concrete handle type.  In fault mode, one designated rank exits
without process-group cleanup; the remaining ranks enter a second native
collective and are expected to be reclaimed by ``fault_supervisor.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


EXPECTED_FAULT_EXIT = 86


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


def wait_for(path: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.02)


def native_round(
    group: Any,
    rank: int,
    world_size: int,
    *,
    launch_evidence: Path | None = None,
) -> dict[str, Any]:
    import torch

    from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
    from magi_attention.comm.primitive.grpcoll._handle import GrpCollIntraHandle
    from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr
    from magi_attention.functional.dsa_comm import (
        DsaPayloadKind,
        DsaTypedPayload,
        DsaWorkTracker,
        build_dsa_comm_plan,
        materialize_dsa_comm_plan,
        start_dsa_group_cast,
        start_dsa_group_reduce,
    )
    from magi_attention.meta.collection.dsa_meta import DsaFragmentSpec
    from magi_attention.meta.solver.dsa_dispatch import build_dsa_dispatch_plan

    sample_length = 128 * world_size
    per_rank_fragments = [
        [DsaFragmentSpec(0, owner * 128, (owner + 1) * 128)]
        for owner in range(world_size)
    ]
    dispatch_plan = build_dsa_dispatch_plan(
        [sample_length],
        per_rank_fragments,
        compress_ratio=4,
        policy="sequential",
    )
    comm_plan = build_dsa_comm_plan(dispatch_plan, rank, group)
    device_plan = materialize_dsa_comm_plan(comm_plan, torch.cuda.current_device())
    meta = comm_plan.compressed_kv
    mapping = device_plan.compressed_kv
    local = torch.full(
        (meta.local_row_count, 512),
        float(rank + 1),
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
    )

    with DsaWorkTracker(group):
        cast_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.COMPRESSED_KV, local),
            meta,
            device_map=mapping,
            async_op=True,
        )
        if launch_evidence is not None:
            write_json(
                launch_evidence,
                {
                    "rank": rank,
                    "pid": os.getpid(),
                    "native_group_cast_launched_at": utc_now(),
                },
            )
        remote = cast_work.wait()
        handle = cast_work.native_handle_dict.get("group_cast")
        if not isinstance(handle, GrpCollIntraHandle):
            raise RuntimeError(
                f"native backend did not produce GrpCollIntraHandle: {type(handle)!r}"
            )
        remote_gradient = DsaTypedPayload(
            DsaPayloadKind.COMPRESSED_KV,
            torch.ones_like(remote.tensor, dtype=torch.float32),
        )
        local_gradient = DsaTypedPayload(
            DsaPayloadKind.COMPRESSED_KV,
            torch.ones(
                (meta.local_row_count, 512),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            ),
        )
        reduce_work = start_dsa_group_reduce(
            remote_gradient,
            local_gradient,
            cast_work,
            async_op=True,
            output_dtype=torch.float32,
        )
        owner_gradient = reduce_work.wait().tensor

    expected = torch.full_like(owner_gradient, float(world_size))
    if not torch.equal(owner_gradient, expected):
        maximum_error = float((owner_gradient - expected).abs().max().item())
        raise RuntimeError(f"native reverse sum mismatch, max error {maximum_error}")
    buffer = grpcoll_buffer_mgr.get_buffer(group, DsaPayloadKind.COMPRESSED_KV.value)
    num_rdma_ranks = int(buffer.runtime.get_num_rdma_ranks())
    if num_rdma_ranks != 1:
        raise RuntimeError(
            f"single-node native probe expected num_rdma_ranks=1, got {num_rdma_ranks}"
        )
    config: GrpCollConfig = grpcoll_buffer_mgr.get_config(group)
    return {
        "rank": rank,
        "world_size": world_size,
        "native_handle_type": type(handle).__name__,
        "num_rdma_ranks": num_rdma_ranks,
        "local_rows": meta.local_row_count,
        "remote_rows": meta.receive_row_count,
        "reverse_value": world_size,
        "num_sms": config.num_sms,
        "num_nvl_bytes": config.num_nvl_bytes,
        "num_rdma_bytes": config.num_rdma_bytes,
        "completed_at": utc_now(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fault", "health"), required=True)
    parser.add_argument("--coord-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--fault-rank", type=int, default=3)
    parser.add_argument("--coord-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--pg-timeout-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.world_size != 8:
        raise RuntimeError(f"fault acceptance requires exactly 8 workers, got {args.world_size}")
    if not 0 <= args.rank < args.world_size:
        raise RuntimeError(f"invalid rank {args.rank}")
    if not 0 <= args.local_rank < args.world_size:
        raise RuntimeError(f"invalid local rank {args.local_rank}")
    if not 0 <= args.fault_rank < args.world_size:
        raise RuntimeError(f"invalid fault rank {args.fault_rank}")
    if "NVSHMEM_SYMMETRIC_SIZE" in os.environ:
        raise RuntimeError("NVSHMEM_SYMMETRIC_SIZE must be absent for the CP8 intranode probe")
    os.environ["MAGI_ATTENTION_NATIVE_GRPCOLL"] = "1"

    import torch
    import torch.distributed as dist

    from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
    from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr

    if torch.cuda.device_count() != args.world_size:
        raise RuntimeError(
            f"worker sees {torch.cuda.device_count()} GPUs, expected exactly {args.world_size}"
        )
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=args.rank,
        world_size=args.world_size,
        init_method="env://",
        timeout=timedelta(seconds=args.pg_timeout_seconds),
    )
    group = dist.distributed_c10d._get_default_group()
    grpcoll_buffer_mgr.initialize(
        group=group,
        config=GrpCollConfig(
            num_sms=20,
            nvl_chunk_size=8,
            nvl_buffer_size=256,
            rdma_chunk_size=8,
            rdma_buffer_size=256,
            num_nvl_bytes=1 << 30,
            num_rdma_bytes=0,
        ),
    )
    evidence = native_round(group, args.rank, args.world_size)
    evidence.update(
        {
            "mode": args.mode,
            "pid": os.getpid(),
            "run_id": os.environ.get("MAGI_DSA_FAULT_RUN_ID"),
        }
    )

    args.coord_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "health":
        dist.barrier(group=group)
        write_json(args.coord_dir / f"health.{args.rank}.json", evidence)
        grpcoll_buffer_mgr.release_group(group)
        dist.destroy_process_group()
        print(json.dumps({"event": "health_ok", **evidence}), flush=True)
        return 0

    write_json(args.coord_dir / f"ready.{args.rank}.json", evidence)
    print(json.dumps({"event": "fault_ready", **evidence}), flush=True)
    wait_for(args.coord_dir / "trigger", args.coord_timeout_seconds)

    if args.rank == args.fault_rank:
        armed = [
            args.coord_dir / f"armed.{rank}"
            for rank in range(args.world_size)
            if rank != args.fault_rank
        ]
        deadline = time.monotonic() + args.coord_timeout_seconds
        while not all(path.exists() for path in armed):
            if time.monotonic() >= deadline:
                raise TimeoutError("fault rank timed out waiting for peer armed markers")
            time.sleep(0.02)
        write_json(
            args.coord_dir / "faulted.json",
            {
                "fault_rank": args.rank,
                "pid": os.getpid(),
                "faulted_at": utc_now(),
                "exit_code": EXPECTED_FAULT_EXIT,
            },
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(EXPECTED_FAULT_EXIT)

    (args.coord_dir / f"armed.{args.rank}").touch()
    wait_for(args.coord_dir / "faulted.json", args.coord_timeout_seconds)
    try:
        native_round(
            group,
            args.rank,
            args.world_size,
            launch_evidence=args.coord_dir / f"launched.{args.rank}.json",
        )
    except BaseException as error:
        write_json(
            args.coord_dir / f"peer_error.{args.rank}.json",
            {
                "rank": args.rank,
                "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}",
                "failed_at": utc_now(),
            },
        )
        return 91
    write_json(
        args.coord_dir / f"unexpected_completion.{args.rank}.json",
        {"rank": args.rank, "pid": os.getpid(), "completed_at": utc_now()},
    )
    return 92


if __name__ == "__main__":
    raise SystemExit(main())
