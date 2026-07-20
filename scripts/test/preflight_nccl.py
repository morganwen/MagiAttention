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
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8
    torch.cuda.set_device(local_rank)
    assert torch.cuda.get_device_capability() == (10, 3)

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=45))
    value = torch.tensor(float(rank), device="cuda", dtype=torch.float32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    assert value.item() == 28.0

    identity = torch.tensor([rank, local_rank, 10, 3], device="cuda", dtype=torch.int32)
    gathered = [torch.empty_like(identity) for _ in range(world_size)]
    dist.all_gather(gathered, identity)
    expected = torch.tensor(
        [[index, index, 10, 3] for index in range(world_size)],
        device="cuda",
        dtype=torch.int32,
    )
    assert torch.equal(torch.stack(gathered), expected)
    dist.barrier()
    report = {
        "device": torch.cuda.get_device_name(),
        "local_rank": local_rank,
        "rank": rank,
        "reduced_sum": value.item(),
        "world_size": world_size,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
