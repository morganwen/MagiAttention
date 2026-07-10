# Magi_DSA V4 测试说明

正式验收硬件是 1 或 2 张 NVIDIA H100 80GB（SM90）。冻结开发镜像为
`magi-dsa-dev:v2`，本机 image id 是
`sha256:9c51e29d1fda8fc1a6e2a8e16c7b0773309e91dcbd44cf6d0182e5e3327c1029`；
依赖源码 revision 见 `docs/magi_dsa_v4_design.md`。

编译与运行必须分离：先按 Dockerfile 构建固定 revision 的镜像和外部 kernel，
测试阶段不得静默编译另一个 revision。公共 API reference 测试命令：

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/scratch.wewen_gpu:/ws \
  -w /ws/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll \
  magi-dsa-dev:v2 timeout 300 \
  pytest -q tests/test_dsa/test_dsa_api.py -m "not dsa_kernel"
```

在冻结镜像中显式运行 FlashMLA/cuDNN 路径：

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/scratch.wewen_gpu:/ws \
  -w /ws/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll \
  magi-dsa-dev:v2 timeout 300 \
  pytest -q tests/test_dsa/test_dsa_api.py -m dsa_kernel
```

可用 `-k contract`、`-k compressor`、`-k reference`、`-k kernel` 过滤。
编译后单个 kernel 执行超过 10 秒视为死锁；后续 kernel 单测使用 30 秒
watchdog，CP 测试使用 60 秒 watchdog。步骤 3 已提供 CP=2 transport-only
覆盖；完整 CP=2 attention 留在步骤 5/6 接入。

双 H100 上运行步骤 3 GroupCast/GroupReduce 通信测试：

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/scratch.wewen_gpu:/ws \
  -w /ws/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll \
  magi-dsa-dev:v2 timeout 60 \
  pytest -q tests/test_dsa/test_dsa_cp.py
```

A2AV fallback、非连续路线、FP32 反向归约和空路线不依赖可选的
`magi_attn_comm` 编译扩展。扩展已安装时会执行 native grpcoll 对拍；否则只
skip 该单项。此文件当前不运行 attention kernel。

fragment 与 solver 是 CPU-only 测试，在同一冻结镜像中运行：

```bash
docker run --rm \
  -v /home/scratch.wewen_gpu:/ws \
  -w /ws/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll \
  magi-dsa-dev:v2 timeout 300 \
  pytest -q tests/test_dsa/test_dsa_dispatch.py tests/test_dsa/test_dsa_solver.py
```

最终目录职责：`test_dsa_api.py` 覆盖 CP=1 公共 API；`test_dsa_cp.py`
覆盖 CP=2；`test_dsa_dispatch.py` 与 `test_dsa_solver.py` 覆盖静态计划；
`test_dsa_pack_kernel.py` 覆盖 SM90 packing/remap/CSR kernel。
