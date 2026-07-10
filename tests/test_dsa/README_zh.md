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
watchdog，CP 测试使用 60 秒 watchdog。步骤 1 不包含 CP=2 数据通信。

最终目录职责：`test_dsa_api.py` 覆盖 CP=1 公共 API；`test_dsa_cp.py`
覆盖 CP=2；`test_dsa_dispatch.py` 与 `test_dsa_solver.py` 覆盖静态计划；
`test_dsa_pack_kernel.py` 覆盖 SM90 packing/remap/CSR kernel。
