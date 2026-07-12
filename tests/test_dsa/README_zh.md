# Magi_DSA V4 测试说明

## B300 平台与镜像构建源

当前步骤 1–7 的验收目标是 8× NVIDIA B300 SXM6 AC 节点（SM103）上的
单个 `world_size=cp=8` 进程组，固定使用 GPU 0–7。CP=1 只作为同权重、
同全局输入的数值/梯度 oracle，不是分布式出口；CP=2 只保留兼容回归，也不能
替代 CP=8 出口。此前把 8 卡拆成四个 CP=2 pair 的口径已废弃；单组 CP8
步骤 1–7 矩阵现已通过。

宿主机权威 worktree 为：

```text
/home/scratch.wewen_gpu/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll
```

B300 镜像配方在主 checkout 中，不在 worktree 中：

```text
/home/scratch.wewen_gpu/MagiAttention/agents/docker/magi-dsa-b300-step5/Dockerfile
```

配方目录保留了历史步骤号。它通过 registry digest
`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`
固定 `nvcr.io/nvidia/pytorch:26.06-py3`，并固定 FlashMLA、
cudnn-frontend、fast-hadamard-transform 和 CUTLASS DSL 源码。使用该文件
构建当前本地测试 tag：

```bash
docker build \
  -f /home/scratch.wewen_gpu/MagiAttention/agents/docker/magi-dsa-b300-step5/Dockerfile \
  -t magi-dsa-b300-cp8:dev \
  /home/scratch.wewen_gpu/MagiAttention
```

fast-hadamard-transform 按 compute capability 10.3 构建。FlashMLA 使用
`FLASH_MLA_DISABLE_SM90=1` 生成 B300 使用的 SM100-family 路径。
`dsa_pack.py` 的 mapping/frontend 不绑定某个 minor arch，且每个编译
cache key 都包含实际设备的 `(major, minor)`，因此 SM103 有独立的
packing/remap/CSR cache entry。

镜像 tag 是可变别名。本次步骤 1–7 复验使用的基础 image id 为
`sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da`。
每次测试报告都要记录并核对 `docker image inspect` 返回的 image ID。

## Native 通信是验收前置条件

编译与执行必须分开。最终 B300 验收镜像还必须包含
`nvidia-nvshmem-cu13==3.6.5`，以及用当前仓库 revision 按 compute
capability 100 构建的 `magi_attention.magi_attn_comm`（仓库构建在
B300 上使用 SM100 family target）。编译时不得通过
`DISABLE_NVSHMEM` 去掉 NVSHMEM。

上述基础 Dockerfile 提供 B300 计算依赖，但它本身不能证明 native
通信前置条件已满足。CP=8 native 复验为每个 `(group, buffer_name)` 固定
`GrpCollConfig.num_nvl_bytes=1073741824`（1 GiB）。full path 常驻四个 typed
payload buffer 和一个 replicated-gradient buffer；计入每 buffer 的 32 MiB
workspace 后，每 GPU 至少需要 `5,536,482,080` B（约 5.15625 GiB），步骤 8
计时前必须执行 dry-allocation。单节点全 NVLink 路径使用 `num_rdma_bytes=0`，
因此 `NVSHMEM_SYMMETRIC_SIZE` 必须 unset，报告为 N/A。步骤 8/9 必须把
`B300_IMAGE` 设为衍生的、已启用 native
的不可变 image ID，基础 tag 仍缺扩展时不得使用。本次步骤 1–7 复验则使用下面
明确记录的增量容器。

本轮 CP8 复验在隔离容器中为上述基础 image 安装 NVSHMEM 3.6.5。
实际装载的 `magi_attn_ext` SHA256 为
`0427073e7a0f16450bade229528638bfd1c9f610bf31d00f04d7f7899ffa0eaf`，
`magi_attn_comm` SHA256 为
`ca98aa8439007b647c52b8aa4e18b6b0beb631593846445bd877528346c87030`；
`cuobjdump --list-elf` 只列出 `sm_100`，两者均已在 SM103 设备实际装载。
CP8 测试还断言了实际 handle 为 `GrpCollIntraHandle`，并执行 native full
backward 而没有 fallback；这个增量容器仍不是步骤 8/9 尚待构建的不可变
native 镜像。

```bash
NATIVE_CONTAINER=magi-dsa-b300-cp8-native
test "$(docker inspect --format '{{.Image}}' "${NATIVE_CONTAINER}")" = \
  'sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da'
docker exec -w /workspace "${NATIVE_CONTAINER}" \
  python -c 'import nvidia.nvshmem; import magi_attention.magi_attn_comm'
```

如果扩展只安装在镜像内的某个 checkout，就不能用另一份 bind-mounted
package 将其遮蔽。应当在挂载的 worktree 中构建扩展，或直接运行验收
镜像内完全相同的仓库 revision。

所有 `test_native_grpcoll_*` 用例都必须在 B300 的 8-rank 进程组上实际执行。
如因 `magi_attn_comm` 或 NVSHMEM 不可用而 skip，则验收失败；CP=8 的步骤 1–7
验收必须是零 skip。

native row 在内部按 dtype 所需对齐补零，wait 后恢复 logical row shape；这同时
覆盖 BF16 compressed-Ki 和 hidden width 64 等 FP32 GroupReduce。大型 replicated
FP32 梯度按不超过 4096 elements 的对齐 row 分块，避免 native scratch 随单条
超宽展平 row 膨胀。

## Runtime 与故障边界

CP=1 支持完整 module deepcopy 和 `torch.save`，反序列化时重建锁并清空
device-bound cache，forward plan 在下次使用时按需重新物化。包括 CP=8 在内的
分布式 `ProcessGroup` 是 PyTorch 不可 pickle 的外部状态，因此分布式 runtime
只支持 `state_dict`：先在目标进程组上新建 runtime，再加载状态。

正常退出和协调可恢复异常会 drain 已发起 work。collective wait 或 rank-local
compute/kernel 失败时，会在 drain 公共前缀后 best-effort 请求
`ProcessGroup.abort()`。B300 实测表明，abort 不能保证释放已经卡在
CUDA/NCCL/NVSHMEM stream wait 的 peer，因此不可恢复故障采用 fail-stop：外部
launcher 必须用 60 秒 watchdog 终止全部 worker。步骤 7 套件证明协调异常的
drain/reuse 和 abort 请求；步骤 9 必须在该 launcher 下测试隔离单 rank 故障，
不得宣称同进程恢复。

## 验证命令

下面是本轮已执行的 CP=8 步骤 1–7 复验命令。单卡命令只产生
oracle 证据；唯一分布式出口是已把当前 worktree 挂载到 `/workspace` 的增量
native 容器中的一个 8-rank 进程组。不要再使用旧的复制文件
`/tmp/native-test/test_dsa_cp.py`。步骤 8/9 必须用新构建的不可变 native image
替换这个增量容器。

```bash
WORKTREE=/home/scratch.wewen_gpu/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll
B300_BASE_IMAGE=sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da
NATIVE_CONTAINER=magi-dsa-b300-cp8-native
GPU=0
GPU_GROUP=0,1,2,3,4,5,6,7
GRPCOLL_NUM_NVL_BYTES=1073741824  # 1 GiB；test config 必须使用同一固定值

# CP=1 只作为数值/梯度 oracle，不计为分布式出口。
docker run --rm --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES="${GPU}" \
  -v "${WORKTREE}:/workspace/MagiAttention" \
  -v /home/scratch.wewen_gpu/megatron-lm:/ws/megatron-lm:ro \
  -e MAGI_DSA_MEGATRON_PATH=/ws/megatron-lm \
  -w /workspace/MagiAttention "${B300_BASE_IMAGE}" \
  timeout 1200 pytest -q -rs \
    tests/test_dsa/test_dsa_api.py \
    tests/test_dsa/test_dsa_dispatch.py \
    tests/test_dsa/test_dsa_megatron.py \
    tests/test_dsa/test_dsa_solver.py \
    tests/test_dsa/test_dsa_pack_kernel.py

# 步骤 3/5/6/7 的唯一分布式出口：一个 GPU0..7、world_size=cp=8 组。
# 单节点 NVLink 路径使用 num_rdma_bytes=0；NVSHMEM_SYMMETRIC_SIZE 为 unset/N/A。
docker exec \
  -e CUDA_VISIBLE_DEVICES="${GPU_GROUP}" \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29617 \
  -w /workspace "${NATIVE_CONTAINER}" \
  bash -lc 'unset NVSHMEM_SYMMETRIC_SIZE; \
    test -z "${NVSHMEM_SYMMETRIC_SIZE+x}"; \
    timeout 5400 pytest -q -rs tests/test_dsa/test_dsa_cp.py'
```

`test_dsa_cp.py` 已不是 transport-only。它当前包含：

- A2AV 和 native GroupCast/GroupReduce transport，包括空路线和
  payload-private 状态（步骤 3/4）；
- CP=8 reference/kernel forward、sequential/balanced plan 和 CP=1 oracle 对拍
  （步骤 5）；
- CP=8 reference/kernel backward、owner/replicated 归约和 saved-state 检查
  （步骤 6）；以及
- 2×2 overlap 矩阵、两个 in-flight microbatch、reentrant 和 retained
  backward、gradient accumulation、空 rank、协调真实 collective 的 drain/reuse、
  abort-request 检查和提前退出 drain（步骤 7）。

`test_dsa_megatron.py` 对只读 checkout 的精确 revision
`c6449f0b23be397449f21c0967c5fc90785e55ea` 验证 ratio=4 compressor forward
parity。源码缺失、依赖导入失败或 revision 不符都会产生 skip；即使 pytest 返回
0，该 skip 仍属于验收失败。

`test_dsa_pack_kernel.py` 包含 packing/remap/FP32-CSR 执行 watchdog。编译后
kernel 测试使用 30 秒 watchdog；CP8 transport 用例使用 180 秒，完整路径/
并发用例使用 600–900 秒，SM100 cuDNN sparse-backward 首次冷编译使用
1200 秒预算。

当前 2026-07-12 CP8 结果：API `32 passed`，dispatch+solver `36 passed`，
packing+Megatron `18 passed`，单个八卡 CP 文件 `27 passed, 0 skipped`
（637.08 秒），正式目录合计 `113 passed, 0 skipped`；旧回归为单卡
`17 passed`、CP8 `2 passed`。Black、isort、Ruff 0.12.5、compileall 和
`git diff --check` 均通过。

## 当前范围边界

步骤 1–7 已切换到 CP8 并完成上述 8-rank 矩阵。步骤 8 的性能校准、最终
solver 系数拟合、固定 20 packs 和 E2E 门槛没有运行；步骤 9 的最终不可变镜像、
launcher 隔离故障矩阵、集成和原子提交也没有运行。上述结果不能作为步骤 8 或
9 已完成的证据。
