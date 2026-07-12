# Magi_DSA V4 设计文档

本文与仓库根目录 `PLAN.md` 共同冻结 Magi_DSA V4 的实现口径；发生冲突时以 `PLAN.md` 的“冻结合同”为准。步骤 1–7 已在 8× NVIDIA B300 SXM6 AC 节点（SM103）上的单个 `world_size=cp=8` 进程组完成复验，固定使用 GPU 0–7。CP=1 只作为同权重、同全局输入的数值/梯度 oracle，不是分布式出口；CP=2 只保留兼容回归，也不能替代 CP=8 出口。此前把 8 卡拆成四个 CP=2 pair 的结果已废弃。

## 可复现基线

- MagiAttention 基线：`529fb0a4e273b3557a56d8afd60b74da46688095`。
- DSA 计算原型：`98e043cafebbdcbce6835e26849d96624136ef70`。
- Megatron dsv4 数值参考：`c6449f0b23be397449f21c0967c5fc90785e55ea`。
- DeepSeek 官方 HF 配置：`deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`；使用该 revision 的 `config.json` 和 `inference/config.json`。
- FlashMLA nv_dev：`b7643bd54521f563b839b98289b5cd048c062ba2`。
- NVIDIA cudnn-frontend：`f00538322e9d3d439fe8c5f3144644e58ee66823`（安装包 `nvidia-cudnn-frontend==1.27.0`）。
- fast-hadamard-transform：`e7706faf8d1c3b9f241e36860640ad1dac644ede`。
- CUTLASS 子仓：`81a43e6d92cdd8c20d22392f9579604ed5f710a1`；FA4 子仓：`ee1d15159cda6f3f97bfab9e487da146a8254970`；`nvidia-cutlass-dsl==4.5.2`。
- B300 镜像的权威构建源是主 checkout 中的 `/home/scratch.wewen_gpu/MagiAttention/agents/docker/magi-dsa-b300-step5/Dockerfile`（目录名保留了历史步骤号）。它直接固定 `nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`（NGC build `337426143`，build ref `d557151f4c7ddca284cb5e8d5ce78cee4d80f7e5`）。
- 该配方把 fast-hadamard-transform 构建到 compute capability 10.3，并使用 `FLASH_MLA_DISABLE_SM90=1` 只构建 FlashMLA SM100-family 对象；B300 上由 SM103 路径运行。本轮步骤 1–7 测试 tag 为 `magi-dsa-b300-step7:dev`，image id 为 `sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da`；tag 可变，报告以 image id 为准。
- 上述基础配方固定 FlashMLA、cudnn-frontend、fast-hadamard-transform 和 CUTLASS DSL revision，但仍不等于步骤 8/9 的最终 clean、不可变 native 镜像。本轮容器安装 `nvidia-nvshmem-cu13==3.6.5`，并挂载当前源码与扩展：`magi_attn_ext` SHA256 为 `0427073e7a0f16450bade229528638bfd1c9f610bf31d00f04d7f7899ffa0eaf`，`magi_attn_comm` SHA256 为 `ca98aa8439007b647c52b8aa4e18b6b0beb631593846445bd877528346c87030`。CP8 native 用例已断言实际 handle 为 `GrpCollIntraHandle`，不是 A2AV fallback。每个 `(group, buffer_name)` 固定 `GrpCollConfig.num_nvl_bytes=1073741824`（1 GiB）；四个 typed payload buffer 加一个 replicated-gradient buffer含 workspace 后，每 GPU 静态下限为 `5,536,482,080` B（约 5.15625 GiB），native full backward 已实际通过。单节点全 NVLink 且 `num_rdma_bytes=0` 时 `NVSHMEM_SYMMETRIC_SIZE` 保持 unset，报告为 N/A；步骤 8 仍须在最终 packs 上独立 dry-allocation。

## 数值与模块边界

- compressor 输入 hidden `x`，不是投影后的 latent KV。内容投影和门控投影都从 `hidden_size` 出发。
- Indexer 的 Qi、Ki、Weights 由 runtime 内部从 `qr` 和 `x` 投影生成；Indexer 分支 detach 输入，KL 不向 trunk `x/qr` 回传。
- cuDNN sparse backward 原生返回 `d_sink`。sink 等价于 value 为零的附加 softmax 条目，但正式实现不通过补算代替原生梯度。
- Q/KV 最后 64 维已经由模型侧做 partial RoPE。压缩条按 logical block id 使用压缩 RoPE；官方配置使用 `compress_rope_theta=160000`、YaRN factor 16、original max positions 65536。
- 压缩器只处理完整块，sample 尾部不足 ratio 的 token 不生成压缩条。ratio=4 使用重叠 compressor，ratio=128 使用非重叠 compressor。
- 位置 `p`（sample 内 0-based）只能看到 `floor((p+1)/ratio)` 个完整压缩块。
- Indexer score 为 `sum_h weights[q,h] * relu(Qi[q,h] dot Ki[k])`。top-k 按 score 降序、同分按小 logical block id；合法项在前，其余填 `-1`。
- KL target 来自主 Q 与 compressed KV 的逐头注意力分布；predict 来自 Indexer。CP 下可微返回值为 `local_kl_sum / global_query_tokens`。

## 公共 API 与 tensor schema

正式入口位于：

```text
magi_attention/api/dsa_attn_interface.py
magi_attention/dsa_runtime_mgr.py
magi_attention/functional/dist_dsa.py
```

`experimental/dsa_v4` 保留为已验证的计算/kernel 原型，公共调用方不再直接调用 `forward_cp` 或 `forward_cp_packed`。

- `DsaPackedMeta` 保存 packed sample 边界 `cu_seqlens`（1-D contiguous int32、首项 0、单调不减）。
- `MagiDSAInput` 具名保存 `x`、`qr`、`q`、`latent_kv`、FP32 `sink` 和 `packed_meta`。
- 行张量均为 packed THD：`x [T,hidden_size]`、`qr [T,q_lora_rank]`、`q [T,64,512]`、`latent_kv [T,512]`、`sink [64]`。
- `calc_dsa(input, runtime_mgr)` 返回 `O [T,64,512]` 与可微 FP32 标量 `kl_loss`。
- 一个 `MagiDSARuntimeMgr` 只服务其 config 固定的一种 ratio，并持有 compressor/Indexer 参数。CP=1 计算路径继续复用 `MagiDSAV4.forward_packed`，仅作为 oracle；步骤 2 按 packed layout/policy 缓存 fragment plan，步骤 3/4 为分布式 CP 提供 transport 和 device packing，步骤 5/6 把完整分布式 forward/backward 接入 `calc_dsa`，步骤 7 在同一 runtime 中加入可独立开关的 overlap 与 per-call 并发状态。当前验收必须在一个 CP=8 进程组上覆盖这些路径。
- ratio=0 不构建 compressor/Indexer；ratio=128 只构建 compressor；ratio=4 同时构建 compressor 和 Indexer。

V1 固定配置为：Hq=64、Hkv=1、D=512、rope dim=64、window=128、Hidx=64、Didx=128、topk=512，主路径 BF16，sink FP32。`hidden_size`、`q_lora_rank` 和 `softmax_scale` 由模型配置提供。

## Kernel 接入边界

- sparse forward 复用 FlashMLA `flash_mla_sparse_fwd`。
- sparse backward/d_sink、Indexer forward/top-k、score recompute 和 Indexer backward 复用 cudnn-frontend `cudnn.deepseek_sparse_attention`。
- 统一 wrapper 位于 `experimental/dsa_v4/kernels.py`；步骤 1 只接线，不重写、不 fork 外部 kernel。
- reference backend 只用于数值对拍；正式 kernel backend 的输入为 CUDA BF16，top-k 保持 device resident。
- 唯一计划内新增的计算辅助 kernel 是 `kernel/cutedsl/dsa_pack.py`，用于 packing、remap 与 FP32 CSR reduction。public frontend 和 device mapping schema 不绑定具体 minor arch，编译 cache key 包含设备的 `(major, minor)`、dtype、feature width 和 operation；因此 B300 使用独立 SM103 cache entry。kernel 仅使用 SM100-family 可用的通用 global copy、寄存器 FP32 累加和 128-bit vector copy。

## CP plan 与通信

- dispatch/transfer plan 以 sample-relative logical positions 和可非连续 fragments 表示；128 对齐只约束可切分边界，不假设 fragment owner 相邻。
- compressed KV、compressed Ki、window KV、compressor overlap x 是四种独立 typed payload，分别持有 metadata、buffer slot 和 work handle。
- compressed KV/Ki 静态发送到全部 CP peers；window KV 和 overlap x 按 sample/fragment transfer table 只发送 unique rows。
- 正式数据交换只通过 `group_cast` / `group_reduce`。DSA 生产代码不直接调用 torch P2P、all-gather、all-reduce 或 `all2all_v` 替代。
- backward 使用 forward 的对称 GroupReduce；窗口与压缩条落到同一原 token 的梯度先用 FP32 accumulator 合并。
- collective 进入顺序在全部 rank 一致；空路线仍传合法零长度 metadata。

步骤 2 已建立不可变的 host plan：`DsaFragmentSpec` 使用 sample-relative
坐标；完整 plan 显式保存每 rank fragments、compressed block owner、分段
restore map，以及 window KV / compressor overlap x 的 unique-row transfer
table。sequential baseline 按 128 对齐连续切分；balanced solver 先最小化
ratio=4 最慢 rank 的 Indexer scan cost，再在允许 slack 内按 token、通信、
fragment overhead 和显存预测选择完整 E2E 最小的候选。rank 0 确定性求解后
广播 plan，runtime 按 packed layout 与 policy 缓存。

步骤 3 在 `functional/dsa_comm.py` 中实现了通信层。`DsaCommPlan` 为四类
payload 各自生成一个 `GroupCollectiveArg`：window KV 与 overlap x 根据
transfer table 将相同 owner row 的 destination 合并后只 pack 一次；
compressed KV/Ki 则把 owner-local compressed block 静态发送给所有 peer。
每条 source/destination 空路线也保留显式零长度 split，因此所有 rank 能按
固定顺序进入 collective。

步骤 4 保留 torch `index_select`/`index_copy_` reference map 仅用于测试对拍；
production forward 使用静态 device-resident int32 destination→source map 和 CuTe
DSL row-copy kernel 打包 owner-local unique rows，再调用 `group_cast` 得到带稳定
logical row id 的 typed receive buffer。backward 复用同一元数据和 payload-private
native handle 调用对称 `group_reduce`，GroupReduce 初始 owner rows 也由 row-copy
kernel 提取；返回结果按 inverse CSR map 以固定顺序、无 atomic 写回 owner-local
FP32 accumulator，最后只做一次目标 dtype 转换。device map 可作为静态 plan state
共享，send/receive buffer、work 和 native handle 仍按调用独占。

`dsa_pack.py` 同时提供 logical block/token id→packed-local row 的 int32 remap LUT；
`-1` sentinel 和 unavailable logical id 保持 `-1`，动态 Indexer top-k 全程留在
device。CSR kernel 支持重复 source、空 destination row、极端 fan-in，以及覆盖
非空 row或累加到已有 FP32 accumulator两种模式，为步骤 5/6 的 KV bank
packing 和多路径梯度合并提供固定接口。

native grpcoll 的每一条 transport row 都按 dtype 查询对齐要求，并只在内部
send/receive/reduce buffer 右侧补零：例如 BF16 compressed Ki 从逻辑宽度 128
补到 256，FP32 hidden=64 的 GroupReduce 从 64 补到 128；wait 后统一裁回并恢复
原 logical row shape。A2AV、logical row id 和对外 tensor schema 均不改变。
replicated 参数或 sink 梯度若展平为超宽 FP32 row，会按不超过 4096 elements 的
对齐 row 分块再做 GroupReduce，避免 native scratch 随单 row 宽度膨胀；归约后按
原 numel/shape 裁回。

`DsaCommBufferSlot`、`DsaGroupCastWork`、`DsaGroupReduceWork` 和 native handle
dictionary 都按单次调用、单 payload 创建，不进入 runtime 共享静态对象。生产
DSA 通信源码只调用 `group_cast` / `group_reduce`；A2AV 是该 primitive 的内建
fallback，不在 DSA 层直接调用。

![步骤 2 负载均衡改前与改后](assets/dsa_step2_load_balance.svg)

步骤 3 修改前：

![步骤 3 通信修改前](assets/dsa_step3_comm_before.svg)

步骤 3 修改后：

![步骤 3 通信修改后](assets/dsa_step3_comm_after.svg)

## Saved-state 与并发

- forward 只保存 O、FP32 LSE、`topk_idx`、`topk_length` 和 compressor gate 反向必需中间量。
- 不保存 compressed、remote/packed tensor、work 或 CUDA event；backward 按静态 plan 重收输入并重算压缩条，不重算 top-k 或 sparse forward。
- work、event、remote buffer 和 saved-state 都属于单次 autograd 调用，不放入 runtime 共享静态状态。
- d_sink 与 runtime replicated 参数梯度由 DSA 内部 GroupReduce，并标记为 CP-reduced，外层不得对同一 CP group 重复归约。
- 步骤 7 的 `DsaOverlapConfig` 独立控制 compressed GroupCast/Indexer projection 与 dKi GroupReduce/sparse backward 两组 overlap。CP=8 出口必须同时检查 2×2 开关矩阵、两个 in-flight microbatch、reentrant/retain-graph backward、gradient accumulation 和空 rank，并分别实际执行 A2AV 与 native。旧 CP=2 对真正嵌套 backward 的通过结果只保留为兼容历史，不能证明 CP=8 的 payload-private 状态与 grpcoll stream 串行语义。
- CP=1 runtime 支持完整 module deepcopy/`torch.save`，反序列化时重建锁并清空 device/process-group 绑定的 forward-plan cache。CP>1（包括 CP=8）的 `ProcessGroup` 是 PyTorch 不可 pickle 的外部状态，只支持保存 `state_dict`，再在目标进程组上新建 runtime 并加载。
- 正常退出和协调可恢复异常会 drain 已发起 work；collective wait 或 rank-local compute/kernel 失败会在公共前缀 drain 后 best-effort 请求 `ProcessGroup.abort()`。真实 B300 探针表明该调用不能保证解除 peer 已进入的 CUDA stream wait，因此不可恢复 CUDA/NCCL/NVSHMEM 故障采用 fail-stop 语义，必须由外部 launcher 在 60 秒 watchdog 内终止全部 worker；不承诺同一进程组恢复。步骤 7 测试覆盖真实协调异常 drain/reuse 与 abort 请求，隔离单 rank 故障的 launcher teardown 留给步骤 9 最终子进程矩阵。

## 正式测试与当前命令

正式测试只从当前 worktree 的 `tests/test_dsa/` 运行。宿主机权威 worktree 是 `/home/scratch.wewen_gpu/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll`；不再从旧 native 镜像的 `/tmp/native-test/test_dsa_cp.py` 或主 checkout 的同名文件运行。下面是本轮步骤 1–7 的 CP8 复验命令；CP=1 命令只生成 oracle 证据，分布式出口由同一个 8-rank 组完成。增量 native 容器不是步骤 8/9 尚待构建的最终不可变 native 镜像。

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
# 单节点 NVLink 路径使用 num_rdma_bytes=0；NVSHMEM_SYMMETRIC_SIZE 必须 unset/N/A。
docker exec \
  -e CUDA_VISIBLE_DEVICES="${GPU_GROUP}" \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29617 \
  -w /workspace "${NATIVE_CONTAINER}" \
  bash -lc 'unset NVSHMEM_SYMMETRIC_SIZE; \
    test -z "${NVSHMEM_SYMMETRIC_SIZE+x}"; \
    timeout 5400 pytest -q -rs tests/test_dsa/test_dsa_cp.py'
```

当前 `test_dsa_cp.py` 不再是 transport-only：CP=8 目标矩阵包含 A2AV/native transport、reference/kernel forward、reference/kernel backward/saved-state，以及步骤 7 的 overlap、并发、reentrant/retain-graph、空 rank、协调异常 drain/reuse 和 abort-request 用例。其中所有 `test_native_grpcoll_*` 用例必须在安装了 `magi_attn_comm` 和 NVSHMEM 的 B300 镜像内实际执行；`-rs` 输出中出现 skip 即为验收失败。CP=2 节点只算兼容回归。`test_dsa_megatron.py` 同样必须加载只读挂载且 HEAD 精确为 `c6449f0b23be397449f21c0967c5fc90785e55ea` 的 checkout，不能以 skip 代替 parity。编译后的单 kernel 测试使用 30 秒 watchdog；CP8 transport、完整路径/并发和首次 sparse-backward 冷编译分别使用 180、600–900 和 1200 秒预算，隔离故障的 launcher fail-stop 硬期限仍为 60 秒。

2026-07-12 的当前 CP8 结果：API `32 passed`，dispatch+solver `36 passed`，packing+Megatron `18 passed`，单个八卡 `test_dsa_cp.py` `27 passed, 0 skipped`（637.08 秒），正式目录合计 `113 passed, 0 skipped`；旧回归为单卡 `17 passed`、CP8 `2 passed`。CP8 文件覆盖 A2AV/native、7168-wide reverse、三 ratio reference/kernel forward/backward、全部梯度、7 空 rank、2×2 overlap、并发/reentrant、retain-graph 和协调异常 drain/reuse。Black、isort、Ruff 0.12.5、compileall 与 `git diff --check` 通过。

步骤 8 的性能校准、最终 solver 系数回填、固定 20 packs 和 E2E 门槛没有运行；步骤 9 的最终不可变镜像、launcher 隔离故障矩阵、集成与原子提交也没有运行。当前步骤 1–7 证据不得被解读为步骤 8/9 已完成。

## grpcoll 探针保存

原型 worktree `agents/worktrees/magi-dsa-v4` 的未提交 grpcoll 探针没有被覆盖或 reset；完整可应用补丁保存在 `agents/backups/magi-dsa-v4-grpcoll-probe-20260709.patch`，SHA256 为 `b6a6b8fadb48a38dc2c9b38bb1abd1fab8d5d86f27fedb67d298bded836416ee`。
