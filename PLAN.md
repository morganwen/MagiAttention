# PLAN.md：Magi_DSA V4 唯一任务与实施计划

## 文档规则

- 本文件是唯一有效的任务合同和执行计划，不再维护 `AGENTS.md`。
- “冻结合同”只有用户可以修改；“当前状态”和“实施步骤”由执行者随进度更新。
- 每一步完成后记录 worktree、commit、测试命令、结果和报告路径。未通过出口测试不得进入下一步。

## 冻结合同

### 目标与范围

- 在 MagiAttention 增加独立 `MagiDSARuntimeMgr + calc_dsa` 旁路，完成 DeepSeek V4 hybrid attention 的 BF16 训练 forward/backward。
- 验收硬件仅为当前单节点 8× NVIDIA B300 SXM6 AC（SM103，compute capability 10.3）。正式分布式验收固定为一个覆盖 GPU `0..7` 的 `world_size=8` 进程组，且 `cp_size=8`；不得拆成四个 CP=2 pair。CP=1 只用于本地 API 与数值 oracle，既不替代 CP=8 验收，也不参与性能门槛。CP=2 保留为兼容路径但不作为本轮出口，多机不作为完成条件。
- 从本次硬件冻结变更起，步骤 1–7 的出口和最终验收证据必须来自上述 B300/SM103 节点。此前 SM90/H100 的测试、profile 和镜像记录只保留为实现历史，不得单独证明任何当前出口；无需维持旧 SM90 数值或性能结果。
- 支持 packed 变长 causal batch、MQA latent shared-KV，以及 ratio=0、4、128 三种固定层形态。一个实例只服务一种 ratio。
- ratio=0：纯 window + sink；ratio=4：重叠 compressor + lightning Indexer top-k + window + sink；ratio=128：非重叠 compressor + 完整压缩 causal prefix + window + sink。
- V1 不做 dense warm-up、真实模型接入、TP、FP8/FP4、decode/cache、attention recompute、CUDA graph、同调用混合 ratio、top-k 驱动的动态 KV 通信。

### 张量与数值合同

- 结构化 `MagiDSAInput` 包含 `x`、`qr`、主 `q`、`latent_kv`、FP32 `sink` 和 packed metadata，不使用十几个无名位置参数。
- 主 Q 为 `[T,64,512]`；latent shared-KV 为 `[T,512]`，Hkv=1；输出 O 为 `[T,64,512]`。V 使用完整 512 维。
- Q/KV 的最后 64 维已经由模型侧做 partial RoPE；runtime 内生成的压缩条按 block id 做 RoPE。
- Indexer 为 Hidx=64、Didx=128、topk=512；window=128；sink 为 `[64]` FP32 可学习参数。
- ratio=4 的位置 `p` 只能看到 `floor((p+1)/4)` 个完整压缩块；ratio=128 同理。sample 尾部不足一个完整块时不生成压缩条。
- top-k 按 score 降序、同分按小 block id；`topk_idx` 为 `[T,512]` int32 global logical block id，合法项连续在前，余下连续填 `-1`。top-k 全程留在 device，不允许 `.item()` 或隐式 D2H。
- forward 返回 `(O, kl_loss)`。CP 下每 rank 的可微 KL 为 `local_kl_sum / global_query_tokens`；日志值对 detached scalar 用 GroupReduce 汇总。
- backward 产生 dx、dqr、dQ、owner-local dKV、compressor/Indexer 参数梯度和 d_sink。Indexer 输入分支 detach，KL 不向 trunk x/qr 回传。
- d_sink 和 runtime 内 replicated 参数梯度由 DSA 内部 GroupReduce，结果标记为 CP-reduced；外层不得沿同一 CP group 再次归约。
- 窗口和压缩条对同一原 token 的梯度先用 FP32 accumulator 合并，最后转回目标 dtype。

### Kernel 与通信合同

- sparse forward 使用 FlashMLA nv_dev；sparse backward、Indexer、top-k、score recompute 和 Indexer backward 使用 cudnn-frontend deepseek_sparse_attention。
- 外部 kernel 依赖不 fork、不改源码；确需 patch 必须先经用户确认。
- 新 packing/remap/CSR kernel 用 `magi_attention/kernel/cutedsl` 风格的 CuTe DSL；本阶段只在 B300/SM103 验收。public frontend、device mapping schema 和编译缓存不得硬编码单一架构；kernel 只使用通用 global copy、寄存器 FP32 累加和 128-bit vector copy，编译 key 必须包含 GPU arch，并在验收报告中记录实际 key 为 SM103；不复制 Magi-MSA 的专用 kernel。
- 正式 CP 数据交换只调用 `group_cast` / `group_reduce`。DSA 源码不得直接用 torch P2P、all-gather、all-reduce 或 `all2all_v` 替代。
- compressed KV、compressed Ki、window KV 和 compressor overlap x 使用独立 typed payload、metadata、buffer slot 和 work handle。
- compressed KV/Ki 静态 GroupCast 到所有 CP peers；不根据 top-k 动态选择通信对象。
- window KV 和 overlap x 根据 sample/fragment 静态 transfer table 只传 unique rows，不假设 fragment 位于相邻 rank。
- backward 使用与 forward 对称的 GroupReduce。所有 rank 以一致顺序进入 collectives；空路线也传合法零长度 metadata。
- work、CUDA event、remote/packed buffer 和 saved-state 属于单次 autograd 调用，不能存入并发调用共享的 runtime 静态对象。

### Saved-state 与可靠性合同

- forward 只保存 O、FP32 LSE、topk_idx、topk_length 和 compressor gate 必需中间量；不保存压缩条、remote/packed tensor、work 或 event。
- backward 按静态 plan 重收原始输入并重算压缩条；不重算 Indexer top-k 或 sparse forward。
- kernel 单次执行超过 10 秒视为死锁，编译后的 kernel 单测使用 30 秒 watchdog；CP 通信测试默认 60 秒 watchdog。
- 正常退出和可恢复异常必须 wait/drain 已发起 work。collective wait 失败或 rank-local compute/kernel 异常时，DSA 在 drain 公共前缀后 best-effort 调用 `ProcessGroup.abort()`（不可用时尝试 destroy），不伪造 cancel，也不承诺 CUDA/NCCL/NVSHMEM 异步故障后同一进程组可恢复。
- 单 rank 不可恢复故障采用 fail-stop 语义：外部 launcher 必须用 60 秒 watchdog 终止全部 worker；进程内 abort 只是尽力释放 peer，不能替代 launcher。必测 gradient accumulation、两个并发 microbatch、reentrant backward、空 rank、协调可恢复异常、abort 请求和提前退出；真实隔离故障/launcher teardown 纳入步骤 9 的独立子进程矩阵。

### 性能合同

- 固定 workload：CP=8 全局 196608 tokens，每 rank 目标 24576。
- DatasetSampler 使用 seed=42、pack_num=20、chunk_ratio=0.25；baseline 和 candidate 使用完全相同的 packs、kernel、dtype 和计时协议。
- sequential baseline 为 128 对齐连续分配、GroupCast/GroupReduce、模块严格串行；candidate 为 balanced fragments 加可验证 overlap。
- ratio=4 的 Indexer 时间包含 `indexer_forward + top-k + score_recompute + indexer_backward`。
- CP=8 必须满足：

```text
mean_pack(max_rank_indexer_time)_balanced
<
mean_pack(max_rank_indexer_time)_sequential
```

- Indexer 单项报告每个 pack 的 `max/mean-1`，但不设单项 5% 绝对门槛。
- 完整路径硬门槛：CP=8 candidate 的 `mean_pack(max_rank_e2e)` 小于对应 baseline；CP=8 每个 pack 的完整路径 `max/mean-1 <= 5%`。
- ratio=4/128 执行性能硬门槛；ratio=0 只报告。必须给出 2×2 overlap 消融和最慢 rank 的真实重叠时间。

## 冻结基线

- MagiAttention：`529fb0a4e273b3557a56d8afd60b74da46688095`。
- 当前 DSA 计算原型：worktree `agents/worktrees/magi-dsa-v4`，commit `98e043cafebbdcbce6835e26849d96624136ef70`。
- Magi-MSA 结构参考：`/home/scratch.wewen_gpu/Magi-MSA`，commit `9670ae222b4a93ce63994982c111604f9f0d6265`。
- Megatron dsv4 数值参考：`/home/scratch.wewen_gpu/megatron-lm`，commit `c6449f0b2`。
- FlashMLA：`b7643bd54521f563b839b98289b5cd048c062ba2`；`nvidia-cutlass-dsl==4.5.2`。
- cudnn-frontend：`f00538322e9d3d439fe8c5f3144644e58ee66823`（`nvidia-cudnn-frontend==1.27.0`）；fast-hadamard-transform：`e7706faf8d1c3b9f241e36860640ad1dac644ede`。
- CUTLASS 子仓：`81a43e6d92cdd8c20d22392f9579604ed5f710a1`；FA4 子仓：`ee1d15159cda6f3f97bfab9e487da146a8254970`。
- DeepSeek V4 官方 HF config：`deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`。
- 当前 CP8 步骤 1–7 开发测试镜像：`magi-dsa-b300-step7:dev`，image id `sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da`；NGC 26.06 基础镜像 digest：`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`。native 复验容器 `magi-dsa-b300-cp8-native` 在该 image 上安装 `nvidia-nvshmem-cu13==3.6.5` 并挂载当前源码/SM100-family 扩展；这仍不是步骤 8/9 所需的最终 clean revision、不可变 native 镜像。步骤 8 性能/步骤 9 最终验收仍须重新构建并回填 pinned production/native image id 与完整构建清单。

## 目标代码位置

```text
magi_attention/api/dsa_attn_interface.py       # 公共入口
magi_attention/dsa_runtime_mgr.py              # 静态 plan/runtime
magi_attention/functional/dist_dsa.py          # forward/backward autograd 编排
magi_attention/functional/dsa_comm.py          # GroupCast/GroupReduce 执行
magi_attention/meta/collection/dsa_meta.py     # fragments/rank plan dataclass
magi_attention/meta/solver/dsa_dispatch.py     # transfer/packing plan
magi_attention/meta/solver/dsa_solver.py       # Indexer-balanced solver
magi_attention/kernel/cutedsl/dsa_pack.py      # arch-aware copy/remap/CSR；B300/SM103 验收
```

当前 `experimental/dsa_v4` 保留为计算原型；runtime 稳定前不同时做目录搬迁。

## 现有 Kernel 位置与复用边界

- MagiAttention 的统一 wrapper 已在 `agents/worktrees/magi-dsa-v4/magi_attention/experimental/dsa_v4/kernels.py`：
  - `_KernelSparseAttn`：FlashMLA sparse forward + cuDNN sparse backward/d_sink。
  - `indexer_select_kernel`：cuDNN Indexer forward + top-k。
  - `_KernelIndexerKL` / `indexer_kl_loss_kernel`：cuDNN score recompute + Indexer backward。
- FlashMLA 冻结源码在 B300 开发/验收镜像的 `/opt/FlashMLA`，仅构建 SM100 family，Python 入口是 `flash_mla_sparse_fwd`；实际报告必须记录 `sm_100` cubin，并证明它已在 compute capability 10.3 的 B300 上装载和执行，不暗示存在单独的 `sm_103` cubin。
- cuDNN DSA 安装在镜像的 `/usr/local/lib/python3.12/dist-packages/cudnn/deepseek_sparse_attention/`，使用其中的 `indexer_forward`、`indexer_top_k`、`score_recompute`、`indexer_backward` 和由已安装版本为 SM103 选择的 sparse-attention backward；计划和生产代码不得绑定旧的架构文件名。
- compressor 和 Indexer projection 当前在 `experimental/dsa_v4/compressor.py`、`indexer.py`，是 PyTorch module；compressor backward 由 autograd 产生，不新增 compressor kernel，除非性能实测证明需要并经计划更新。
- 步骤 5/6 只把现有 wrapper 接入新的 CP runtime，不重写或 fork FlashMLA/cuDNN kernel。唯一计划内新增的计算辅助 kernel 是步骤 4 的 `dsa_pack.py`，负责 packing、remap 和 FP32 CSR reduction。

## 正式测试目录与职责

最终测试固定放在：

```text
tests/test_dsa/
├── README.md
├── README_zh.md
├── test_dsa_api.py
├── test_dsa_cp.py
├── test_dsa_dispatch.py
├── test_dsa_megatron.py
├── test_dsa_solver.py
└── test_dsa_pack_kernel.py
```

- `README.md` / `README_zh.md`：记录不可变 B300 镜像、单个 8 卡 CP 组、SM103 要求、编译与运行分离、watchdog、环境变量、CP=1 oracle/CP=8 命令和测试过滤方式。
- `test_dsa_api.py`：CP=1 public API、三形态、reference/kernel、forward/backward、compressor、Indexer、sink、KL、packed、saved-state 和 projection/RoPE 梯度链。
- `test_dsa_cp.py`：DistTestBase `world_size=8`，对比同一 global input 的 CP=1 reference；覆盖 sequential/balanced、8-rank 非连续 fragments、GroupCast/Reduce、全部梯度、2×2 overlap、并发、协调异常 drain/reuse 和 best-effort abort 请求。
- `test_dsa_dispatch.py`：fragment 覆盖、128 对齐、block owner、transfer table、window/overlap 路由、恢复顺序、空 rank 和随机计划。
- `test_dsa_megatron.py`：对固定 `megatron-lm@c6449f0b23be397449f21c0967c5fc90785e55ea` 做 ratio=4 compressor forward parity；验收命令必须挂载该 checkout，revision/dependency 不符造成的 skip 视为失败。
- `test_dsa_solver.py`：sample-relative 成本公式、ratio 分形态、确定性、plan cache、Indexer 比较门槛和 overlap-aware E2E 目标。
- `test_dsa_pack_kernel.py`：B300/SM103 copy/remap/CSR kernel 与 torch reference 对拍，覆盖零 row、重复 source、极端 fan-in、tail、越界、实际编译 arch 和 watchdog。
- 数值测试使用 `magi_attention.testing.precision` 校准容差；CP 测试使用 `DistTestBase`，不为 CP 单独放宽容差。
- 当前 `tests/test_attn/test_dsa_v4.py`、`test_dsa_v4_cp.py` 在新目录测试全绿前保留；随后把有效用例迁入前两个文件，避免双份长期维护。
- `agents/tests/magi-dsa-v4` 只算探索证据。kernel、Megatron parity、packed CP 等稳定用例必须迁入 `tests/test_dsa` 才算正式验收。

## 当前状态

- 当前结论（2026-07-12）：步骤 1–7 已按用户要求在单个 GPU0–7、`world_size=cp_size=8` 组上完成 B300/SM103 复验；先前四个 CP2 pair 的结果继续只作历史。验收代码是 branch `magi-dsa-v4-plan-grpcoll` 上以 `b9a75402320d0f067380001180804c6fddf5ff93` 为基线的当前未提交 diff；本轮未代用户创建 commit。
- 环境：image `sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da`，PyTorch `2.13.0a0+8145d630e8.nv26.06`、CUDA 13.3、8× B300 capability `(10,3)`、全 GPU pair `NV18`。容器内安装 `nvidia-nvshmem-cu13==3.6.5`；`magi_attn_ext`/`magi_attn_comm` SHA256 分别为 `0427073e7a0f16450bade229528638bfd1c9f610bf31d00f04d7f7899ffa0eaf` / `ca98aa8439007b647c52b8aa4e18b6b0beb631593846445bd877528346c87030`。native handle 实测为 `GrpCollIntraHandle`，`num_rdma_ranks=1`，没有 A2AV fallback。
- CP8 native 配置固定 `num_sms=20`、`num_nvl_bytes=1073741824`（每个 `(group,buffer_name)` 1 GiB）、`num_rdma_bytes=0`，`NVSHMEM_SYMMETRIC_SIZE` unset/N/A。7168-wide BF16 GroupCast/FP32 GroupReduce 的 7-way fan-out/fan-in 逐元素精确；五个常驻 buffer 含 workspace 的静态下限为 5.15625 GiB/GPU，native full backward 已实际分配并通过。

| 步骤 | 当前 B300/SM103 出口证据 |
| --- | --- |
| 1 | CP=1 API/数值 oracle、ratio 0/4/128 reference/kernel、packed、全部梯度、saved-state 与序列化通过；runtime 明确接受 CP8。 |
| 2 | dispatch/solver 与成本模型通过；含 1000 随机 plans、CP8 broadcast cost，以及全局 196608-token 单样本/64 等长/20、80 ragged packed 门禁。ragged Indexer `max/mean` 最坏不超过 1.0014，求解约 1.3 秒且 fragment `<=64`。 |
| 3 | CP8 A2AV/native 四类 GroupCast/GroupReduce、零长度路线、非连续/非相邻 owner、7168-wide reverse 与 backend handle 断言全部零 skip。 |
| 4 | B300 packing/remap/FP32 CSR、tail/fan-in/sentinel/越界和 Megatron 固定 revision parity 通过。 |
| 5 | CP8 reference/kernel 完整 forward、ratio 0/4/128、sequential/balanced、packed/非连续 fragments 与 CP=1 oracle 对拍通过。 |
| 6 | CP8 reference/kernel/native 完整 backward、四路反向 GroupReduce、owner/replicated 参数与 sink 梯度、最小 saved-state、短样本可微零 KL 通过。 |
| 7 | CP8 2×2 overlap、gradient accumulation、双 in-flight、A2AV/native reentrant、retain-graph、7 空 rank、提前退出、协调 collective drain/reuse 与 abort-request 通过。 |

- 当前正式结果：`test_dsa_api.py` `32 passed`，dispatch+solver `36 passed`，packing+Megatron `18 passed`，单个 CP8 `test_dsa_cp.py` `27 passed, 0 skipped`（637.08 秒），合计 `113 passed, 0 skipped`。旧回归为单卡 `17 passed`、CP8 `2 passed`，合计 `19 passed`。Black、isort、Ruff 0.12.5、compileall 和 `git diff --check` 通过；静态扫描只允许 packed `cu_seqlens` 边界 D2H，DSA production orchestration 没有直接 torch collective。
- 历史步骤 0–6 commits 和旧 H100/早期 B300 记录保留用于追溯，但出口以本表的当前 B300 零 skip 复验为准。`magi_comm.py` 技术探针未被覆盖/reset；备份仍为 `agents/backups/magi-dsa-v4-grpcoll-probe-20260709.patch`（SHA256 `b6a6b8fadb48a38dc2c9b38bb1abd1fab8d5d86f27fedb67d298bded836416ee`）。
- 完成口径：步骤 1–7 表示 DSA 旁路在开发 worktree 中已功能接通；步骤 8 表示达到冻结的 B300 CP8 性能门槛；步骤 9 表示形成可安装、可从公共 API 使用且可合并的 MagiAttention 库级支持。真实模型接入仍不在本轮范围内。
- 步骤 8、9 本轮均未运行：没有 B300 性能校准/20-pack timing、最终不可变 native 镜像、launcher 隔离故障矩阵或集成交付结果。

## 具体实施步骤

### 步骤 0：清理并冻结设计输入

- 修改：本 `PLAN.md`、`docs/magi_dsa_v4_design.md`。
- 历史动作：删除旧 P2P/allgather 正式方案描述；冻结 HF revision、子仓状态、当时的 H100 开发镜像和测试命令；查明 cudnn-frontend 精确 commit 并把 Dockerfile 改为 commit pin；保存现有 dirty grpcoll 探针。硬件/镜像冻结现已由本文件的 B300 条款取代。
- 出口：所有 revision 可复现，设计文档与本计划一致，主 worktree 用户改动未丢失。

完成记录（2026-07-09）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤基线 `b3a7aa2df6057f4ae7a512587e284a9ea487e97d`。
- commit：`3c49075f`（`Freeze reproducible DSA design inputs`）。
- 验证：重建 `magi-dsa-dev:v1`/`v2`；`pip check` 返回 `No broken requirements found`；镜像内核对 cudnn-frontend/FHT/FlashMLA/CUTLASS DSL revision。
- 报告：`docs/magi_dsa_v4_design.md`、`agents/docker/magi-dsa-*/Dockerfile`、`agents/backups/magi-dsa-v4-grpcoll-probe-20260709.patch`。

### 步骤 1：建立公共 API 和 runtime 空骨架

- 新增：`api/dsa_attn_interface.py`、`dsa_runtime_mgr.py`、`functional/dist_dsa.py`。
- 新增测试：`tests/test_dsa/README.md`、`README_zh.md`、`test_dsa_api.py`；先迁入当前单卡三形态、packed、sink 和 compressor 用例。
- 动作：定义 `MagiDSAInput`、`DsaPackedMeta`、`MagiDSARuntimeMgr`、`calc_dsa`；CP=1 调用现有 `MagiDSAV4` kernel path，并让 runtime 显式接受验收用 `cp_size=8`（CP2 仅保留兼容）。
- 测试：字段 shape/dtype/config 拒绝测试；CP=1 三种 ratio 的 O、KL 和全部梯度与当前 API 一致。
- 出口：公共接口可运行，后续不再直接调用 `forward_cp(_packed)`。

完成记录（2026-07-09）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 0 commit `3c49075f`。
- commit：`5bc0ab3d`（`Add public Magi DSA runtime API`）。
- 历史测试（非当前验收）：旧冻结镜像 `sha256:9c51e29d1fda8fc1a6e2a8e16c7b0773309e91dcbd44cf6d0182e5e3327c1029` 上 `test_dsa_api.py` 为 `21 passed`，旧回归为 `17 tests OK`；当前出口以另行追加的 B300/SM103 复验记录为准。
- 静态检查：Black 与 isort 对新增 Python 文件检查通过；`python -m compileall` 通过。
- 历史报告：`tests/test_dsa/README.md`、`tests/test_dsa/README_zh.md`；当时 CP=2 只构造 `DsaStaticPlan(communication_ready=False)`，未引入 collective。当前 CP8 出口以“当前状态”重新回填的正式结果为准。

### 步骤 2：实现 fragment plan 和 Indexer solver

- 新增：`meta/collection/dsa_meta.py`、`meta/solver/dsa_dispatch.py`、`meta/solver/dsa_solver.py`。
- 新增测试：`tests/test_dsa/test_dsa_dispatch.py`、`test_dsa_solver.py`。
- 动作：
  1. 定义 `DsaFragmentSpec(sample_id,q_begin,q_end)`、compressed block owner、rank plan 和 restore map。
  2. 实现 128 对齐 sequential plan。
  3. 实现 ratio=4 成本：位置 `p` 的可见块数为 `floor((p+1)/4)`，使用 sample 内位置。
  4. 按成本降序分配 fragment，再用 move/swap/128 对齐 split/merge 降低最慢 rank Indexer 成本；约束 token 数、显存和 fragment 数。
  5. 在最优 Indexer slack 内选择完整预测 E2E 最小的 plan；rank 0 确定性求解并广播。
- CP8 规模门禁：对全局 196608 tokens 的单样本、等长多 sample 和 ragged packed 输入，atom-level LPT 超出 64-fragment 预算时使用 sample-aware coarse-LPT，再展开回 128-aligned atoms；大规模候选禁止进入无收益的 O(n²) refine。必须同时证明求解时延有界、每 rank fragment `<=64`、token 约束成立且 ratio=4 Indexer `max/mean-1 <= 1%`。
- CP8 cost/memory model 显式计入 compressed owner 的 `(cp_size-1)` peer fan-out、remote receive rows、owner send buffer 和每 rank 全局 compressed KV/Ki 驻留；ratio=4 使用 KV+Ki 字节，ratio=128 仅使用 KV。步骤 8 只校准这些系数，不得再改变公式。
- 测试：1000 组随机 packed plans；覆盖无重叠/空洞、128 对齐、多 sample、非连续 fragments、空 rank、block owner、稳定 plan hash。
- 出口：示例 `[0:256)+[768:1024)` / `[256:768)` 能均衡 scan cost；sequential/balanced report 可读且重复运行一致。

完成记录（2026-07-09）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 1 记录 commit `5bc8f3a9`。
- commit：`68bc8491`（`Add deterministic DSA fragment load balancer`）。
- 实现：sample-relative immutable fragment/rank plan、compressed block owner、分段 restore map、window/overlap unique-row transfer table、128 对齐 sequential baseline、ratio=4 Indexer cost 降序分配与 move/swap/split/merge refinement、token/显存/fragment 约束、Indexer slack 内完整 E2E 选择、stable SHA256 plan hash、runtime cache 和 rank-0 plan 广播。
- 历史测试（非当前验收）：旧冻结镜像上的 dispatch/solver、API、旧回归与静态检查曾通过；当前出口必须在 B300/SM103 验收 revision 上按步骤 9 最终矩阵重跑并保存零 skip 报告。
- 出口证据：1024-token 示例 sequential Indexer cost 为 `[32640, 98176]`（`max/mean-1=50.098%`），balanced 为 `[65408, 65408]`（`0.000%`），输出 fragments 等价于 `[0:256)+[768:1024)` / `[256:768)`；固定 49152-token 单 sample 静态预测从 `[75491328, 226486272]`（`50.002%`）变为 `[150988800, 150988800]`（`0.000%`）。
- 报告：`docs/magi_dsa_v4_design.md`、`tests/test_dsa/README.md`、`tests/test_dsa/README_zh.md`；步骤 2 只生成 host metadata，CP tensor 数据移动仍由步骤 3 接入。

### 步骤 3：实现 GroupCast/GroupReduce 和 reference packing

- 新增：`functional/dsa_comm.py`；先用 torch reference map 验证，不进入性能计时。
- 新增测试：`tests/test_dsa/test_dsa_cp.py` 的 transport-only 用例；此时不运行完整 attention。
- 动作：为 window KV、overlap x、compressed KV、compressed Ki 分别生成 `GroupCollectiveArg`；forward GroupCast 得到 unique receive buffer，backward 用对称 GroupReduce 回 owner。
- 测试：单个 CP8 组的 transport-only；逐项验证 8 ranks 的 send/recv rows、重复 fragment FP32 求和、7 个空 rank、非相邻 owner、A2AV fallback 和 native grpcoll 语义一致。
- 出口：DSA 生产通信代码不直接调用 torch collectives/all2allv；四类 payload 的 forward/reverse 都与 host reference 一致。

完成记录（2026-07-10）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 2 图示基线 commit `870e53e5`。
- 实现 commits：`29a9ffa3`（`Add typed DSA group collective transport`）、`f81d121e`（`Align native DSA compressed index transport`）。
- 实现：新增 `functional/dsa_comm.py`，为 window KV、overlap x、compressed KV、compressed Ki 建立四套独立 typed metadata、`GroupCollectiveArg`、per-call buffer slot/work/native handle；token 路线按 transfer table 合并 destination 并 reference-pack unique owner rows，compressed KV/Ki 静态发给全部 peer；空 source/destination 路线保留显式零长度 split；backward 复用 forward route 做对称 GroupReduce，并以 FP32 communication/owner accumulator 合并后一次性转回目标 dtype。
- 生产边界检查：DSA 通信源码只调用 `group_cast` / `group_reduce`，未直接调用 torch P2P、all-gather、all-reduce、all-to-all 或 `all2all_v`；torch `index_select`/`index_copy_` reference seam 留给步骤 4 的架构感知 kernel 替换。
- 历史 CP=2 测试（非当前验收）：旧 H100 native 镜像曾得到 transport `7 passed`；该结果不计入 B300 出口。当前验收镜像必须安装 pinned `magi_attn_comm`/NVSHMEM，并在 B300 上以 native grpcoll、零 skip 重跑相同及后续完整路径矩阵。
- 回归：`pytest -q tests/test_dsa/test_dsa_api.py tests/test_dsa/test_dsa_dispatch.py tests/test_dsa/test_dsa_solver.py` 为 `48 passed`；Black、isort、compileall 通过。镜像未安装 flake8，未执行该项。
- native 修正：步骤 3 首次暴露 compressed Ki 的 BF16 D=128 不满足 grpcoll 256-element transport alignment；当前 B300 实现已统一为所有 native send/receive/reduce row 按 dtype 对齐（包括 BF16 128→256、FP32 hidden 64→128），wait 后裁回 logical row shape。replicated FP32 梯度另按不超过 4096 elements 的对齐 row 分块，避免单条超宽 row 放大 native scratch；未修改/fork grpcoll kernel。
- 图示：`docs/assets/dsa_step3_comm_before.{dot,svg,png}`、`docs/assets/dsa_step3_comm_after.{dot,svg,png}`；设计说明同步在 `docs/magi_dsa_v4_design.md`。

### 步骤 4：实现架构感知 packing/remap/CSR kernel（B300/SM103 验收）

- 新增：`kernel/cutedsl/dsa_pack.py` 和正式 kernel 单测。
- 新增测试：`tests/test_dsa/test_dsa_pack_kernel.py`。
- 动作：
  1. 实现 device-resident int32 destination→source row copy map、logical block/token id→local packed row remap LUT，以及 FP32 CSR reduce map；静态 host plan 负责上传前的完整覆盖、CSR 单调性和越界校验，动态 top-k/remap 全程留在 device。
  2. row copy 支持 BF16、FP32 和 int32，flatten trailing dimensions，覆盖 D=128、D=512 和任意 hidden width；满足对齐时走 128-bit vector copy，否则走有界 tail 路径。
  3. CSR reduce 以 FP32 source/destination accumulator 工作，固定 CSR 顺序、无 atomic；支持覆盖非空 destination row或累加到已有 accumulator，空 destination row 保持不变，最终 dtype conversion 留给调用方。
  4. device mapping 是可共享只读的静态 plan state；packed/remote tensor、output buffer、work 和 event 仍为 per-call state。CuTe 编译 cache key 包含 arch、dtype、feature width 和 operation；当前出口必须实际生成/命中 SM103 key。
  5. `functional/dsa_comm.py` 的 production GroupCast pack、GroupReduce initial row copy 和 owner restore 改用新 kernel；步骤 3 torch reference seam 保留仅用于对拍。
- 测试：与步骤 3 reference 对拍；覆盖重复 source、极端 fan-in、零 row、非连续 map、`-1` sentinel、missing logical id、tail 和静态 plan 越界拒绝；编译后单批 kernel 使用 10 秒 watchdog，完整 kernel 单测使用 30 秒 watchdog。
- 出口：CP transport 在 B300 上使用实际 SM103 编译产物，无 D2H top-k/remap，同输入重复运行稳定。

完成记录（2026-07-10）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 3 关闭记录 commit `f2248fef`。
- 实现 commit：`c9856309`（`Add device-resident DSA packing kernels`）。
- 实现：新增架构中性 `kernel/cutedsl/dsa_pack.py` public frontend 与 arch-aware CuTe compile cache；实现 BF16/FP32/int32 destination→source row copy、device-resident int32 logical-id remap、固定 CSR 顺序且无 atomic 的 FP32 reduce，并提供上传前校验的 immutable device map。`dsa_comm.py` 的 production GroupCast pack、GroupReduce initial row copy 和 owner restore 全部改用新 kernel，torch reference 仅保留测试对拍；mapping 为静态共享 state，buffer/work/native handle 保持 per-call。
- 历史 Kernel/CP 测试（非当前验收）：旧双 H100 镜像曾为 `24 passed`；当前出口必须在 B300/SM103 上确认实际 arch key/cubin，并按步骤 9 native 最终矩阵零 skip 重跑。
- 回归：同一镜像运行 `pytest -q tests/test_dsa/test_dsa_api.py tests/test_dsa/test_dsa_dispatch.py tests/test_dsa/test_dsa_solver.py` 为 `48 passed`；运行 `python tests/test_attn/test_dsa_v4.py` 为 `17 tests OK`。
- 静态检查：Black、isort、compileall 通过；镜像未安装 flake8，未执行该项。production GroupCast/GroupReduce 入口不再调用 torch `index_select/index_copy_`，这两个操作只存在于保留的 reference 函数；动态 remap 不包含 `.item()`/`.cpu()`/D2H。
- 历史实现出口：D=128、D=512、7168 hidden width、非 128-bit 整数倍 tail、重复 source、4096-way fan-in、零 row、`-1`/missing id、静态越界拒绝和重复运行稳定性已有测试；当前 B300 出口仍以步骤 1–7 复验与步骤 9 最终矩阵为准。

### 步骤 5：接通完整 forward

- 修改：`functional/dist_dsa.py`、`dsa_runtime_mgr.py` 和现有 DSA kernel wrapper。
- Kernel 复用：直接调用现有 `indexer_select_kernel`、`indexer_kl_loss_kernel` 和 `_KernelSparseAttn` 的 FlashMLA forward；本步骤不新写 sparse/Indexer kernel。
- 动作顺序：
  1. dispatch 本 rank Q/x/qr/KV fragments。
  2. 启动 window KV、overlap x GroupCast。
  3. 计算本地 compressed KV/Ki 并 GroupCast 到全组。
  4. 计算 Indexer Q/weights，执行 top-k/KL。
  5. packing/remap 后拼接 window 与压缩索引。
  6. 一次 FlashMLA sparse forward，返回 local O 和 KL contribution。
- 测试：CP=1 oracle/CP=8、三种 ratio、packed 多 sample、8-rank 非连续 fragments；比较 top-k、O、FP32 LSE 和 KL。
- 落位：单卡 oracle 用例写入 `test_dsa_api.py`，八卡用例写入 `test_dsa_cp.py`；把稳定的 `agents/tests` kernel/Megatron parity forward 用例迁入正式目录。
- 出口：sequential 与 balanced forward 都和同一 CP=1 global reference 对齐。

完成记录（2026-07-10）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 4 关闭记录 commit `7572de87`。
- 实现 commit：`9bf41223`（`Connect complete Magi DSA CP forward`）。
- 实现：`MagiDSARuntimeMgr` 增加 sequential/balanced dispatch policy 和按 plan hash/device 缓存的只读 forward plan；CP=8 按 rank fragment 顺序接收 owner-local 输入，使用四类独立 typed GroupCast，按静态 overlap/window transfer table 收集 X/KV，在 compressed block owner 上生成 KV/Ki，再按 global logical block id 重排。window 索引由 device LUT remap，ratio=4 的 top-k 保持 device-resident 并按 sample/fragment 计算 local KL contribution，ratio=128 生成完整 causal compressed prefix；最终 window 与 compressed rows 进入一次 sparse attention。work、buffer 和 receive tensor 保持 per-call，不进入 runtime cache。
- 历史 B300 验证镜像：`magi-dsa-b300-step5:dev`（image id `sha256:5ecfa64c2164301e301e8e7348aeaf12e99dde0b4d2bdf0ada7f5dfdd314425a`），基于 NGC 26.06 digest `sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`，冻结 FlashMLA/cudnn-frontend/FHT/CUTLASS DSL revision，FlashMLA 仅构建 SM100 family；实际设备为双 NVIDIA B300 SXM6 AC、compute capability 10.3。当前出口已由“当前状态”中的 step7 image/零 skip 复验取代。
- 历史测试（不计当前出口）：`pytest -q tests/test_dsa/test_dsa_api.py tests/test_dsa/test_dsa_dispatch.py tests/test_dsa/test_dsa_solver.py tests/test_dsa/test_dsa_pack_kernel.py tests/test_dsa/test_dsa_cp.py` 当时为 `76 passed`，其中分布式部分是 CP2；当前必须由 CP8 结果替换。
- 回归：`python tests/test_attn/test_dsa_v4.py` 为 `17 tests OK`；`pytest -q tests/test_attn/test_dsa_v4_cp.py` 为 `2 passed`。Black、isort、compileall 和 `git diff --check` 通过；生产 `dist_dsa.py`/`dsa_comm.py` 不含 torch P2P/all-gather/all-reduce/all-to-all，也不含动态 `.item()`/`.cpu()`/D2H。
- 边界：本步骤只关闭 forward；GroupCast 在步骤 6 前仍不承担 autograd reverse，完整 dKV、compressor/Indexer-K 参数梯度、d_sink/replicated parameter GroupReduce 和 saved-state 收紧继续属于步骤 6。

### 步骤 6：接通完整 backward 和 saved-state

- 修改：`functional/dist_dsa.py`、`functional/dsa_comm.py`。
- Kernel 复用：继续使用 `_KernelSparseAttn.backward` 的 cuDNN sparse backward/d_sink，以及 `_KernelIndexerKL` 的 score recompute/Indexer backward；compressor backward 保持 PyTorch autograd。
- 动作顺序：重收 KV/x并重算压缩条；运行 KL backward 并 GroupReduce dKi；运行 sparse backward；FP32 CSR 合并 dKV；GroupReduce 回 owner；执行两个 compressor backward；GroupReduce d_sink 和内部参数梯度。
- 测试：dx、dqr、dQ、dKV、compressor/Indexer 参数梯度、d_sink 与 CP=1 reference 对齐；saved-tensor hooks 确认未保存 compressed/remote/packed tensor；释放 forward 临时 buffer 后 backward 仍通过。
- 落位：CP=1 backward/saved-state 写入 `test_dsa_api.py`；CP=8 全梯度和 owner reduce 写入 `test_dsa_cp.py`。
- 出口：全部梯度正确，参数梯度明确为 CP-reduced，saved-state 满足冻结合同。

完成记录（2026-07-10）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 5 关闭记录 commit `071f7ad2`。
- 实现 commit：`5c374507`（`Connect complete Magi DSA backward`）。
- 实现：CP=1 和 CP=8 都由高层 custom autograd boundary 管理最小 saved-state；forward 保存 owner-local 原始输入、O、FP32 LSE、topk 和 topk_length，不保存 compressed/remote/packed payload、collective work 或 event。CP=8 backward 重新发起 window KV、overlap X、compressed KV、compressed Ki 四路 GroupCast 并重算两个 compressor；KL 使用已保存 top-k 做 Indexer score recompute/backward，dKi 经 FP32 inverse CSR 和 GroupReduce 回 owner；sparse backward 复用 cuDNN DSA ABI 产生 dQ/dKV/d_sink，window/compressed dKV 经 FP32 CSR 合并和对称 GroupReduce 回 owner。compressor backward 保持 PyTorch autograd；d_sink 和全部 runtime 参数梯度使用独立的 FP32 GroupReduce 做 replicated sum，并标记 `_magi_dsa_cp_reduced`。
- saved-state：CP=1 packed 变长路径也改为不保存 compressed KV/Ki；reference 测试 seam 仅在 backward 重算 reference sparse attention，正式 kernel 路径只消费已保存 O/LSE/top-k，不重算 sparse forward 或 Indexer top-k。saved-tensor hooks 对 CP=1/CP=8 均只观察到 5 个原始输入加 O/LSE/topk/topk_length；forward 临时通信 buffer 释放后 backward 通过。
- 历史 B300 验证：在 8× NVIDIA B300 SXM6 AC 节点上曾只使用其中 2 卡；当时镜像为 `magi-dsa-b300-step5:dev`（image id `sha256:5ecfa64c2164301e301e8e7348aeaf12e99dde0b4d2bdf0ada7f5dfdd314425a`）。该结果与后续 CP2 step7 结果都不能替代当前 CP8 出口。
- 正式测试：`test_dsa_api.py` 为 `24 passed`；dispatch/solver/pack 三文件为 `43 passed`；`test_dsa_cp.py` 的 A2AV 路径为 `11 passed`，当前镜像未安装 `magi_attn_comm`，故既有 native transport-only 用例为 `1 skipped`。CP backward 使用全局 token-mean 随机上游梯度，在相同容差下对齐 CP=1，不为 CP 单独放宽 mismatch threshold；覆盖 dx、dqr、dQ、owner dKV、全部 compressor/Indexer 参数和 d_sink。步骤 5 forward 两项在修正不稳定的 fragment-count 测试假设后单独复跑为 `2 passed`。
- 回归：`tests/test_attn/test_dsa_v4.py tests/test_attn/test_dsa_v4_cp.py` 为 `19 passed`。Black、isort、compileall 和 `git diff --check` 通过；镜像未安装 flake8。生产 `dist_dsa.py`/`dsa_comm.py` 仍不直接调用 torch P2P/all-gather/all-reduce/all-to-all；torch `index_select/index_copy_` 只存在于步骤 3 保留的 reference seam。

### 步骤 7：实现 overlap、并发和故障边界

- 修改：runtime per-call state 和 stream/event 编排。
- 动作：实现两个独立开关：compressed GroupCast 与 Indexer projection；dKi GroupReduce 与 sparse backward。为每次调用独占 work/event/buffer slot。
- 测试：2×2 开关矩阵、gradient accumulation、两个并发 microbatch、reentrant backward、空 rank、协调异常真实 collective drain/reuse、rank-local/collective 失败的 abort 请求、提前退出；正常 CP watchdog 60 秒。单 rank 异步 CUDA/NCCL/NVSHMEM 故障不要求同一进程恢复，步骤 9 由外部 launcher 在隔离子进程中验证 60 秒 fail-stop teardown。
- 落位：runtime/communication 压力用例写入 `test_dsa_cp.py`；kernel watchdog 留在 `test_dsa_pack_kernel.py`。
- 出口：四种开关数值一致，无共享状态污染；正常、提前退出和协调可恢复异常能 drain，失败路径确定发出 best-effort abort 请求。不得把进程内 `ProcessGroup.abort()` 写成对异步 native 故障的可靠恢复机制。

历史完成记录（2026-07-12，已被 CP8 纠偏撤销）：

- 实现：增加两个独立、不可变的 `DsaOverlapConfig` 开关；invocation-local `ContextVar` work tracker 与独立 buffer/event/handle；runtime/RoPE cache 锁与 CP=1 序列化恢复；native row 的通用 dtype 对齐及 logical-shape 恢复；replicated FP32 梯度按最大 4096 elements 分块；短样本 connected-zero KL；stable kernel top-k 与 K-boundary tie 修正。
- 并发：两个 microbatch 在 outer dKi GroupReduce 已发起、尚未 wait 时进入 inner `_attention_backward`；A2AV 与 native 都与两个独立 sequential CP2 backward 对拍，CP2 输入/参数梯度为 bitwise exact，累积梯度按 microbatch 平均后与 CP1 保持原容差。retain-graph、空 rank 和全部 2×2 开关同样通过。
- 可靠性：unit tests 证明正常/提前退出 drain、collective wait 失败立即请求 abort、rank-local compute 失败在 drain 公共前缀后请求 abort；真实双卡协调异常证明四路 A2AV work drain 后健康组可继续运行。真实异步非对称探针同时证明进程内 abort 不能保证释放 peer，因此出口按冻结合同采用 best-effort abort + 外部 launcher fail-stop，而不是同进程恢复。
- 当时 native-enabled 双卡环境中的完整 CP 文件（同时覆盖 A2AV，含 5 个 `test_native_grpcoll_*`）为 `26 passed, 0 skipped`。该结果不再是步骤 7 出口；当前 CP8 结果必须在“当前状态”重新回填。

### 步骤 8：B300 CP8 性能验收

状态：未执行；步骤 1–7 全绿且公共 API 形态冻结后开始。本步骤完成性能校准和验收，不再重复功能接入。

- 产物：新增 `agents/benchmarks/magi-dsa-v4-balance/` 下的 benchmark driver、README 和 report schema；结果写入 `agents/perf/magi-dsa-v4-balance/<RUN_ID>`，profile 写入 `agents/profiles/magi-dsa-v4-balance/<RUN_ID>`。具体命令、环境检查和字段定义放在 README/schema，不在本计划重复。
- 前置：先补齐并冻结公共 config 导出和安装 smoke，使用户无需依赖 `experimental` import；再使用 clean pinned commit、不可变 native 镜像和单节点 GPU `0..7` 的唯一 `world_size=cp_size=8` 组。实际装载 `GrpCollIntraHandle`，禁止 A2AV fallback；镜像 revision、依赖、SM103、NVLink 和冻结的 1 GiB-per-buffer 配置由 driver preflight 校验。
- 校准：按性能合同生成 seed=42 的 20 个固定 packs；在独立 calibration run 中测量 solver 所需算子/通信成本并回填系数，再从冻结系数的 clean measure commit 重建最终镜像。正式计时只重放校准，不再修改系数。
- 测量：baseline 固定为 `sequential:00`，candidate 固定为 `balanced:11`；两者使用相同 packs、kernel、dtype 和等价工作量，并先对拍输出与全部梯度。ratio=4 跑完整 2×2 overlap，ratio=128 验 E2E，ratio=0 只报告；compile、warm-up 和 measure 分离，每 pack 记录 10 次并取中位数。
- 出口：
  1. ratio=4 的 balanced Indexer 时间严格优于 sequential；
  2. ratio=4/128 的 balanced E2E 严格优于 sequential；
  3. ratio=4/128 的每个 candidate pack 均满足 rank `max/mean-1 <= 5%`；
  4. 20/20 packs、8/8 ranks、10/10 iterations 和 native backend 证据完整，ratio=4 timeline 证明两个 overlap 区间真实发生；无 skip、NaN、watchdog、OOM、JIT/cache miss 或工作量不等价。
- 证据：保存 revision/image/environment、pack hashes、raw timing、summary、固定 pack profile 和 `artifact_manifest.sha256`。measure 后若 runtime/kernel/solver 行为变化，必须使用新 revision、image 和 RUN_ID 重跑本步骤。

### 步骤 9：最终验收与交付

状态：未执行；步骤 8 全部门槛通过并冻结 artifact manifest 后开始。


- 公共接口：从安装后的 `magi_attention.api` 可稳定导入并运行 `calc_dsa`、结构化输入、runtime 和所需 config；用户示例不得依赖 `experimental` import。删除experimental。
- 最终矩阵：用 `final_matrix.json` 顺序运行 CP=1 oracle 和单个 CP8 native 零-skip 测试，覆盖三种 ratio、reference/kernel、sequential/balanced、packed forward/backward 与全部梯度、solver/packing/Megatron compressor parity、四类 GroupCast/Reduce、2×2 overlap、并发/reentrant、空 rank、drain/abort/提前退出。隔离单 rank 不可恢复故障由外部 launcher 在 60 秒内回收全部 8 个 worker。
- 文档：同步设计、公共 API/config、最小可运行示例、tensor/ratio/KL/梯度与 saved-state 合同、CP8 native 配置、故障语义、测试命令和性能结果；删除或降级尚无正式证据的 full-layer/drop-in 声明。
- 交付：保留用户现有 dirty worktree，不 reset/clean/stash；把本任务整理为可审查原子 commits，并在独立干净 integration worktree 验证 cherry-pick、最终矩阵和固定子仓指针。

## 决策记录

- 2026-07-09：只保留 `PLAN.md`，删除 `AGENTS.md`；本文件同时承担冻结合同和动态计划。
- 2026-07-09：保持独立 `MagiDSARuntimeMgr + calc_dsa` 旁路。
- 2026-07-09：ratio=4 用 128 对齐、可非连续 fragments 平衡 Indexer；Indexer 时间包含 forward、top-k、score recompute 和 backward。
- 2026-07-09：Indexer candidate 必须优于 sequential，但不设单项 5% 绝对门槛；完整路径继续执行 5% 门槛。
- 2026-07-09：KL 返回每 rank 的可微 local contribution；d_sink 和内部参数梯度由 DSA 内部 GroupReduce，结果标记为 CP-reduced。
- 2026-07-09：正式 CP 通信使用 GroupCast/GroupReduce；compressed KV/Ki 保持静态全组可见。
- 2026-07-09：步骤 5/6 复用现有 FlashMLA/cuDNN wrapper，不重写外部 kernel；新增计算辅助 kernel 仅限步骤 4 的 packing/remap/CSR。
- 2026-07-09：正式 DSA 测试按 Magi-MSA 的五类结构建立在 `tests/test_dsa`；现有测试和 `agents/tests` 稳定用例迁入后才计入验收。
- 2026-07-10（已被 2026-07-12 硬件决定取代）：步骤 4 最初以 SM90/H100 为出口，但 public frontend、静态 device mapping 和 arch-aware 编译缓存保持架构中性。
- 2026-07-12（已被下一条纠正）：曾误把 8 卡验收拆成四个 CP2 pair；对应步骤 1–7 结果全部降为历史证据。
- 2026-07-12：按用户澄清，8 卡验收必须是单个 `world_size=cp_size=8` 组；CP1 只作 oracle，CP2 只作兼容且不计出口。步骤 1–7 必须以 CP8 在 B300 重跑，步骤 8/9 也只能使用唯一八卡组，禁止 pair 分片；步骤 8/9 本轮仅修改计划、不运行。
- 2026-07-12：真实 B300 A2AV 故障探针确认 `ProcessGroup.abort()` 不能保证解除已经卡在 CUDA stream wait 的 peer。冻结异常合同据此改为 best-effort abort + 外部 launcher 60 秒 fail-stop watchdog；协调异常继续验证 drain/reuse，不再声称不可恢复 native 故障可在同一进程组恢复。
- 2026-07-12：步骤 8/9 的顶层计划只保留前置、动作、硬门槛和出口；命令、schema 字段、容器挂载与 Git 操作细节下沉到 benchmark README/validator。完成口径限定为 MagiAttention 库级 DSA 支持，真实模型接入另立任务。
