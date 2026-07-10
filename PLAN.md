# PLAN.md：Magi_DSA V4 唯一任务与实施计划

## 文档规则

- 本文件是唯一有效的任务合同和执行计划，不再维护 `AGENTS.md`。
- “冻结合同”只有用户可以修改；“当前状态”和“实施步骤”由执行者随进度更新。
- 每一步完成后记录 worktree、commit、测试命令、结果和报告路径。未通过出口测试不得进入下一步。

## 冻结合同

### 目标与范围

- 在 MagiAttention 增加独立 `MagiDSARuntimeMgr + calc_dsa` 旁路，完成 DeepSeek V4 hybrid attention 的 BF16 训练 forward/backward。
- 验收硬件仅为 SM90/H100 80GB：CP=1 用 1 卡，CP=2 用同机 2 卡。SM100、CP>2 和多机不作为完成条件。
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
- 新 packing/remap/CSR kernel 用 `magi_attention/kernel/cutedsl` 风格的 CuTe DSL；本阶段只在 SM90/H100 验收，但 public frontend、device mapping schema 和编译缓存不得绑定 SM90。kernel 只使用 SM90/SM100 共有的通用 global copy、寄存器 FP32 累加和 128-bit vector copy，编译 key 必须包含 GPU arch；不复制 Magi-MSA 的 SM100 kernel。
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
- 正常和可恢复异常必须 wait/drain 已发起 work；不可恢复 collective 故障有界销毁 process group，不伪造 cancel。
- 必测 gradient accumulation、两个并发 microbatch、reentrant backward、空 rank、kernel/collective 异常和提前退出。

### 性能合同

- 固定 workload：CP=1 全局 24576 tokens；CP=2 全局 49152 tokens，每 rank 目标 24576。
- DatasetSampler 使用 seed=42、pack_num=20、chunk_ratio=0.25；baseline 和 candidate 使用完全相同的 packs、kernel、dtype 和计时协议。
- sequential baseline 为 128 对齐连续分配、GroupCast/GroupReduce、模块严格串行；candidate 为 balanced fragments 加可验证 overlap。
- ratio=4 的 Indexer 时间包含 `indexer_forward + top-k + score_recompute + indexer_backward`。
- CP=2 必须满足：

```text
mean_pack(max_rank_indexer_time)_balanced
<
mean_pack(max_rank_indexer_time)_sequential
```

- Indexer 单项报告每个 pack 的 `max/mean-1`，但不设单项 5% 绝对门槛。
- 完整路径硬门槛：CP=1、CP=2 candidate 的 `mean_pack(max_rank_e2e)` 都小于对应 baseline；CP=2 每个 pack 的完整路径 `max/mean-1 <= 5%`。
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
- H100 开发镜像：`magi-dsa-dev:v2@sha256:9c51e29d1fda8fc1a6e2a8e16c7b0773309e91dcbd44cf6d0182e5e3327c1029`；NGC 26.05 基础镜像 digest：`sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`。

## 目标代码位置

```text
magi_attention/api/dsa_attn_interface.py       # 公共入口
magi_attention/dsa_runtime_mgr.py              # 静态 plan/runtime
magi_attention/functional/dist_dsa.py          # forward/backward autograd 编排
magi_attention/functional/dsa_comm.py          # GroupCast/GroupReduce 执行
magi_attention/meta/collection/dsa_meta.py     # fragments/rank plan dataclass
magi_attention/meta/solver/dsa_dispatch.py     # transfer/packing plan
magi_attention/meta/solver/dsa_solver.py       # Indexer-balanced solver
magi_attention/kernel/cutedsl/dsa_pack.py      # SM90 copy/remap/CSR
```

当前 `experimental/dsa_v4` 保留为计算原型；runtime 稳定前不同时做目录搬迁。

## 现有 Kernel 位置与复用边界

- MagiAttention 的统一 wrapper 已在 `agents/worktrees/magi-dsa-v4/magi_attention/experimental/dsa_v4/kernels.py`：
  - `_KernelSparseAttn`：FlashMLA sparse forward + cuDNN sparse backward/d_sink。
  - `indexer_select_kernel`：cuDNN Indexer forward + top-k。
  - `_KernelIndexerKL` / `indexer_kl_loss_kernel`：cuDNN score recompute + Indexer backward。
- FlashMLA 冻结源码在开发镜像 `magi-dsa-dev:v2:/opt/FlashMLA`；SM90 sparse forward 本体在 `csrc/sm90/prefill/sparse/`，Python 入口是 `flash_mla_sparse_fwd`。
- cuDNN DSA 安装在镜像的 `/usr/local/lib/python3.12/dist-packages/cudnn/deepseek_sparse_attention/`，使用其中的 `indexer_forward`、`indexer_top_k`、`score_recompute`、`indexer_backward` 和 `sparse_attention_backward/dsa_bwd_sm90.py`。
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
├── test_dsa_solver.py
└── test_dsa_pack_kernel.py
```

- `README.md` / `README_zh.md`：记录镜像、H100 要求、编译与运行分离、watchdog、环境变量、单卡/双卡命令和测试过滤方式。
- `test_dsa_api.py`：CP=1 public API、三形态、reference/kernel、forward/backward、compressor、Indexer、sink、KL、packed、saved-state 和 projection/RoPE 梯度链。
- `test_dsa_cp.py`：DistTestBase CP=2，对比同一 global input 的 CP=1 reference；覆盖 sequential/balanced、非连续 fragments、GroupCast/Reduce、全部梯度、2×2 overlap、并发和故障收敛。
- `test_dsa_dispatch.py`：fragment 覆盖、128 对齐、block owner、transfer table、window/overlap 路由、恢复顺序、空 rank 和随机计划。
- `test_dsa_solver.py`：sample-relative 成本公式、ratio 分形态、确定性、plan cache、Indexer 比较门槛和 overlap-aware E2E 目标。
- `test_dsa_pack_kernel.py`：SM90 copy/remap/CSR kernel 与 torch reference 对拍，覆盖零 row、重复 source、极端 fan-in、tail、越界和 watchdog。
- 数值测试使用 `magi_attention.testing.precision` 校准容差；CP 测试使用 `DistTestBase`，不为 CP 单独放宽容差。
- 当前 `tests/test_attn/test_dsa_v4.py`、`test_dsa_v4_cp.py` 在新目录测试全绿前保留；随后把有效用例迁入前两个文件，避免双份长期维护。
- `agents/tests/magi-dsa-v4` 只算探索证据。kernel、Megatron parity、packed CP 等稳定用例必须迁入 `tests/test_dsa` 才算正式验收。

## 当前状态

- 已完成：三形态 PyTorch reference、compressor、Indexer、FlashMLA/cuDNN kernel、packed 原型、连续 CP=2 对拍和初版 profile。
- 证据：`agents/tests/magi-dsa-v4`、`agents/profiles/magi-dsa-v4`、`agents/perf/magi-dsa-v4`。
- 未完成：SM90 packing、完整 CP forward/backward、saved-state 收紧、最终并发/死锁/性能验收。
- 当前 `magi_comm.py` 是未提交技术探针，不直接作为生产实现；重构前先提交或备份，禁止覆盖/reset。
- 已完成：步骤 0、步骤 1、步骤 2、步骤 3；grpcoll dirty probe 已保存为 `agents/backups/magi-dsa-v4-grpcoll-probe-20260709.patch`（SHA256 `b6a6b8fadb48a38dc2c9b38bb1abd1fab8d5d86f27fedb67d298bded836416ee`）。
- 步骤 3 实现 commits：`29a9ffa3`、`f81d121e`；A2AV fallback 与真实 native grpcoll 的 CP=2 transport-only 全部通过。
- 下一步：步骤 4。

## 具体实施步骤

### 步骤 0：清理并冻结设计输入

- 修改：本 `PLAN.md`、`docs/magi_dsa_v4_design.md`。
- 动作：删除旧 P2P/allgather 正式方案描述；冻结 HF revision、子仓状态、H100 镜像和当前有效测试命令；查明 cudnn-frontend 精确 commit 并把 Dockerfile 改为 commit pin；保存现有 dirty grpcoll 探针。
- 出口：所有 revision 可复现，设计文档与本计划一致，主 worktree 用户改动未丢失。

完成记录（2026-07-09）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤基线 `b3a7aa2df6057f4ae7a512587e284a9ea487e97d`。
- commit：`3c49075f`（`Freeze reproducible DSA design inputs`）。
- 验证：重建 `magi-dsa-dev:v1`/`v2`；`pip check` 返回 `No broken requirements found`；镜像内核对 cudnn-frontend/FHT/FlashMLA/CUTLASS DSL revision。
- 报告：`docs/magi_dsa_v4_design.md`、`agents/docker/magi-dsa-*/Dockerfile`、`agents/backups/magi-dsa-v4-grpcoll-probe-20260709.patch`。

### 步骤 1：建立公共 API 和 runtime 空骨架

- 新增：`api/dsa_attn_interface.py`、`dsa_runtime_mgr.py`、`functional/dist_dsa.py`。
- 新增测试：`tests/test_dsa/README.md`、`README_zh.md`、`test_dsa_api.py`；先迁入当前单卡三形态、packed、sink 和 compressor 用例。
- 动作：定义 `MagiDSAInput`、`DsaPackedMeta`、`MagiDSARuntimeMgr`、`calc_dsa`；先让 CP=1 调用现有 `MagiDSAV4` kernel path，CP=2 暂只创建 plan 不通信。
- 测试：字段 shape/dtype/config 拒绝测试；CP=1 三种 ratio 的 O、KL 和全部梯度与当前 API 一致。
- 出口：公共接口可运行，后续不再直接调用 `forward_cp(_packed)`。

完成记录（2026-07-09）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 0 commit `3c49075f`。
- commit：`5bc0ab3d`（`Add public Magi DSA runtime API`）。
- 测试：冻结镜像 `sha256:9c51e29d1fda8fc1a6e2a8e16c7b0773309e91dcbd44cf6d0182e5e3327c1029` 上运行 `python -m pytest -q tests/test_dsa/test_dsa_api.py`，结果 `21 passed`（reference/kernel、ratio 0/4/128、packed、O/KL、全部输入/参数梯度）；运行 `PYTHONPATH=. python tests/test_attn/test_dsa_v4.py`，结果 `17 tests OK`。
- 静态检查：Black 与 isort 对新增 Python 文件检查通过；`python -m compileall` 通过。
- 报告：`tests/test_dsa/README.md`、`tests/test_dsa/README_zh.md`；CP=2 只构造 `DsaStaticPlan(communication_ready=False)`，未引入 collective。

### 步骤 2：实现 fragment plan 和 Indexer solver

- 新增：`meta/collection/dsa_meta.py`、`meta/solver/dsa_dispatch.py`、`meta/solver/dsa_solver.py`。
- 新增测试：`tests/test_dsa/test_dsa_dispatch.py`、`test_dsa_solver.py`。
- 动作：
  1. 定义 `DsaFragmentSpec(sample_id,q_begin,q_end)`、compressed block owner、rank plan 和 restore map。
  2. 实现 128 对齐 sequential plan。
  3. 实现 ratio=4 成本：位置 `p` 的可见块数为 `floor((p+1)/4)`，使用 sample 内位置。
  4. 按成本降序分配 fragment，再用 move/swap/128 对齐 split/merge 降低最慢 rank Indexer 成本；约束 token 数、显存和 fragment 数。
  5. 在最优 Indexer slack 内选择完整预测 E2E 最小的 plan；rank 0 确定性求解并广播。
- 测试：1000 组随机 packed plans；覆盖无重叠/空洞、128 对齐、多 sample、非连续 fragments、空 rank、block owner、稳定 plan hash。
- 出口：示例 `[0:256)+[768:1024)` / `[256:768)` 能均衡 scan cost；sequential/balanced report 可读且重复运行一致。

完成记录（2026-07-09）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 1 记录 commit `5bc8f3a9`。
- commit：`68bc8491`（`Add deterministic DSA fragment load balancer`）。
- 实现：sample-relative immutable fragment/rank plan、compressed block owner、分段 restore map、window/overlap unique-row transfer table、128 对齐 sequential baseline、ratio=4 Indexer cost 降序分配与 move/swap/split/merge refinement、token/显存/fragment 约束、Indexer slack 内完整 E2E 选择、stable SHA256 plan hash、runtime cache 和 rank-0 plan 广播。
- 测试：冻结镜像上 `pytest -q tests/test_dsa/test_dsa_dispatch.py tests/test_dsa/test_dsa_solver.py` 为 `26 passed`（含 1000 组随机 balanced packed plans）；`pytest -q tests/test_dsa/test_dsa_api.py` 为 `22 passed`；`pytest -q tests/test_attn/test_dsa_v4.py` 为 `17 passed`；Black、isort、compileall 通过。
- 出口证据：1024-token 示例 sequential Indexer cost 为 `[32640, 98176]`（`max/mean-1=50.098%`），balanced 为 `[65408, 65408]`（`0.000%`），输出 fragments 等价于 `[0:256)+[768:1024)` / `[256:768)`；固定 49152-token 单 sample 静态预测从 `[75491328, 226486272]`（`50.002%`）变为 `[150988800, 150988800]`（`0.000%`）。
- 报告：`docs/magi_dsa_v4_design.md`、`tests/test_dsa/README.md`、`tests/test_dsa/README_zh.md`；步骤 2 只生成 host metadata，CP tensor 数据移动仍由步骤 3 接入。

### 步骤 3：实现 GroupCast/GroupReduce 和 reference packing

- 新增：`functional/dsa_comm.py`；先用 torch reference map 验证，不进入性能计时。
- 新增测试：`tests/test_dsa/test_dsa_cp.py` 的 transport-only 用例；此时不运行完整 attention。
- 动作：为 window KV、overlap x、compressed KV、compressed Ki 分别生成 `GroupCollectiveArg`；forward GroupCast 得到 unique receive buffer，backward 用对称 GroupReduce 回 owner。
- 测试：CP=2 transport-only；逐项验证 send/recv rows、重复 fragment FP32 求和、零长度路线、非相邻 owner、A2AV fallback 和 native grpcoll 语义一致。
- 出口：DSA 生产通信代码不直接调用 torch collectives/all2allv；四类 payload 的 forward/reverse 都与 host reference 一致。

完成记录（2026-07-10）：

- worktree：`agents/worktrees/magi-dsa-v4-plan-grpcoll`；步骤 2 图示基线 commit `870e53e5`。
- 实现 commits：`29a9ffa3`（`Add typed DSA group collective transport`）、`f81d121e`（`Align native DSA compressed index transport`）。
- 实现：新增 `functional/dsa_comm.py`，为 window KV、overlap x、compressed KV、compressed Ki 建立四套独立 typed metadata、`GroupCollectiveArg`、per-call buffer slot/work/native handle；token 路线按 transfer table 合并 destination 并 reference-pack unique owner rows，compressed KV/Ki 静态发给全部 peer；空 source/destination 路线保留显式零长度 split；backward 复用 forward route 做对称 GroupReduce，并以 FP32 communication/owner accumulator 合并后一次性转回目标 dtype。
- 生产边界检查：DSA 通信源码只调用 `group_cast` / `group_reduce`，未直接调用 torch P2P、all-gather、all-reduce、all-to-all 或 `all2all_v`；torch `index_select`/`index_copy_` reference seam 留给步骤 4 的 SM90 kernel 替换。
- CP=2 测试：冻结基础镜像未预装可选 `magi_attn_comm` 时为 `6 passed, 1 skipped`；从同一冻结镜像编译 pinned 源码并安装 `nvidia-nvshmem-cu13==3.6.5` 后，最终镜像 `magi-dsa-native-step3:final`（image id `sha256:87ee9e4e72fe1d641ac481b6960819c42aa2004cdd2458b2c6e1173f1fcbb677`）运行 `timeout 60 pytest -q test_dsa_cp.py` 为 `7 passed`。真实双 H100 覆盖 A2AV/native、非连续 send/recv rows、四类 payload forward/reverse、FP32 owner 求和、零长度路线和空 rank。
- 回归：`pytest -q tests/test_dsa/test_dsa_api.py tests/test_dsa/test_dsa_dispatch.py tests/test_dsa/test_dsa_solver.py` 为 `48 passed`；Black、isort、compileall 通过。镜像未安装 flake8，未执行该项。
- native 修正：真实 kernel 首次暴露 compressed Ki 的 BF16 D=128 不满足 grpcoll 256-element transport alignment；仅对 native send/receive buffer 右侧补零到 D=256，接收后裁回逻辑 D=128，反向 FP32 D=128 原生满足对齐。完整 native 测试随后通过；未修改/fork grpcoll kernel。
- 图示：`docs/assets/dsa_step3_comm_before.{dot,svg,png}`、`docs/assets/dsa_step3_comm_after.{dot,svg,png}`；设计说明同步在 `docs/magi_dsa_v4_design.md`。

### 步骤 4：实现 SM90 packing/remap/CSR kernel

- 新增：`kernel/cutedsl/dsa_pack.py` 和正式 kernel 单测。
- 新增测试：`tests/test_dsa/test_dsa_pack_kernel.py`。
- 动作：
  1. 实现 device-resident int32 destination→source row copy map、logical block/token id→local packed row remap LUT，以及 FP32 CSR reduce map；静态 host plan 负责上传前的完整覆盖、CSR 单调性和越界校验，动态 top-k/remap 全程留在 device。
  2. row copy 支持 BF16、FP32 和 int32，flatten trailing dimensions，覆盖 D=128、D=512 和任意 hidden width；满足对齐时走 128-bit vector copy，否则走有界 tail 路径。
  3. CSR reduce 以 FP32 source/destination accumulator 工作，固定 CSR 顺序、无 atomic；支持覆盖非空 destination row或累加到已有 accumulator，空 destination row 保持不变，最终 dtype conversion 留给调用方。
  4. device mapping 是可共享只读的静态 plan state；packed/remote tensor、output buffer、work 和 event 仍为 per-call state。CuTe 编译 cache key 包含 arch、dtype、feature width 和 operation，使同一 frontend 可在后续独立增加 SM100 验证和调优。
  5. `functional/dsa_comm.py` 的 production GroupCast pack、GroupReduce initial row copy 和 owner restore 改用新 kernel；步骤 3 torch reference seam 保留仅用于对拍。
- 测试：与步骤 3 reference 对拍；覆盖重复 source、极端 fan-in、零 row、非连续 map、`-1` sentinel、missing logical id、tail 和静态 plan 越界拒绝；编译后单批 kernel 使用 10 秒 watchdog，完整 kernel 单测使用 30 秒 watchdog。
- 出口：CP transport 使用 SM90 kernel，无 D2H top-k/remap，同输入重复运行稳定。

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
- 测试：CP=1/CP=2、三种 ratio、packed 多 sample、非连续 fragments；比较 top-k、O、FP32 LSE 和 KL。
- 落位：单卡用例写入 `test_dsa_api.py`，双卡用例写入 `test_dsa_cp.py`；把稳定的 `agents/tests` kernel/Megatron parity forward 用例迁入正式目录。
- 出口：sequential 与 balanced forward 都和同一 CP=1 global reference 对齐。

### 步骤 6：接通完整 backward 和 saved-state

- 修改：`functional/dist_dsa.py`、`functional/dsa_comm.py`。
- Kernel 复用：继续使用 `_KernelSparseAttn.backward` 的 cuDNN sparse backward/d_sink，以及 `_KernelIndexerKL` 的 score recompute/Indexer backward；compressor backward 保持 PyTorch autograd。
- 动作顺序：重收 KV/x并重算压缩条；运行 KL backward 并 GroupReduce dKi；运行 sparse backward；FP32 CSR 合并 dKV；GroupReduce 回 owner；执行两个 compressor backward；GroupReduce d_sink 和内部参数梯度。
- 测试：dx、dqr、dQ、dKV、compressor/Indexer 参数梯度、d_sink 与 CP=1 reference 对齐；saved-tensor hooks 确认未保存 compressed/remote/packed tensor；释放 forward 临时 buffer 后 backward 仍通过。
- 落位：CP=1 backward/saved-state 写入 `test_dsa_api.py`；CP=2 全梯度和 owner reduce 写入 `test_dsa_cp.py`。
- 出口：全部梯度正确，参数梯度明确为 CP-reduced，saved-state 满足冻结合同。

### 步骤 7：实现 overlap、并发和故障收敛

- 修改：runtime per-call state 和 stream/event 编排。
- 动作：实现两个独立开关：compressed GroupCast 与 Indexer projection；dKi GroupReduce 与 sparse backward。为每次调用独占 work/event/buffer slot。
- 测试：2×2 开关矩阵、gradient accumulation、两个并发 microbatch、reentrant backward、空 rank、kernel/collective 异常、提前退出；CP watchdog 60 秒。
- 落位：runtime/communication 压力用例写入 `test_dsa_cp.py`；kernel watchdog 留在 `test_dsa_pack_kernel.py`。
- 出口：四种开关数值一致，无共享状态污染，所有故障在 watchdog 内收敛。

### 步骤 8：H100 校准、最终 solver 和性能验收

- 新增产物：`agents/benchmarks/magi-dsa-v4-balance`、`agents/perf/magi-dsa-v4-balance`、`agents/profiles/magi-dsa-v4-balance`。
- 动作：测量 Indexer tiles/top-k/score recompute/backward、packing、GroupCast/GroupReduce 和 overlap，回填 solver 系数；在固定 20 packs 上运行 sequential serial 与 balanced 2×2 overlap。
- 出口：ratio=4 Indexer 比较门槛、ratio=4/128 E2E 门槛和 CP=2 完整路径 5% 门槛全部通过；分项和真实 overlap 时间齐全。

### 步骤 9：文档、原子提交与集成

- 动作：同步 API、config、tensor contract、communication plan、saved-state、测试和性能结果；更新 `docs/source/user_guide/magi_api.md` 和 `tests/test_dsa/README*`；整理可审查原子 commits。
- 迁移：检查主工作区用户改动后 cherry-pick；集成复测；子仓指针单独核对。
- 出口：所有冻结合同有测试或报告证据，才能标记完成。

## 决策记录

- 2026-07-09：只保留 `PLAN.md`，删除 `AGENTS.md`；本文件同时承担冻结合同和动态计划。
- 2026-07-09：保持独立 `MagiDSARuntimeMgr + calc_dsa` 旁路。
- 2026-07-09：ratio=4 用 128 对齐、可非连续 fragments 平衡 Indexer；Indexer 时间包含 forward、top-k、score recompute 和 backward。
- 2026-07-09：Indexer candidate 必须优于 sequential，但不设单项 5% 绝对门槛；完整路径继续执行 5% 门槛。
- 2026-07-09：KL 返回每 rank 的可微 local contribution；d_sink 和内部参数梯度由 DSA 内部 GroupReduce，结果标记为 CP-reduced。
- 2026-07-09：正式 CP 通信使用 GroupCast/GroupReduce；compressed KV/Ki 保持静态全组可见。
- 2026-07-09：步骤 5/6 复用现有 FlashMLA/cuDNN wrapper，不重写外部 kernel；新增 kernel 仅限步骤 4 的 SM90 packing/remap/CSR。
- 2026-07-09：正式 DSA 测试按 Magi-MSA 的五类结构建立在 `tests/test_dsa`；现有测试和 `agents/tests` 稳定用例迁入后才计入验收。
- 2026-07-10：步骤 4 仍只以 SM90/H100 为出口硬件，但 packing/remap/CSR 使用架构中性的 public frontend、静态 device mapping 和 arch-aware 编译缓存；后续 SM100 只新增构建、验证与调优，不改变 mapping/communication API。CP>2/多机继续不作为 V1 验收条件，其正确性扩展复用同一 mapping 与 GroupCast/GroupReduce 接口，性能扩展另行校准 topology-aware solver 和 internode transport。
