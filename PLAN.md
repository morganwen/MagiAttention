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
- 新 packing/remap/CSR kernel 用 `magi_attention/kernel/cutedsl` 风格的 SM90 CuTe DSL；不复制 Magi-MSA 的 SM100 kernel。
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
- FlashMLA：`b7643bd54521f563b839b98289b5cd048c062ba2`；cudnn-frontend 1.27.0；`nvidia-cutlass-dsl==4.5.2`。
- CUTLASS 子仓：`81a43e6d92cdd8c20d22392f9579604ed5f710a1`；FA4 子仓：`ee1d15159cda6f3f97bfab9e487da146a8254970`。
- DeepSeek V4 官方 HF config revision：待冻结。

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

## 当前状态

- 已完成：三形态 PyTorch reference、compressor、Indexer、FlashMLA/cuDNN kernel、packed 原型、连续 CP=2 对拍和初版 profile。
- 证据：`agents/tests/magi-dsa-v4`、`agents/profiles/magi-dsa-v4`、`agents/perf/magi-dsa-v4`。
- 未完成：独立 runtime、非连续 fragment plan、真实 Indexer-balanced solver、正式 GroupCast/GroupReduce、SM90 packing、saved-state 收紧、最终并发/死锁/性能验收。
- 当前 `magi_comm.py` 是未提交技术探针，不直接作为生产实现；重构前先提交或备份，禁止覆盖/reset。
- 当前进行：步骤 0。

## 具体实施步骤

### 步骤 0：清理并冻结设计输入

- 修改：本 `PLAN.md`、`docs/magi_dsa_v4_design.md`。
- 动作：删除旧 P2P/allgather 正式方案描述；冻结 HF revision、子仓状态、H100 镜像和当前有效测试命令；保存现有 dirty grpcoll 探针。
- 出口：所有 revision 可复现，设计文档与本计划一致，主 worktree 用户改动未丢失。

### 步骤 1：建立公共 API 和 runtime 空骨架

- 新增：`api/dsa_attn_interface.py`、`dsa_runtime_mgr.py`、`functional/dist_dsa.py`。
- 动作：定义 `MagiDSAInput`、`DsaPackedMeta`、`MagiDSARuntimeMgr`、`calc_dsa`；先让 CP=1 调用现有 `MagiDSAV4` kernel path，CP=2 暂只创建 plan 不通信。
- 测试：字段 shape/dtype/config 拒绝测试；CP=1 三种 ratio 的 O、KL 和全部梯度与当前 API 一致。
- 出口：公共接口可运行，后续不再直接调用 `forward_cp(_packed)`。

### 步骤 2：实现 fragment plan 和 Indexer solver

- 新增：`meta/collection/dsa_meta.py`、`meta/solver/dsa_dispatch.py`、`meta/solver/dsa_solver.py`。
- 动作：
  1. 定义 `DsaFragmentSpec(sample_id,q_begin,q_end)`、compressed block owner、rank plan 和 restore map。
  2. 实现 128 对齐 sequential plan。
  3. 实现 ratio=4 成本：位置 `p` 的可见块数为 `floor((p+1)/4)`，使用 sample 内位置。
  4. 按成本降序分配 fragment，再用 move/swap/128 对齐 split/merge 降低最慢 rank Indexer 成本；约束 token 数、显存和 fragment 数。
  5. 在最优 Indexer slack 内选择完整预测 E2E 最小的 plan；rank 0 确定性求解并广播。
- 测试：1000 组随机 packed plans；覆盖无重叠/空洞、128 对齐、多 sample、非连续 fragments、空 rank、block owner、稳定 plan hash。
- 出口：示例 `[0:256)+[768:1024)` / `[256:768)` 能均衡 scan cost；sequential/balanced report 可读且重复运行一致。

### 步骤 3：实现 GroupCast/GroupReduce 和 reference packing

- 新增：`functional/dsa_comm.py`；先用 torch reference map 验证，不进入性能计时。
- 动作：为 window KV、overlap x、compressed KV、compressed Ki 分别生成 `GroupCollectiveArg`；forward GroupCast 得到 unique receive buffer，backward 用对称 GroupReduce 回 owner。
- 测试：CP=2 transport-only；逐项验证 send/recv rows、重复 fragment FP32 求和、零长度路线、非相邻 owner、A2AV fallback 和 native grpcoll 语义一致。
- 出口：DSA 生产通信代码不直接调用 torch collectives/all2allv；四类 payload 的 forward/reverse 都与 host reference 一致。

### 步骤 4：实现 SM90 packing/remap/CSR kernel

- 新增：`kernel/cutedsl/dsa_pack.py` 和正式 kernel 单测。
- 动作：实现 int32 destination→source copy、block/token remap、FP32 CSR reduce；支持 D=128、D=512 和 hidden width，128-bit 对齐快路。
- 测试：与步骤 3 reference 对拍；覆盖重复 source、极端 fan-in、零 row、非连续 map、tail 和越界拒绝；应用 10/30 秒 watchdog。
- 出口：CP transport 使用 SM90 kernel，无 D2H top-k/remap，同输入重复运行稳定。

### 步骤 5：接通完整 forward

- 修改：`functional/dist_dsa.py`、`dsa_runtime_mgr.py` 和现有 DSA kernel wrapper。
- 动作顺序：
  1. dispatch 本 rank Q/x/qr/KV fragments。
  2. 启动 window KV、overlap x GroupCast。
  3. 计算本地 compressed KV/Ki 并 GroupCast 到全组。
  4. 计算 Indexer Q/weights，执行 top-k/KL。
  5. packing/remap 后拼接 window 与压缩索引。
  6. 一次 FlashMLA sparse forward，返回 local O 和 KL contribution。
- 测试：CP=1/CP=2、三种 ratio、packed 多 sample、非连续 fragments；比较 top-k、O、FP32 LSE 和 KL。
- 出口：sequential 与 balanced forward 都和同一 CP=1 global reference 对齐。

### 步骤 6：接通完整 backward 和 saved-state

- 修改：`functional/dist_dsa.py`、`functional/dsa_comm.py`。
- 动作顺序：重收 KV/x并重算压缩条；运行 KL backward 并 GroupReduce dKi；运行 sparse backward；FP32 CSR 合并 dKV；GroupReduce 回 owner；执行两个 compressor backward；GroupReduce d_sink 和内部参数梯度。
- 测试：dx、dqr、dQ、dKV、compressor/Indexer 参数梯度、d_sink 与 CP=1 reference 对齐；saved-tensor hooks 确认未保存 compressed/remote/packed tensor；释放 forward 临时 buffer 后 backward 仍通过。
- 出口：全部梯度正确，参数梯度明确为 CP-reduced，saved-state 满足冻结合同。

### 步骤 7：实现 overlap、并发和故障收敛

- 修改：runtime per-call state 和 stream/event 编排。
- 动作：实现两个独立开关：compressed GroupCast 与 Indexer projection；dKi GroupReduce 与 sparse backward。为每次调用独占 work/event/buffer slot。
- 测试：2×2 开关矩阵、gradient accumulation、两个并发 microbatch、reentrant backward、空 rank、kernel/collective 异常、提前退出；CP watchdog 60 秒。
- 出口：四种开关数值一致，无共享状态污染，所有故障在 watchdog 内收敛。

### 步骤 8：H100 校准、最终 solver 和性能验收

- 新增产物：`agents/benchmarks/magi-dsa-v4-balance`、`agents/perf/magi-dsa-v4-balance`、`agents/profiles/magi-dsa-v4-balance`。
- 动作：测量 Indexer tiles/top-k/score recompute/backward、packing、GroupCast/GroupReduce 和 overlap，回填 solver 系数；在固定 20 packs 上运行 sequential serial 与 balanced 2×2 overlap。
- 出口：ratio=4 Indexer 比较门槛、ratio=4/128 E2E 门槛和 CP=2 完整路径 5% 门槛全部通过；分项和真实 overlap 时间齐全。

### 步骤 9：文档、原子提交与集成

- 动作：同步 API、config、tensor contract、communication plan、saved-state、测试和性能结果；整理可审查原子 commits。
- 迁移：检查主工作区用户改动后 cherry-pick；集成复测；子仓指针单独核对。
- 出口：所有冻结合同有测试或报告证据，才能标记完成。

## 决策记录

- 2026-07-09：只保留 `PLAN.md`，删除 `AGENTS.md`；本文件同时承担冻结合同和动态计划。
- 2026-07-09：保持独立 `MagiDSARuntimeMgr + calc_dsa` 旁路。
- 2026-07-09：ratio=4 用 128 对齐、可非连续 fragments 平衡 Indexer；Indexer 时间包含 forward、top-k、score recompute 和 backward。
- 2026-07-09：Indexer candidate 必须优于 sequential，但不设单项 5% 绝对门槛；完整路径继续执行 5% 门槛。
- 2026-07-09：KL 返回每 rank 的可微 local contribution；d_sink 和内部参数梯度由 DSA 内部 GroupReduce，结果标记为 CP-reduced。
- 2026-07-09：正式 CP 通信使用 GroupCast/GroupReduce；compressed KV/Ki 保持静态全组可见。
