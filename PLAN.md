# PLAN.md：Magi_DSA V4 实施计划

## 文件分工

- `AGENTS.md` 是恒定契约；本文件只记录当前设计结论、实施阶段和状态，冲突时以 `AGENTS.md` 为准。
- 阶段完成后回写 commit、测试命令和证据路径。性能结果必须记录硬件、镜像、输入 packs、warmup 和计时区间。
- 开发、验证和提交继续在独立 worktree 完成，验收后再迁移主工作区。

## 当前基线与状态

- MagiAttention 开发基线：`529fb0a4e273b3557a56d8afd60b74da46688095`。
- 当前 DSA 计算原型：worktree `agents/worktrees/magi-dsa-v4`，分支 `magi-dsa-v4`，commit `98e043cafebbdcbce6835e26849d96624136ef70`。
- Magi-MSA 结构参考：`/home/scratch.wewen_gpu/Magi-MSA`，commit `9670ae222b4a93ce63994982c111604f9f0d6265`。
- Megatron dsv4 数值参考：`/home/scratch.wewen_gpu/megatron-lm`，commit `c6449f0b2`。
- FlashMLA：`b7643bd54521f563b839b98289b5cd048c062ba2`；cudnn-frontend 1.27.0；`nvidia-cutlass-dsl==4.5.2`。
- CUTLASS 子仓指针：`81a43e6d92cdd8c20d22392f9579604ed5f710a1`；FA4 子仓指针：`ee1d15159cda6f3f97bfab9e487da146a8254970`；官方 HF config revision 待冻结。
- 当前原型已有三形态 reference、compressor、Indexer、FlashMLA/cuDNN kernel、packed 路径和连续 CP=2 对拍。已有证据在 `agents/tests/magi-dsa-v4`、`agents/profiles/magi-dsa-v4` 和 `agents/perf/magi-dsa-v4`。
- 当前缺口是：CP 仍围绕连续 `cuts` 和 torch collectives；solver 未支持非连续 fragments 和真实 Indexer 成本；saved-state、GroupCast/GroupReduce、最终性能验收尚未完成。
- 现有未提交 `magi_comm.py` 只作为 GroupCast/GroupReduce 技术探针，重构时保留其改动，不直接视为生产实现。

## 已冻结的设计结论

### 独立旁路

- 保持独立 `MagiDSARuntimeMgr + calc_dsa`，不进入原有 `calc_attn`、`DistAttnRuntimeKey` 或 FFA runtime。
- `MagiDSAV4` 持有 compressor、Indexer 等层参数；`MagiDSARuntimeMgr` 持有 packed batch 的静态 dispatch、通信和 packing plan；`calc_dsa` 编排一次 forward/backward。
- 公共输入用结构化 `MagiDSAInput` 和 `DsaPackedMeta`，包含 x、qr、Q、latent KV、sink 与 packed 位置；`calc_dsa` 返回 `(O, kl_loss)`。
- 复用 MagiAttention 的 GroupCast/GroupReduce、range、transfer-table 和测试基础设施，不复用旧 `CommMeta` 的同 head-dim Q/K/V 假设。

目标文件位置：

```text
magi_attention/api/dsa_attn_interface.py
magi_attention/dsa_runtime_mgr.py
magi_attention/functional/dist_dsa.py
magi_attention/functional/dsa_comm.py
magi_attention/meta/collection/dsa_meta.py
magi_attention/meta/solver/dsa_dispatch.py
magi_attention/meta/solver/dsa_solver.py
magi_attention/kernel/cutedsl/dsa_pack.py
```

当前 `experimental/dsa_v4` 先作为已验证计算原型；接口稳定后再决定是否整体提升为正式 `magi_attention/dsa` 包，不在 runtime 重构期间同时做无关搬家。

### KL 与参数梯度

- CP 下每个 rank 返回可微的 `local_kl_sum / global_query_tokens`。各 rank 只反传本地 query 的 KL contribution；日志值对 detached scalar 用 GroupReduce 做 CP 汇总。
- `d_sink` 和 Magi_DSA 内部 replicated 参数梯度由 DSA runtime 使用 GroupReduce 汇总，返回值明确为 CP-reduced。外层不能再沿同一个 CP group 重复归约。
- `dx`、`dqr`、`dQ`、`dKV` 按 token owner 返回；remote 和重复 fragment contribution 在 runtime 内合并。

### Indexer 负载均衡与验收

- 只对 ratio=4 的 CSA 实例做 Indexer-balanced dispatch；ratio=0/128 没有 Indexer，使用各自完整路径成本。
- “Indexer 时间”固定包含 `indexer_forward + top-k + score_recompute + indexer_backward`，不只统计 forward。
- CP=2 的 balanced 方案必须满足；CP=1 只报告 Indexer 时间，不做负载比较：

```text
mean_pack(max_rank_indexer_time)_balanced
<
mean_pack(max_rank_indexer_time)_sequential
```

- 同时报告每个 pack 的 Indexer `max_rank / mean_rank - 1`。不额外规定 Indexer 单项必须低于 5%。
- 最终硬门槛仍针对完整 V4 路径：candidate 的 E2E 必须优于同配置 sequential baseline，且 CP=2 每个 pack 的完整路径 `max/mean-1 <= 5%`。

### 通信原语

- 生产 CP 路径只调用 `group_cast` / `group_reduce`，不直接使用 torch P2P、all-gather、all-reduce 或 `all2all_v` 代替。
- A2AV fallback 和 native grpcoll 由 MagiAttention 现有环境配置选择；DSA 不增加 `comm_backend="torch"/"magi"` 双轨生产配置。
- compressed KV、compressed Ki、window latent KV 和 compressor overlap x 使用独立 typed payload、metadata、buffer slot 和 work handle。
- compressed KV/Ki 静态 GroupCast 到所有 CP peers，等价实现契约中的全组 allgather；不按 top-k 结果动态通信。
- window KV 和 compressor overlap x 从 fragment/sample 位置生成静态 transfer table，只发送 consumer 需要的 unique rows，不假设 fragment 位于相邻 rank。
- backward 使用与 forward 对称的 GroupReduce。重复 window/fragment contribution 先在 FP32 CSR accumulator 合并，最后一次性转回目标 dtype。

## Fragment 与 solver

### Metadata

- `DsaFragmentSpec(sample_id, q_begin, q_end)` 使用 sample-relative 区间；sample 内部切点必须 128 对齐，sample 自然首尾不加 padding。
- 同一 sample 可以在一个 rank 上有多个不连续 fragments；所有 fragments 必须无重叠、无空洞地覆盖原 token。
- Q、x、qr、latent KV 共用同一 dispatch 和 original-position map。
- compressed block 显式记录 sample id、block id、原 token 区间和 owner；owner 是完整块最后一个 token 所在 rank。
- `DsaRankPlan` 保存 fragments、q offsets、compressed blocks、GroupCast/GroupReduce 参数和 device packing/CSR maps。静态 plan 不读取 device top-k，不用 top-k 作为 cache key。

### 成本模型

- ratio=4、位置 `p` 的 query 可见压缩块数是 `floor((p+1)/4)`；成本必须使用 sample 内位置，不能使用 packed global 位置。
- H100 上分别校准 Indexer QK tiles、workspace、top-k chunks、score recompute、indexer backward、sparse、packing 和 GroupCast/GroupReduce 时间，不复制 Magi-MSA 的 SM100 常数。
- 每个 rank 的报告至少拆出 `indexer_us`、`sparse_us`、`compress_us`、`packing_us`、`group_cast_us`、`group_reduce_us`、`overlap_us` 和 `scheduled_total_us`。

### 求解过程

1. 按 sample 生成 128 对齐的候选 fragments，不做 padding。
2. 按 Indexer 成本从高到低排序，分配给当前 Indexer 累计成本最低的 rank，同时约束 token 数、显存和 fragment 数。
3. 用 move、swap、128 对齐 split/merge 降低最慢 rank 的 Indexer 时间。
4. 在最优 Indexer 时间的允许 slack 内，选择完整 `scheduled_total_us` 最小的计划；通信量和 fragment 数作为 tie-break。
5. rank 0 确定性求解并广播静态 plan；重复求解必须产生一致结果。

单序列 CP=2 的直观目标是把早期便宜 query 和后期昂贵 query 配对，例如：

```text
rank 0: [0:256) + [768:1024)
rank 1: [256:768)
```

而不是连续均分 `[0:512)`、`[512:1024)`。

## Runtime 编排

### Forward

1. runtime.dispatch 生成本 rank 的 Q/x/qr/KV fragments 和 sample-relative positions。
2. 异步启动 window KV 与 compressor overlap x 的 GroupCast。
3. 计算本地 compressed KV/Ki，完成后立即 GroupCast 到全组。
4. 通信期间计算本地 Indexer Q/weights projection 和 window metadata。
5. compressed Ki 到齐后执行 Indexer/top-k/KL；数据到齐后 packing/remap 窗口与压缩索引。
6. 一次 FlashMLA sparse forward，sink 进入同一 softmax，返回 local O 和 KL contribution。

### Backward

1. 按静态 plan 重收原始 KV/x 并重算压缩条；不保存或复用 forward remote/packed buffer。
2. KL backward 产生 Indexer 梯度和 packed dKi，立即启动 dKi GroupReduce。
3. sparse backward 产生 dQ、window/compressed dKV 和 d_sink。
4. dKV 先做 FP32 CSR 合并，再 GroupReduce 回 raw token 或 compressed block owner。
5. owner 执行 compressor backward；overlap x 梯度再 GroupReduce 回原 token owner。
6. 等待全部 work 后返回输入梯度和已 CP-reduced 的 sink/内部参数梯度。

### Saved-state 与 overlap

- forward 只保存 O、FP32 LSE、topk_idx、topk_length 和约定的 compressor gate 中间量；不保存 compressed、remote、packed tensor、work 或 event。
- 静态 plan 留在 runtime；work、event、buffer 和 saved-state 属于单次 autograd 调用，不能被并发 microbatch 共享。
- 两个独立 overlap 开关：compressed GroupCast 与本地 Indexer projection；dKi GroupReduce 与 sparse backward。
- 所有 rank 以一致顺序进入 collectives，空路线使用合法零长度 metadata；异常路径必须 drain 已发起 work。

## 实施阶段

### 阶段 0：基线与设计冻结

- 状态：进行中。
- 冻结 HF config revision、子仓指针和 H100 镜像；把本文件结论同步到 `docs/magi_dsa_v4_design.md`。
- 出口：reference revisions 完整，设计文档不再描述 P2P halo/torch allgather 为正式路径。

### 阶段 1：Fragment、metadata 与 reference packing

- 状态：未开始。
- 实现 DSA dataclass、sequential/custom fragment plan、transfer table、copy/CSR map、plan hash 和 cache 隔离。
- 测试 exact coverage、128 对齐、packed 多 sample、非连续 fragments、空 rank、block owner、window 边界、输出恢复和 1000 组随机计划。

### 阶段 2：GroupCast/GroupReduce 与 SM90 packing

- 状态：未开始。
- 接通 A2AV fallback 和 native grpcoll；实现 DSA SM90 copy/remap/CSR kernel。
- CP=2 transport 测试覆盖四类 payload、FP32 reverse reduce、零长度、并发调用和异常 drain。
- 出口：生产 CP 代码不直接调用 torch collectives/all2allv，两种 grpcoll 实现语义一致。

### 阶段 3：Runtime 与 custom autograd

- 状态：未开始。
- 实现 `MagiDSARuntimeMgr`、`calc_dsa`、三形态统一输入、per-call state、KL、d_sink、内部参数归约、saved-state 和两个 overlap 开关。
- sequential plan 下完成 CP=1/CP=2 全量数值、并发和故障注入回归。

### 阶段 4：H100 校准与 Indexer-balanced solver

- 状态：未开始。
- 校准实际 kernel/通信成本，实现确定性 fragment 分配和 refinement。
- 固定 packs 上验证 balanced 的最慢 rank Indexer 时间严格优于 sequential，并记录完整成本分项。

### 阶段 5：最终验收与集成

- 状态：未开始。
- 按 `AGENTS.md` 固定配置运行三形态数值、saved-state、死锁、2×2 overlap 消融和完整性能测试。
- CP=1 全局 24576、CP=2 全局 49152；seed=42、pack_num=20、chunk_ratio=0.25；baseline/candidate 使用相同 packs、kernel、dtype 和 GroupCast/GroupReduce。
- ratio=4/128 执行 E2E 硬门槛，ratio=0 只报告。通过后同步文档、整理原子 commits、迁移主工作区并集成复测。

## 风险与待办

- DeepSeek V4 官方 HF config revision 尚未冻结。
- native grpcoll 与 A2AV fallback 的 split alignment、handle 和 FP32 reduce 约束不同，必须分别验证。
- 非连续 fragments 会增加 packing 和 kernel launch；solver 必须限制 fragment 数，最终选择仍以完整 E2E 为准。
- 当前 dirty grpcoll 探针属于已有工作，重构前先提交或备份，禁止覆盖和 reset。

## 决策记录

- 2026-07-09：保持独立 `MagiDSARuntimeMgr + calc_dsa` 旁路。
- 2026-07-09：ratio=4 用 128 对齐、可非连续 fragments 平衡 Indexer；Indexer 时间包含 forward、top-k、score recompute 和 backward。
- 2026-07-09：Indexer candidate 必须优于 sequential，但不设单项 5% 绝对门槛；完整路径继续执行 5% 门槛。
- 2026-07-09：KL 返回每 rank 的可微 local contribution；d_sink 和内部参数梯度由 DSA 内部 GroupReduce，结果标记为 CP-reduced。
- 2026-07-09：正式 CP 通信使用 GroupCast/GroupReduce；compressed KV/Ki 保持静态全组可见，不做 top-k 驱动通信。
