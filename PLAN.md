# PLAN.md：Magi_DSA V4 的 Indexer 负载均衡与 GroupCast/GroupReduce 改造计划

## 文件分工

- 本文件是动态执行计划，记录当前实现基线、目标设计、阶段状态、下一步动作、验证证据和开放问题。
- `AGENTS.md` 是恒定契约。本文只细化执行方式，不降低或改写 `AGENTS.md` 的张量、数值、saved-state、死锁、性能和验收要求；如有冲突，以 `AGENTS.md` 为准。
- 每完成一个阶段，必须把状态和证据一起回写。证据至少包含 worktree、commit、测试命令和输出路径；性能阶段还要记录硬件、镜像、warmup、计时区间和 pack 集合。

## 本轮用户决策

- DSA 的 CP 候选方案必须对 ratio=4 的 lightning Indexer 做显式负载均衡，不能只按 token 数连续均分。
- DSA 核心 CP 数据交换统一走 MagiAttention 的 `group_cast` / `group_reduce` 接口；正式路径不得直接拼 `torch.distributed.send/recv`、`all_gather`、`all_reduce` 或直接调用 `all2all_v` 来替代它们。
- 架构参考 Magi-MSA commit `9670ae222b4a93ce63994982c111604f9f0d6265` 的独立 runtime、fragment plan、packing map、custom autograd 和模块级 overlap 分层，但不照搬它的 M3-Pro 张量契约、SM100 packing kernel 或完整 causal-prefix All2AllV 通信。
- 保持独立旁路：Magi_DSA 不进入原有 `calc_attn`、`DistAttnRuntimeKey` 或 FFA runtime。现有 `MagiDSAV4` 单卡模块和 kernel wrapper 是计算基线；CP 生产入口重构为独立 runtime manager 加 DSA calc 旁路。

## 基线与当前状态（2026-07-09）

### 冻结基线

- MagiAttention 阅读与开发起点：`529fb0a4e273b3557a56d8afd60b74da46688095`。
- 当前 DSA 已提交实现基线：worktree `agents/worktrees/magi-dsa-v4`，分支 `magi-dsa-v4`，commit `98e043cafebbdcbce6835e26849d96624136ef70`。
- Megatron-LM dsv4 reference：`/home/scratch.wewen_gpu/megatron-lm`，分支 `dsv4-repro`，commit `c6449f0b2`。
- Magi-MSA 设计参考：`/home/scratch.wewen_gpu/Magi-MSA`，分支 `MSA`，commit `9670ae222b4a93ce63994982c111604f9f0d6265`。
- FlashMLA nv_dev：`b7643bd54521f563b839b98289b5cd048c062ba2`。
- cudnn-frontend：1.27.0；`nvidia-cutlass-dsl==4.5.2`。
- CUTLASS 子仓预期指针：`81a43e6d92cdd8c20d22392f9579604ed5f710a1`；FA4 子仓预期指针：`ee1d15159cda6f3f97bfab9e487da146a8254970`。进入新 kernel 阶段前重新执行并记录 `git submodule status`。
- DeepSeek V4 官方 HF config revision：待冻结；这是基线冻结阶段唯一未关闭的 revision 项。

### 已有可保留成果

- `magi_attention/experimental/dsa_v4` 已有三形态 reference、compressor、indexer、FlashMLA sparse forward、cuDNN DSA backward/indexer/KL、packed 变长原型和 CP=2 对拍原型。
- 从 `b40a3ba2` 到 `98e043ca` 共十个已提交原子 commit，覆盖 reference、正式单测、packed、连续 CP、kernel backend、sink、cuDNN Indexer/KL、API adapter、solver/overlap/profile 和最终 cuDNN KL backward；旧计划中“四个原子 commit”的描述作废。
- commit `98e043ca` 已把 KL 的 torch gather backward 替换成 cuDNN `score_recompute + indexer_backward`，这条计算路径继续作为 ratio=4 的 kernel 基线。
- 现有正式测试 `tests/test_attn/test_dsa_v4.py` 与 `tests/test_attn/test_dsa_v4_cp.py`、Megatron adapter 对拍和 H100 profile 是后续重构的回归闸门。证据包括 `agents/tests/magi-dsa-v4/parity.out`、`cp2.out`、`adapter.out`，以及 `agents/profiles/magi-dsa-v4`、`agents/perf/magi-dsa-v4`、`agents/scripts/magi-dsa-v4`；这些结果只证明当前连续切分原型，不代表本轮 runtime、非连续 fragment、GroupCast/GroupReduce 或性能验收已经通过。

### 当前实现缺口

- `cp.py` 仍以 `forward_cp` / `forward_cp_packed` 两个函数和一组全局连续 `cuts` 为中心，没有 `MagiDSARuntimeMgr`、结构化输入 dataclass、显式 rank plan 或每次调用隔离的 call state。
- `solver.py` 只在一条连续 packed token 流上用逐行 proxy cost 放 cut；它不能把同一 sample 拆成多个不连续 logical fragments，也没有按真实 cuDNN Indexer tile、top-k、score recompute、indexer backward 和通信/packing 校准。
- 现有连续切分让 rank 靠后的 query 扫更长压缩前缀；ratio=4 的 Indexer 是主要不平衡来源。只移动一个连续 cut 无法在长 sample 和 packed 多样本上稳定同时平衡 Indexer 与 token 数。
- `cp.py` 的 torch 通信包含邻居 P2P halo、padded all-gather 和 all-reduce；这不符合本轮 GroupCast/GroupReduce 决策。
- `agents/worktrees/magi-dsa-v4` 当前有一个未提交 `magi_comm.py` 技术探针，并改动了 `config.py`、`cp.py` 和设计文档。它证明“全组压缩流”和“左 halo”可映射到 `group_cast/group_reduce`，但仍是同步、连续 shard、手写 host meta 的一对 wrapper，未覆盖非连续 fragments、packing plan、native handle、空路线、调用隔离和两个 overlap window，不作为完成态或可直接合入方案。
- 当前 reference 主要依赖普通 autograd，尚未满足最终 saved-state 契约；CP 测试还在测试体外手工 `all_reduce` sink 和参数梯度。

## 目标架构

### 总体分层

```text
MagiDSAInput + DsaPackedMeta
             │
             ▼
MagiDSARuntimeMgr
  ├─ DsaDispatchPlan / DsaRankPlan
  ├─ DsaGroupCollectivePlan
  ├─ DsaDevicePackingPlan
  └─ per-call DsaCallState
             │
             ▼
calc_dsa / MagiDSACpFunc
  ├─ compressor + compressed-stream GroupCast
  ├─ Indexer + top-k + KL
  ├─ window/raw-KV GroupCast + index packing
  ├─ unified sparse attention
  └─ backward GroupReduce + FP32 CSR merge
             │
             ▼
local O + local/global-normalized KL contribution
```

- 名称 `MagiDSARuntimeMgr` 和 `calc_dsa` 是本轮设计候选；进入公共 API 实现前在设计文档评审中冻结。现有 `MagiDSAV4` 保留为层模块和 CP=1 入口，不把 DSA 逻辑回灌原 `calc_attn`。
- runtime 对象只持有 config、静态 plan、device metadata、独立通信 plan 和可复用的只读 kernel cache。work handle、CUDA event、临时/remote/packed buffer、saved tensor 全部放在单次 `DsaCallState` 或 autograd ctx，禁止并发 microbatch 共享。
- CP=1 继续走当前 kernel chain，但改用同一份结构化输入和 metadata 校验，保证 CP=1/CP=2 不是两套张量契约。

### 公共 tensor schema 候选

- `MagiDSAInput` dataclass：`x`、`qr`、`q`、`latent_kv`、`attn_sink`。所有 token tensor 使用 packed THD；ratio 4 必须校验并消费 `qr`，ratio 0/128 保留同名字段但不把它接入计算图。是否为了兼容当前原型允许这两种形态传 `None`，在 API 评审时冻结。
- `DsaPackedMeta` dataclass：CUDA `cu_seqlens`、host 镜像的 sample lengths、`max_seqlen`、sample-relative position map、original token id。host 镜像只在 runtime 初始化时构造静态 plan，不读取 device top-k。
- `DsaDispatchConfig`：`algorithm in {sequential, indexer_balanced}`、solver calibration、fragment 数上限和确定性开关。
- `DsaModuleOverlapConfig`：至少两个独立开关：
  - `compressed_cast_indexer_projection`：压缩流 GroupCast 与本地 Indexer Q/weights projection、窗口 metadata 构造重叠。
  - `sparse_backward_indexer_reduce`：Indexer/KL 的压缩 Ki 梯度 GroupReduce 与 sparse backward 重叠。
- `MagiDSAOutput` 不额外包装 tensor；正式函数返回 `(output, kl_loss)`，保持已冻结的用户可见语义。CP 下 `kl_loss` 的跨 rank 汇总责任见“开放问题”，评审后写死在接口文档和测试里。

## Fragment、dispatch 与 metadata 设计

### 结构化 metadata

- `DsaFragmentSpec(sample_id, q_begin, q_end)`：sample-relative 的 query 区间；sample 内部切点必须 128 对齐，sample 自然首尾不加 padding。
- `DsaFragment`：补齐 `rank`、`global_begin/end`、`local_begin/end`、sample global begin、sample-relative block begin/end、logical batch id。
- `DsaCompressedBlock`：显式记录 `(sample_id, block_id, token_begin, token_end, owner_rank, owner_local_row)`；owner 是块最后一个 token 的 owner。尾部不足 ratio 的 token 不建块。
- `DsaTransferSegment` / `DsaReceivedSegment`：描述 owner local row、global logical row、consumer rank、receive offset 和长度；窗口 KV、compressor overlap x、压缩 KV、压缩 Ki 使用独立 typed segment 集合。
- `DsaRankPlan` 至少包含：fragments、local token 数、local compressed block 数、Q logical-batch `cu_seqlens`、sample-relative q offsets、窗口/overlap 所需 raw ranges、四类 GroupCast 参数、反向对称 GroupReduce 参数、packing/scatter maps、恢复顺序。
- `DsaDispatchPlan` 包含所有 rank plan、全局 packed metadata、split sizes、restore maps、plan cache key 和版本号。cache key 只用 config、CP group identity、sample lengths、dispatch 算法和静态开关，不含 device top-k 值。
- `DsaDevicePackingPlan` 用 int32 destination-to-source map 与 CSR reduce map，分别覆盖 window KV、compressor overlap、compressed KV、compressed Ki 和输出恢复；top-k metadata 仍是 device int32，不转 host。

### plan 不变量

- 每个 sample 的 fragments 必须无重叠、无空洞地覆盖原 token；内部 fragment 边界 128 对齐；同一 sample 可落在同一 rank 的多个不连续区间。
- Q、x、qr、latent KV 共用完全相同的 dispatch 和 original-position map。
- compressed block id 与原始 token 区间可双向换算；每个完整块恰有一个 owner；ratio=4 重叠需要的左块输入通过静态 overlap transfer 补齐。
- window pack 对每个 query 只引用同 sample 最近 128 个原始 token；跨 sample、负位置和不可见压缩块都写规范行尾 `-1`。
- 所有 rank 即使没有本地 query、没有某类 send split 或 recv split，也构造合法零长度 metadata，并以相同 collective 顺序进入调用。
- plan builder 增加随机属性测试：至少 1000 组 packed sample/fragment assignment 验证覆盖、owner、pack、逆向 CSR reduction 和稳定恢复顺序。

## Indexer 负载均衡设计

### 适用范围

- ratio=4 的 CSA 实例启用 Indexer-balanced solver。
- ratio=0 没有 compressor/Indexer；ratio=128 没有 Indexer。两者仍复用同一 dispatch/communication/runtime 框架，但 solver 目标退化为各自完整路径成本，不伪造 Indexer 成本。
- `SequentialDispatchAlg` 的 128 对齐连续切分继续作为基线。候选允许一个 rank 拿多个不连续 fragments，不做 token padding。

### H100 实测校准特征

- 不复制 Magi-MSA 的 SM103a 常数。先对当前 H100/SM90 的 cuDNN `indexer_forward`、TRT-LLM radix top-k、`score_recompute`、`indexer_backward` 做独立 microbenchmark，冻结 launch、tile 和带宽回归系数。
- ratio=4 的每个 fragment 按 sample-relative位置计算，不使用 packed global 位置。至少记录：
  - Q token 数和 logical batch 数。
  - 每个 Q tile 的最大可见压缩块数 `floor((p+1)/4)`，以及实际 Indexer QK tile 数。
  - Indexer workspace 元素数和最大压缩前缀 pitch。
  - top-k chunk 数与有效选中数 `min(512, floor((p+1)/4))`。
  - KL score recompute 和 indexer backward 的选中 entry 数。
  - sparse forward/backward 的有效 entry 数：window 常数项加压缩 top-k。
  - compressor、packing/CSR、GroupCast/GroupReduce 字节数、固定 latency 和可被两个 overlap window 隐藏的时间。
- 成本报告拆成 `indexer_us`、`topk_us`、`kl_us`、`sparse_fwd_us`、`sparse_bwd_us`、`compress_us`、`packing_us`、`group_cast_us`、`group_reduce_us`、`serial_total_us`、`overlap_us`、`scheduled_total_us`。

### 求解流程

1. 按 sample 生成 128 对齐的候选 fragments；优先保留较大连续片段，只有为了 Indexer 平衡才继续 split，避免 fragment/packing 启动开销失控。
2. 用 Indexer variable cost 做 deterministic LPT 初始分配；同成本按 `sample_id, q_begin, rank` 稳定 tie-break。
3. 通过 move、swap、aligned split/merge 搜索降低最慢 rank 的 Indexer 时间；保持精确覆盖、块 owner 和每 rank token/显存上界。
4. 取得当前最优 `max_indexer_us` 后，只保留在 `indexer_slack` 内的候选，再以 `max(scheduled_total_us)` 为主目标，以总通信字节、packing 字节、fragment 数为依次 tie-break。这样既落实 Indexer 平衡，又不违背 `AGENTS.md` 对完整 V4 最慢 rank 的最终目标。
5. rank 0 求解并通过静态对象广播 plan；其他 rank 只校验 plan hash 和本地派生 device metadata。禁止把 top-k tensor 或 device `.item()` 放进 solver/cache。
6. 对同一输入和 config 重复求解必须产生字节一致的 plan；提供 sequential 与 balanced 的可读 report，列出每 rank fragments、Indexer 预测、完整路径预测和通信量。

### 负载验收口径

- 不新增未经用户批准的“Indexer 单项硬门槛”。默认把 Indexer 的 max/mean、最慢/最快和占完整路径比例作为强制报告项；最终硬门槛仍是 `AGENTS.md` 的 CP=2 实测 `max_rank_time / mean_rank_time - 1 <= 5%`，且候选 E2E 必须优于同通信原语的 sequential 严格串行 baseline。
- 如果用户要求 Indexer 本身也设独立阈值，在设计评审阶段补充具体百分比并写入 config validation、solver test 和性能验收表，之后再实现。

## GroupCast/GroupReduce 通信设计

### 原则

- DSA 只调用公开 `magi_attention.comm.primitive.grpcoll.group_cast/group_reduce`。底层是 A2AV fallback 还是 native grpcoll 由 MagiAttention 现有环境配置和 `GroupCollectiveArg` 派生类决定，DSA 不增加 `comm_backend="torch"/"magi"` 双轨生产配置。
- DSA 不直接复用现有 `CommMeta`，因为它假设同一 head dim 的 Q/K/V；改为 DSA 专用 dataclass 持有多个独立 `GroupCollectiveArg`。构造和 sanity check 复用 `GroupCastRanges`、transfer-table 扫描线、`A2AVBasedGroupCollectiveArg` / `NativeGroupCollectiveArg` 的现有风格。
- latent KV/压缩 KV 的 row width 512、indexer Ki 的 row width 128、hidden x 的 row width `hidden_size`，分别使用独立 typed payload、collective arg、buffer name/slot 和 work handle，禁止塞进一个 `GrpCollBuffer`。
- forward 的 owner→consumer fan-out 用 GroupCast；backward 严格用同一拓扑的 GroupReduce 回 owner。remote contribution 先在 FP32 CSR accumulator 合并，GroupReduce 也以 FP32 input/output 执行，最后一次性 cast 回输入 dtype。

### 各数据流

| 数据流 | Forward GroupCast | 本地 packing | Backward GroupReduce |
|---|---|---|---|
| window latent KV | owner 只发静态窗口所需的 unique raw rows | 按 fragment 展开成统一 sparse index 所需的 raw 区 | packed dKV 先 CSR 合并到 unique rows，再回物理 owner |
| ratio=4 compressor overlap x | 发完整块计算缺少的左侧输入；非连续 fragment 不假设邻 rank | 重建本地完整压缩块输入 | compressor dx 的 remote overlap contribution 回原 token owner |
| compressed KV | 每 rank 把 owned 完整压缩块 GroupCast 到所有 CP peers，等价实现契约要求的全组 allgather；本地块零拷贝拼入 | 按 sample/block id 排成全局压缩网格 | sparse dcompressed_KV 对称 GroupReduce 回 block owner，再过 compressor backward |
| compressed indexer Ki | 与 compressed KV 相同拓扑，但独立 D=128 payload | 按同一 block grid 供 Indexer | KL/indexer dKi 对称 GroupReduce 回 block owner，再过 indexer compressor backward |
| replicated sink grad | forward 不通信 | 每 rank 产生 local FP32 d_sink | 用 GroupReduce 的 replicated-destination 拓扑得到全局 d_sink，或在设计评审中明确交给外部 CP 参数 reducer；不能静默保留测试体外 `all_reduce` |

- compressed stream 的 GroupCast 是静态全组可见，不按 device top-k 动态选择目的 rank；这不引入 `AGENTS.md` 明确排除的“只通信选中压缩条”优化。
- window/overlap 使用静态 fragment/sample 位置推导 transfer table。不能保留当前“只从左邻 rank 收固定 h 行”的假设；同一 rank 的不连续 fragments 可能从多个 owner 拉不同区间。
- GroupCast 输出先保持 unique received layout，再用 DSA SM90 packing kernel展开成每个 logical fragment 的 kernel layout；反向先在 FP32 CSR 中消除同一 token 被多个 fragment/window 引用的重复，再 GroupReduce。

### work、stream 与异常

- 每个 collective 都以 `async_op=True` 发起并返回 `WorkWithPostProcessFn`；producer stream 记录 event，consumer stream 在读取 buffer 前 wait，buffer 在 work 完成前 `record_stream` 保活。
- native grpcoll 的 symmetric handle 通过对应 `GroupCollectiveArg.to_group_cast_args()` / `to_group_reduce_args()` 共享，不手工重建；A2AV fallback 与 native 跑同一语义测试。
- 并发 microbatch 的 buffer name/slot 必须带 runtime instance id 和 call sequence，或由 per-call allocator 独占，不能复用全局固定槽位。
- 正常路径、可恢复异常和提前退出都 drain 已发起 work。不可恢复 collective 故障按 `AGENTS.md` 有界销毁 process group，交给 60 秒外部 watchdog；不伪造 cancel。
- 所有 rank 以固定顺序调用：window KV cast → compressor-overlap x cast → compressed KV cast → compressed Ki cast；某形态不存在的流使用一致的形态静态分支，某 rank 的空路线仍传零长度合法参数。

## Packing 与 SM90 kernel

- 参考 Magi-MSA 的 destination→source copy map 和 CSR reduce map 数据模型，以及其 128-bit vectorized copy/FP32 register reduction 思路；不复制 `magi_attention/kernel/cutedsl/msa_pack.py` 的 SM100 实现。
- 新增 DSA 专用 SM90 CuTe DSL packing/remap kernel，代码风格跟 `magi_attention/kernel/cutedsl`：
  - int32 token copy map，支持 BF16/FP32/int32 和 D=128、D=512、可配置 hidden width 的 128-bit 对齐快路。
  - CSR FP32 reduce，输出可选 FP32 或末端 BF16 cast；窗口与压缩条重复贡献先 FP32 合并。
  - block id↔token range、global logical id↔owner local id 的 device remap；top-k 行尾 `-1` 保持规范。
- 先保留纯 torch/range reference 仅供数值对拍；正式 H100 和性能路径必须走 SM90 kernel。
- kernel 单次执行 10 秒死锁阈值、编译后单测 30 秒 watchdog；测试包括零 row、重复 source、极端 CSR fan-in、非连续 map、misaligned tail 和越界拒绝。

## Forward 与 backward 编排

### Forward

1. runtime.dispatch 用同一静态 map 得到本 rank 的 x/qr/q/KV fragments 和 sample-relative positions。
2. 异步发起 window KV 与 compressor-overlap x 的 GroupCast。
3. 计算本地完整块的主 compressed KV 和 indexer compressed Ki；每个流完成后立即 GroupCast 到全组。
4. GroupCast 窗口内计算本地 Indexer Q、weights projection、Hadamard/RoPE 和 window index metadata。
5. 等 compressed Ki 后运行 cuDNN Indexer/top-k；等 compressed KV 和 raw KV 后做 packing/remap，把窗口与压缩 top-k 拼成统一 int32 索引。
6. 一次 FlashMLA sparse forward，sink 进入同一 softmax；返回 local O、FP32 LSE、topk_idx、topk_length 和 local/global-normalized KL contribution。

### Backward

1. 按静态 plan 重新 GroupCast 原始 KV/x 和压缩输入，重算压缩条；不读取 forward remote/packed buffer，不重算 top-k 或 sparse forward。
2. KL backward 消费 `d_kl`，用 cuDNN score recompute/indexer backward 产生 local dQidx/dWeights 和 packed dKi；立即启动 dKi GroupReduce。
3. 在独立 stream 跑 sparse backward，得到 dQ、window packed dKV、compressed packed dKV 和 d_sink。
4. window dKV 与 compressed dKV 分别做 FP32 CSR merge；异步 GroupReduce 回 raw token owner和 compressed block owner。
5. compressed KV/dKi owner 完成两个 compressor backward；其中 overlap x contribution 再通过 GroupReduce 回 raw x owner。合并主 compressor、indexer 参数和本地 query projection 的梯度。
6. 全部 work/event wait 完成后返回 local dx、dqr、dq、dkv、全局口径 d_sink 和 runtime 参数梯度；末端才从 FP32 转 BF16。

### Saved state

- 自定义 `MagiDSACpFunc` 只保存契约允许的 O、FP32 LSE、topk_idx、topk_length 和 compressor gate 中间量；原始 autograd 输入按 PyTorch backward 需要保留，静态 plan 以非 tensor ctx 引用保存。
- 不保存 compressed KV/Ki、remote unique buffer、packed KV、communication work、event 或临时 remap tensor。
- saved-tensor hooks 同时检查 shape/dtype/data_ptr；forward 后主动释放和覆写 remote/packed buffer，再跑 backward 与 CP=1 reference 对拍。

## 实施阶段

### 阶段 0：现场与基线冻结

- 状态：进行中。
- 已完成：冻结 Magi、DSA prototype、Megatron、Magi-MSA、FlashMLA、cudnn-fe commit/version；审阅当前未提交 grpcoll 探针并把缺口写入本文。
- 待完成：冻结官方 HF config revision；在新 worktree 核对子仓指针和 H100 镜像；把已有效测试命令与输出重新归档。
- 出口：所有 reference revision 可复现，本文“当前状态”不再含待填 revision。

### 阶段 1：API、runtime 与通信设计评审

- 状态：当前下一步。
- 冻结 `MagiDSAInput`、`DsaPackedMeta`、runtime manager/calc 名称、KL CP 汇总语义、sink/replicated parameter gradient 的 reducer 责任、Indexer 是否增加单项硬阈值。
- 把本文目标架构同步到 `docs/magi_dsa_v4_design.md`，删除旧的“邻居 P2P halo + torch allgather”正式描述；未评审前不改公共 API。
- 出口：用户评审通过，开放接口/数值问题清零。

### 阶段 2：fragment plan、metadata 与 reference packing

- 状态：未开始。
- 实现 DSA dataclass、sequential plan、任意不连续 fragment plan、transfer table、copy/CSR map 和 plan hash/cache 隔离。
- 用纯 host/reference 测试覆盖 exact coverage、128 对齐、空 rank、多 sample、跨 rank sample、块 owner、window 不越界、ratio 4 overlap、恢复顺序和 1000 组随机计划。
- 出口：静态 plan 不变量全绿，尚不启动通信 kernel。

### 阶段 3：GroupCast/GroupReduce 与 SM90 packing

- 状态：未开始。
- 先接 A2AV-based `GroupCollectiveArg` correctness path，再接 native grpcoll；四类 payload 分槽并实现 async work 生命周期。
- 实现 DSA SM90 copy/remap/CSR kernel；reference 与 kernel 双向对拍，正式测试放 `tests`，探索 benchmark 放 `agents/benchmarks/magi-dsa-v4-grpcoll`。
- CP=2 transport-only 测试覆盖 forward fan-out、backward FP32 reduce、空路线、非连续 fragments、并发 call、异常 drain。
- 出口：生产 CP 源码不再直接调用 torch P2P/allgather/allreduce/all2allv；两种 grpcoll backend 语义一致。

### 阶段 4：独立 runtime 与 custom autograd

- 状态：未开始。
- 将当前 `forward_cp(_packed)` 重构到 runtime manager + `calc_dsa` 旁路；三个 ratio 共用 input/dispatch/restore 契约，形态差异只在静态 plan 和 kernel chain。
- 实现 per-call state、forward/backward 编排、KL scalar、d_sink、runtime parameter grads、saved-state 收紧和两个独立 overlap 开关。
- 出口：CP=1 与 CP=2 在 sequential plan 上全量数值对拍通过，saved-state、两并发 microbatch、gradient accumulation、reentrant backward 和故障注入通过。

### 阶段 5：H100 Indexer 成本校准与 balanced solver

- 状态：未开始。
- 在 1×H100 校准 ratio=4 Indexer/top-k/KL/sparse/packing 阶段模型，在 2×H100 校准 GroupCast/GroupReduce latency/bandwidth 和 overlap 可隐藏时间；原始数据放 `agents/perf/magi-dsa-v4-indexer-balance`。
- 实现 deterministic fragment generation、LPT 初分配、move/swap/split/merge refinement 和 plan report/broadcast。
- solver 单测验证 sample-relative 公式、ratio 分形态、稳定性、cache 隔离、Index-only 最优保持和 overlap-aware critical path。
- 出口：固定 packs 上预测与实测排序一致，balanced plan 的 Indexer max/mean 显著优于 sequential，且不牺牲完整路径硬门槛。

### 阶段 6：完整数值、死锁和性能验收

- 状态：未开始。
- 按 `AGENTS.md` 跑 CP=1 全局 24576、CP=2 全局 49152、DatasetSampler seed=42、pack_num=20、chunk_ratio=0.25；baseline 和 candidate 共用同一 packs、kernel、dtype、GroupCast/GroupReduce 和计时协议。
- sequential baseline 模块级严格串行；candidate 为 Indexer-balanced plan 加两个模块 overlap。跑 2×2 overlap 消融，并报告最慢 rank 的实际 overlap 时长。
- 数值覆盖三形态 output/LSE/KL/top-k/全部输入与参数梯度；无 tie 精确、tie 稳定；CP 容差不放宽。
- 性能硬门槛：CP=1、CP=2 candidate 的 `mean_pack(max_rank_e2e)` 都小于各自 baseline；CP=2 每 pack `max/mean-1 <= 5%`。
- 出口：数值、saved-state、并发/故障、死锁 watchdog、端到端和负载门槛全部通过，报告归档。

### 阶段 7：文档、原子 commit 与集成

- 状态：未开始。
- 同步 API、config、metadata schema、communication plan、saved-state、测试、性能、限制和精确 revisions；不把实验探针描述成正式路径。
- worktree 整理为可审查原子 commits；确认主目录用户改动后再 cherry-pick，主目录/目标集成分支复测，子仓指针单独验收。
- 出口：`AGENTS.md` 完成条件逐条有证据，才标记 V4 attention training support 完成。

## 回归与验收矩阵

- Solver/metadata：sequential、balanced、custom noncontiguous；ratio 0/4/128；单/多 sample；空 rank；随机计划；cache/group 隔离。
- Communication：A2AV grpcoll fallback、native grpcoll；window、overlap x、compressed KV、compressed Ki；零长度；FP32 reverse reduce；异常 drain。
- Numerics：PyTorch FP32/BF16 reference、Megatron `c6449f0b2`、kernel；output、LSE、KL、top-k、dQ、dKV、dx、dqr、compressor/indexer params、d_sink。
- Runtime：CP=1/2、saved hooks、释放 remote 后 backward、gradient accumulation、两个并发 microbatch、reentrant autograd、四种 overlap 组合。
- Performance：ratio 4/128 设门槛，ratio 0 只报告；sequential serial 对 balanced/overlap；最慢 rank、Indexer 单项、通信/packing、真实 overlap 全报告。

## 明确不做

- 不把 DSA 塞进 `calc_attn`/FFA runtime，不复用旧 `CommMeta` 的同 head-dim 假设。
- 不把 Magi-MSA 的 SM100 kernel 或固定 M3-Pro shape 拷入 H100 路径。
- 不按 top-k 结果动态通信选中压缩条；compressed KV/Ki 保持静态全组 GroupCast。
- 不做 dense warm-up、attention recompute、FP8/FP4、TP、decode cache、CUDA graph、同调用混合 ratio、SM100 专项、CP>2 或多机验收。
- 不改 cudnn-frontend、FlashMLA 源码；需要 patch 时先请用户确认。

## 开放问题与风险

- 待用户评审：CP 下返回的 `kl_loss` 是每 rank 的 `local_sum/global_tokens` contribution，还是 runtime 内构造成每 rank 都相同的 global scalar；两者的 `d_kl` 和重复 backward 语义不同，不能实现时自行选择。建议保留可微的 local contribution，每 rank 只反传本地 query 的 KL，日志值再做 detached CP reduction；这样不会因每 rank 对同一个 global scalar 反传而重复放大梯度。
- 待用户评审：d_sink 和 runtime-owned replicated parameter gradients 由 DSA 内部 GroupReduce 成全局梯度，还是由外部统一 CP parameter reducer 负责。建议 DSA 内部完成并把这些梯度明确标为 CP-reduced，使 CP=1/CP=2 API 直接一致；外层不得再次沿 CP group reduce。正式测试不能继续靠测试体外临时 `all_reduce` 掩盖接口责任。
- 待用户决定：是否给 Indexer 单项 max/mean 设置独立硬阈值。建议不新增独立验收阈值，把它作为 solver 第一阶段优化目标和强制报告项，最终仍以完整路径 5% 硬门槛约束；如果需要单项门槛，请在实现前给出具体百分比。
- native grpcoll 当前对 split alignment、buffer/handle 复用和 A2AV fallback 的 `acc_reduce/comm_dtype` 有不同约束；阶段 3 必须分别测试，不能把 fallback 通过等同于 native 通过。
- arbitrary fragments 会增加 logical batch、packing 和 launch 数；solver 必须对 fragment 数和显存设上界，避免为了 Indexer proxy 最优反而让 E2E 退化。
- ratio=4 的 compressor overlap 和 128 对齐 fragment 组合要按 Megatron 逐块语义复核；不能从 MSA 的 16-token block 规则类推。
- 当前 dirty grpcoll 探针属于用户/现有工作成果，重构时先形成独立 commit 或备份，不覆盖、不 reset。

## 决策记录

- 2026-07-09：用户要求在当前 DSA 计划上增加 Indexer 负载均衡，并把正式 CP 通信原语改为 GroupCast/GroupReduce；指定 Magi-MSA 独立 runtime 实现为结构参考。
- 2026-07-09：设计结论是复用 Magi-MSA 的 fragment/plan/runtime/autograd 分层，不复用其 causal-prefix All2AllV。DSA raw 流只通信窗口与 compressor overlap；compressed KV/Ki 通过 GroupCast 实现静态全组可见，反向通过对称 GroupReduce 回 owner。
- 2026-07-09：当前 `magi_comm.py` 只定性为 transport spike；生产方案使用 DSA typed metadata 和现有 `GroupCollectiveArg`，不保留 `comm_backend="torch"` 作为正式双轨。
- 2026-07-09：ratio=4 先优化 Indexer max cost，再在 slack 内最小化完整 scheduled E2E；ratio 0/128 不伪造 Indexer 项。最终性能门槛仍以 `AGENTS.md` 的完整路径为准。
