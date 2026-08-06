# Magi-DSA v4 High-Level Design

> **历史文档（已被架构层面取代）：** 本文保留 Q13–Q16、correctness/profile 和旧实现证据。
> 2026-07-21 起，CSA `ratio=4` 的当前权威架构是
> [`magi_dsa_v4_design.md`](./magi_dsa_v4_design.md)；若两者冲突，以新文档为准，禁止据本文恢复
> Query worker、`INDEXER_QW`、Top-K restore 或 owner-stable layout。
> 2026-07-22 曾取消 CSA dual-LSE/四输出 backport并采用 full-domain Indexer KL；该裁决已于
> 2026-07-23 被用户明确推翻。当前实现以冻结 924 为基线应用已审核 dual-LSE 回移：
> 完整 `sparse_lse` 只服务 sparse backward，compressed-prefix LSE 服务 teacher，predictor
> 只在 selected Top-K 内归一化并调用 sparse Indexer backward，未选 candidate 梯度为 0。
> 本文内相反记录只保留为历史证据，不能覆盖当前数学合同。

历史状态（截至 2026-07-20）：**阶段 A–D correctness 与阶段 E profile 已通过；Q16 复验中。**
Q13 natural-only、Q14/Q15 历史裁决、Q16 backend-native Top-K、128K workload 和 5% 均衡门槛仍可
作为证据复用；本文后续出现的“冻结设计”仅描述当时合同，不再约束当前 CSA 架构实现。

### Stage A / Preflight 验收记录（PASSED，2026-07-19）

正式镜像 `magi-dsa-v4:preflight-68c2f15`（image ID
`sha256:41aa27add6026422c3154ead255596f717d29220cb6b346483b436fa7f7c324c`）在目标单机完成阶段 A：

- 8 张 `NVIDIA B300 SXM6 AC` 均为 SM103，driver `595.58.03`，CUDA `13.2`，NCCL `2.29.7`；
- cuDNN frontend `1.26.0` / backend runtime `92400` 的 Indexer score、top-k=512 exact 以及 sparse
  attention backward smoke 通过；
- FlashMLA `1.0.0+9241ae3` sparse prefill 与 PyTorch reference 对齐；
- 8-rank NCCL all-reduce/all-gather 在 60 秒硬时限内通过；Nsight Systems `2026.2.1.210` 可用；
- 固定镜像的 FlashMLA wheel SHA-256 为
  `51c390e1520b661077f9ec3522fcb0e9c1b535fc7bcfd08fbc0184762037ac5a`。

完整命令、原始日志、revision、硬件/依赖、镜像身份和 manifest 保存在
`artifacts/release/20260719T181733Z-preflight/`。Stage A 通过后才开始 runtime/kernel/test 实现。

### Q11 / Preflight backend ABI 阻塞（RESOLVED，2026-07-19）

初始合同中的 `cuDNN frontend v1.23.0` 只有 `cudnn.NSA`，不含 Q5–Q8 所需的
`cudnn.deepseek_sparse_attention`、Indexer score/top-k 和 DSA sparse backward。用户于
2026-07-19 选择方案 1，批准精确升级 frontend 与匹配 backend。冻结组合现改为：

- `NVIDIA/cudnn-frontend@v1.26.0`，tag commit
  `35fd7b0d0e1d4952b904c79341c5e84e3af0a328`，wheel `nvidia-cudnn-frontend==1.26.0`；
- cuDNN backend `9.24.0`，CUDA 13 wheel build `nvidia-cudnn-cu13==9.24.0.43`。

官方 v1.26.0 release notes 明确把该 frontend 列为 cuDNN 9.24.0 及后续版本的推荐版本，并列出
`q_causal_offsets`、SM100F、DSA backward、Indexer alignment/top-k 修复。候选镜像在 B300 SM103
上确认运行时 backend 为 `92400`，Indexer score（含非零 `q_causal_offsets`）与 PyTorch reference
对齐、top-k=512 IDs/values exact，且 sparse attention backward 产生有限非零梯度。Q11 因而关闭；
旧 v1.23 调查证据保留在 `artifacts/release/20260719T173619Z-preflight-blocked/`。

### Q12 / CuTe DSL 依赖升级（RESOLVED，2026-07-19）

Q11 的精确 frontend 带来一个必须单独评审的 Q2 依赖变更：

- v1.26.0 的官方 `pyproject.toml` 将 `nvidia-cutlass-dsl[cu13]` 精确锁定为 `4.5.0`；
- 保持旧 Q2 的 `4.4.2` 时，导入 DSA backward 立即失败：`cutlass.cute.nvgpu` 不提供
  `OperandMajorMode`，因此该组合不能提供冻结 ABI；
- 使用 `nvidia-cutlass-dsl==4.5.0`、对应 `libs-base/libs-cu13==4.5.0`、
  `apache-tvm-ffi==0.1.8.post0` 和 `quack-kernels==0.4.1` 时，同一 B300 smoke 全部通过。

用户于 2026-07-19 独立批准把 Q2 的 CuTe DSL 从 4.4.2 升级到 4.5.0，包含精确的
`nvidia-cutlass-dsl-libs-base==4.5.0` 与 `nvidia-cutlass-dsl-libs-cu13==4.5.0`；TVM-FFI 与 Quack
保持不变。不得用 `--no-deps` 强行组合 frontend 1.26.0 与 CuTe 4.4.2。Q12 由此关闭；上述正式
镜像完整 Preflight 已通过。

### Q13 / Natural-only release gate（APPROVED，2026-07-19）

用户按 Magi-MSA 的实际验收方式批准取消完整 forced top-k release gate，并冻结以下修订：

- 正式 `MagiDSAInput` 不包含 `forced_topk_ids` 或 `forced_topk_length`；production 始终使用 natural
  Indexer selection；
- CP1、CP2、CP8 correctness 均验证 natural forward/backward。sequential、balanced 与 CP1
  pure-PyTorch reference 按当前 Q16 合同验证 natural Top-K length/结构，并保持 Indexer raw
  score/LSE、output、KL、全部输入梯度及 Compressor/Indexer 参数梯度门槛；
- natural top-k 的历史 Q14/Q15 comparator/tie 语义已由 Q16 backend-native 合同取代；小型 Indexer
  单测覆盖 cuDNN 原始输出保留、global offset、长度/padding/唯一性及 cutoff 合法性，不再用完整 forced
  backward 改写 selection；
- 与 MSA 一样，plan setup、JIT、kernel 编译和完整 forward/backward prewarm 位于 60 秒 watchdog
  之外。CP2 gate 是预热后的 natural sequential/balanced 完整前后向在 60 秒内完成并与 reference
  对齐；
- 旧 forced backward timeout 与对齐诊断 artifacts 作为历史记录保留，但不再构成当前 blocker 或
  release gate。
- Q13 批准时的下一关是 CP2 natural sequential/balanced 完整前后向；该关及后续 Q14/Q15 验证均已
  留下历史证据，当前以 Q16 复验结果为准。

### Q14 / 独立数值实现之间的 Top-K 位序合同（APPROVED，2026-07-20）

本节保留 Q14 当时的事实、诊断与历史裁决；其中 Top-K ID 集合/位序 comparator 已由后续 Q16 取代，
Indexer raw-score tolerance 和其余数值门槛仍有效。

CP8 natural backward 的 production sequential 与 balanced 结果逐元素一致，但二者与 pure-PyTorch
CP1 reference 在部分 Query 的 Top-K **位序**上不同。最小 forward-only 诊断保存在
`artifacts/correctness/20260720T023158Z-cp8-cp8-topk-diagnostic/`：

- 8 个 rank 的 post-compile 执行均在 `0.019s` 内结束，无 timeout、未闭合 phase 或 collective stall；
- 所有差异行的 `length` 与选中 ID 集合完全一致，最终 production ID 顺序也都与固定 cuDNN backend
  返回的 FP32 score 降序完全一致；
- pure-PyTorch 与 cuDNN score 的逐 rank 最大绝对差为 `1.28e-3` 至 `3.46e-3`。局部相邻 score 间隔
  小于该数值误差时，两种合法 FP32 归约顺序产生相反位序；没有候选漏选、越界 ID、route/restore
  错位或 causal-length 差异。

事实来源对照结果：

- Magi-MSA 的 exact-ID 分布式测试用同一个 MSA backend 分别计算 global reference 和 distributed
  candidate，因此验证的是 dispatch/restore bitwise exact，不覆盖独立 score 实现的归约位序差异；
- 官方 DeepSeek-V4 固定 reference revision 直接对 `index_score` 调用 `torch.topk`，没有定义不同
  kernel 归约树之间的 bitwise 位序 ABI；
- 固定 `cudnn-frontend@35fd7b0d...` 的官方 DSA reference test 明确写明 Top-K order 可以不同，并按
  每行选中集合及排序后的 score values 验证。其 Indexer score reference 同样使用 pure-PyTorch FP32
  计算并以 tolerance 对齐，不要求 bitwise score。

用户批准方案 1，并补充冻结为以下可执行合同：

- 在相同 backend、输入和 seed 下，production sequential 与 balanced 的 ordered Top-K tensor 和
  `topk_length` 继续逐元素 exact；这验证 dispatch/restore 没有置换、丢失或重复。
- production 与独立 pure-PyTorch reference 比较 canonical global IDs。逐行先用各自的
  `topk_length` 过滤 padding；effective length 必须 exact；两边的有效 global ID 必须分别唯一；随后按
  global ID 升序排列的有效 tensor 必须 exact。不同 reduction backend 的 Top-K 内部 score-order 位序
  不要求 exact，也不把 padding sentinel 纳入集合比较。
- Indexer raw score 继续按 Q9 预先冻结的 forward `atol=rtol=5e-3` 与 pure-PyTorch FP32 reference
  对齐。Indexer LSE、output、KL、全部输入与 Compressor/Indexer 参数梯度、cutoff tie policy 和 natural
  backward 门槛均保持不变。
- cutoff tie 仍由小型 Indexer 单测验证；其 deterministic secondary key 后续由 Q15 明确为 canonical
  global ID 升序。Q14 不把 near-equal-but-non-tied score 重新定义成 tie。
- 本修订只改变跨 reduction backend 的验收 comparator。Q15 之外不得修改 production 数学、backend
  ABI 或非 tie score-order 行为。

Q14 因此关闭；上述历史诊断保留为审计证据，CP8 natural backward 按新 comparator 重跑后才可进入
128K profile。

### Q15 / 128K grouped Top-K exact-tie（APPROVED，2026-07-20）

本节保留 Q15 的历史诊断和曾批准实现；其 global-ID secondary key、全候选重选与 deterministic 位序
要求已由后续 Q16 明确取消，不能再作为当前 production 或验收合同。

CP1 与 CP8 natural backward 按 Q14 通过后，正式 128K sequential capture 完成了 8 ranks × 5 steps，
但 capture 外的同输入 shadow 对拍在 8/8 ranks 违反 Q14 的 ordered Top-K exact。三次有实质差异的
尝试均已停止并封存：

1. `artifacts/profile/20260720T033502Z-dsv4-flash-128k/` 生成了可打开的 sequential aggregate
   `.nsys-rep`；其中 40 个 step、80 个 score/top-k logical ranges 和 8 worker processes 均通过 SQLite
   审计。失败只发生在 capture 停止后的 balanced shadow Top-K 对拍。
2. `artifacts/correctness/20260720T034125Z-cp8-128k-topk-diagnostic/` 证明全部 effective length exact；
   735 行 ordered mismatch 中 733 行仅为同集合位序交换，2 行发生 cutoff 同分候选替换。
3. `artifacts/correctness/20260720T034616Z-cp8-128k-raw-score-diagnostic/` 抓取 12 个代表行在两种 worker
   layout 下的 raw score。每行两种 plan 的 score SHA-256、finite mask 和每个 FP32 元素均 exact，
   `max_abs=0`；但 Top-K 仍有 736 行 ordered mismatch 和 4 行 cutoff 集合替换。两次诊断的 cutoff
   替换行数不同，也证明固定输入下的 radix Top-K tied-candidate 选择受 grouped execution layout/调度
   影响，而不是 score reduction 误差、route 错位或输入变化。

代表性直接证据：global row 31147 的 ID 5242 与 5244 raw score 都是 `0.9311350584030151`；global
row 41200 的 ID 3242 与 9869 都是 `0.5746891498565674`。同分候选在 sequential/balanced 中交换。
Indexer LSE、KL 和 sparse LSE 仍满足冻结门槛，但 cutoff 集合变化使 output 在 ranks 1/2/5/6 合计
3708 个元素超出 `atol=rtol=5e-3`，最大绝对差为 `0.015411376953125`。预热后的两 plan 单步总执行
仍只需 `0.0415–0.0485s`，没有 timeout、collective stall 或遗留进程。

冻结 cuDNN frontend `1.26.0` 的 `indexer_top_k_wrapper` 仅暴露 `top_k/next_n/return_val/num_copy_bits`
等参数，没有 deterministic tie secondary-key 选项；底层 radix filter 在不同 grouped layout 下不能
独立满足同-backend ordered exact。

用户批准 Q15 方案 1，并进一步确认 secondary key 的方向为 canonical global compressed-block ID
升序（较小 ID 优先）。冻结后的可执行语义是：

1. Top-K 主键保持 raw FP32 Indexer score 降序；只有 score 逐元素 exact 相等时才比较 global ID。
2. cutoff 上严格大于 cutoff score 的候选必须全部保留；等于 cutoff 的候选按 global ID 升序补足到
   effective length。该规则必须扫描完整可见 score 行，不能只在 cuDNN 已返回的 K 项内排序。
3. 有效 Top-K 输出按 `(score descending, canonical global ID ascending)` 排列；padding 仍位于有效前缀
   之后并使用 `-1` sentinel。sample 内 local ID 加固定 block offset 后即为 canonical global ID，因此
   local/global 升序等价，但 public/result 语义始终以 global ID 定义。
4. 不允许以 epsilon、量化、score bit 改写或重采样实现 tie-break；near-equal-but-non-tied score 必须
   保持 raw score 决定的顺序。cuDNN wrapper ABI、raw score、LSE、output、KL、全部梯度及既有数值门槛
   不变。
5. 小型 Indexer 单测必须覆盖 cutoff 候选集合重选、内部 exact-tie 位序、不同 sample global offset、
   padding/空行和 near-equal 反例。CP1、CP8 与 128K shadow 对拍必须在进入正式 balanced capture 前
   验证该行为。

该裁决只批准上述 exact-tie production 行为变化，不批准放宽 comparator、修改 backend ABI 或改变
非 tie 数学。历史三次失败仍保留为审计证据；Q15 的实现与 CP1/CP8 复验通过后才可重新运行 128K
正式 profile。

### Q16 / Backend-native Top-K exact-tie（APPROVED，2026-07-20）

用户随后明确批准完全取消 exact-tie tie-break，并接受 sequential/balanced 在 backend-native cutoff
tie 上选择不同有效集合。Q16 只修订 Top-K selection/finalization 和 ID comparator；cuDNN ABI、raw
FP32 score、Indexer/sparse LSE、output、KL 与全部梯度数值门槛均不变。冻结的可执行合同如下：

1. 固定 cuDNN `DSA.indexer_top_k_wrapper` 是 production selection/order 的唯一来源。wrapper 返回的
   sample-local IDs 和位序必须逐元素保留；production 后处理只允许加 per-row canonical global block
   offset、按 `topk_length` 截取有效前缀，并把无效后缀写为负 sentinel。
2. production 不再读取完整 score row 做 tie resolution，不再扫描 cutoff 候选、执行第二次 Top-K、
   stable sort 或 `(score, global ID)` 双键排序；不得用 epsilon、量化、score bit 改写或重采样间接制造
   tie-break。exact tie 的有效集合与内部位序均不施加 secondary key。
3. sequential、balanced 和 independent pure-PyTorch reference 的 `topk_length` 必须逐行 exact；每个
   结果自身的有效前缀必须全为非负 global IDs 且逐行唯一，padding 必须为负。跨结果的有效 ID 集合和
   位序只保存为诊断信息，不再要求 ordered exact 或 canonical-set exact。
4. 小型 backend 单测必须证明 production 结果逐元素等于 cuDNN 原始 local IDs 加 global offset，并对
   同一 backend raw score 验证 selection 合法性：严格高于 cutoff 的候选全部保留，不选择低于 cutoff
   的候选，等于 cutoff 的候选可任取补足 effective length；非同分 score 仍按降序，exact tie 位序不限。
   near-equal-but-non-tied score 不得由 Magi 代码改写成 tie。
5. Indexer raw score 继续按 `atol=rtol=5e-3` 对齐；Indexer/sparse LSE 与 output 沿用
   `atol=rtol=5e-3`，KL、常规输入/参数梯度沿用 `atol=rtol=2e-2`，BF16 distributed KV gradient
   特例不变。若 sequential/balanced 因选择不同集合而未通过任一现有数值门槛，当前实现必须停止并
   请求用户裁决，不能把 Q16 解释为允许放宽数值误差。

2026-07-22 用户对上述第 5 条作出补充裁决：backend-native exact-cutoff-tie 可以产生不同 output。
当 sequential、balanced 或 reference 的 canonical Top-K 集合不同时，对应 Query 行的 output 不参与
跨结果 `atol=rtol=5e-3` gate，但每个结果自身必须全部有限，并保存 mismatch row、全量 output
max-abs 与 canonical-set 相同行的 max-abs。canonical 集合相同的 Query 行继续执行原 output 门槛；
Indexer/sparse LSE、raw score、KL、loss、全部输入/参数梯度门槛均不变，也不得对 near-equal score、
非 tie 行或其他 tensor 扩展此例外。production 仍原样保留 cuDNN IDs/位序，禁止二次 Top-K、排序或
secondary key。

Q14/Q15 的历史 artifacts 继续保留为审计证据；新的 correctness/profile 结果必须显式同时报告
`topk_backend_native_valid`、length/uniqueness、ordered/canonical 差异诊断和原数值门槛结果。

### Stage D/E / CP8 与 Profile 验收记录（PASSED，2026-07-20）

Q15 实现后的 CP8 natural forward/backward 通过，证据为
`artifacts/correctness/20260720T044108Z-cp8-cp8-natural-backward/`。预热后 8 ranks 执行时间为
`0.067113–0.073805s`，sequential/balanced 的 raw score、LSE、output、KL 与梯度对齐均通过。

正式 128K profile 位于 `artifacts/profile/20260720T044956Z-dsv4-flash-128k/`，按
balanced → sequential 捕获；两份 aggregate `.nsys-rep` 均包含 8 ranks × 5 steps，每个 rank/step
都能在 NVTX 时间线展开 `magi_dsa::indexer_score` 和 `magi_dsa::indexer_topk`。160 条逐 rank
phase 记录与 20 组 range 完整且 manifest 通过。Balanced 最差 score relative rank range 为
`0.021076`，最差 top-k 为 `0.017900`，10/10 step/phase 均低于 `0.05`。

用户明确取消了本次重复 smoke；正式 run 记录 `smoke=skipped_by_user`，release 继续引用既有短输入
smoke 证据，但 installed-wheel 镜像必须额外通过 CP8 natural forward/backward，不得只做 import 检查。

## 1. 设计结论

采用 Magi-MSA 的“全局 packed metadata → sample-relative fragments → cold solver →
immutable device maps → warm forward/backward → 对称反向归约”骨架，但不复制 MSA 的完整
Query relocation。DSA 的 raw window、compressed KV、attention Q 和 output 体积更大，且需与
模型原有 CP shard 对齐；因此主 token owner 保持稳定，只在 `ratio=4` 的 CSA 分支把低维
Indexer Query/weight 临时发给 balanced Indexer workers。`ratio=128` 的 HCA 没有 Indexer，
直接使用 causal-visible compressed prefix 加 raw window。

Magi-DSA v4 支持以下互斥 layer form：

| ratio | 语义 | Indexer | Attention keys |
| --- | --- | --- | --- |
| `0` | window-only | 无 | 最近 128 个 raw KV |
| `4` | CSA | top-k=512 | raw window + selected compressed KV |
| `128` | HCA | 无 | raw window + 全部 causal-visible compressed KV |

冻结的模型 recipe 来自官方 `DeepSeek-V4-Flash-Base`：`hidden_size=4096`、`q_lora_rank=1024`、
64 个 Query heads、KV/head dim 512、RoPE dim 64、Indexer 64 heads × 128 dim、`topk=512`、
raw window 128，BF16 为主路径且 sink 使用 FP32。配置事实来源固定为 revision
`8855555deef230a27a21a8d6f294b7b7497759b6`，不能与 V4-Pro 的 128 Query heads、
`q_lora_rank=1536`、`topk=1024` 混用。

Q1 已冻结为以下 ownership/autograd 合同：

- `MagiDSALayer` 属于模型侧，拥有并注册全部 Compressor/Indexer 可训练参数，负责把它们交给
  DSA functional 执行；optimizer、checkpoint/state dict 和模型参数 reducer 只从 layer 看见它们。
- `MagiDSARuntimeMgr` 是无参数 runtime，只拥有静态 plan、通信 metadata、device maps、stream/
  handle 与执行编排状态；它不注册、复制或隐藏任何 trainable parameter。
- runtime 不隐式 `detach`、安装改变梯度的 hook，或偷偷归约模型参数。Indexer 分支默认使用
  graph-connected `x/qr`；如需阻断 trunk 梯度，模型侧必须只对该分支显式使用 detached 输入。

Q2 已冻结为以下 kernel 工具链合同：

- 不另设 QDSL 仓库或 style guide；本文的 “QDSL” 只是 Magi 现有 Quack helpers + NVIDIA CuTe
  Python DSL 写法的简称。Q12 复审后版本精确锁定为 `nvidia-cutlass-dsl==4.5.0`、
  `nvidia-cutlass-dsl-libs-base==4.5.0`、`nvidia-cutlass-dsl-libs-cu13==4.5.0`、
  `apache-tvm-ffi==0.1.8.post0` 和 `quack-kernels==0.4.1`，后续升级仍必须单独评审。
- `magi_attention/kernel/cutedsl/msa_pack.py` 是直接风格参考：Python host wrapper 做 CUDA、
  contiguous、dtype、shape、alignment/architecture 校验；`@cute.jit` 负责 specialization 与
  launch，`@cute.kernel` 实现 device body，调用方 stream 必须原样透传。
- 编译使用 symbolic shape、fake tensor 与 TVM-FFI，不为 JIT 创建真实输入；MSA 当前按
  `(dtype, feature width)` 缓存，DSA 因目标架构/算子更多，采用有界
  `(operation, major, minor, dtype, feature width)` key。warm path 不做 host sync、plan/map 构建
  或逐 token Python 循环；输出/workspace 分配策略必须显式、可测并计入 profile，而不是笼统
  假设 wrapper 零分配。copy 明确 vector width，BF16/FP32 归约使用 FP32 accumulator。

Q3 的模型、依赖、机器与 profile workload 合同已冻结：

- DeepSeek 配置使用上述官方 V4-Flash revision。FlashMLA 使用官方 `main` revision
  `9241ae3ef9bac614dd25e45e507e089f888280e0`；cuDNN backend 使用 `9.24.0`
  （CUDA 13 wheel build `9.24.0.43`），cuDNN frontend 使用 `v1.26.0` tag commit
  `35fd7b0d0e1d4952b904c79341c5e84e3af0a328`。该组合由用户在 Q11 方案 1 复审中批准；镜像和结果
  必须记录 exact revision/version，后续不得因上游分支变化而静默漂移。
- 目标为单机 8 张 B300（SM103），`world_size=rank_size=cp_size=8`。Profile 固定为单条 BF16 CSA
  `ratio=4` 的 128K causal sequence，`cu_seqlens = [0, 131072]`。sequential 与 balanced 使用完全相同
  且显式记录的 seed、输入、配置和镜像，各捕获 5 个 forward steps；运行中不得重采样或改变
  layout。sequential 是 Magi-DSA 内部 baseline，不承担均衡门槛，也不是外部实现比较。

Q4–Q10 按 Magi-MSA 代码/测试与 DeepSeek-V4 官方参考冻结为以下合同：

- **Q4 / 分布式出口**：release 验收使用 CP8，CP1 是数学/reference 边界，CP2 是最低多卡回归。
  production operator 接收 contiguous owner-local CP shard，并返回相同 owner-local Query 顺序；不把
  replicated-global tensor 放入公共 operator boundary。与 MSA 一样，dispatch 是 attention 计算之外的
  显式层；global gather/dispatch helper 只用于测试和诊断。
- **Q5 / Indexer 调用**：每个 CSA rank、每个 step 对 score 和 top-k 各有一次 grouped logical backend
  invocation，使用 varlen/fragment metadata 覆盖该 rank 的所有连续或不连续区域。禁止按 fragment 或
  shape bucket 从 Python/runtime 重复调用。一个 logical invocation 内部允许固定、有界且可观测的
  backend 子 kernel；因此合同约束的是一次 operator 调用和一次 profile 区间，而不是声称 GPU 上只有
  一个物理 kernel launch。
- **Q6 / KL placement**：保留 selection-worker、selected-KL-owner。worker 把 top-k IDs、length 和
  Indexer LSE/冻结 ABI 所需的最小 auxiliary state 一并逆置换回 token owner；owner 已持有 attention
  Q、sparse LSE、本地 `q_idx/weights`，并按 consumer union 取得 compressed KV/Ki，因此在 owner
  执行 selected-KL backward。selection 本身不可微，不把整个 Indexer forward/backward 迁到 worker。
- **Q7 / 通信原语**：沿用 MSA 的 `magi_attention.comm.primitive.all2all_v` 作为所有 typed-row route、
  QW permutation 和 auxiliary restore 的统一 production primitive；它封装 NCCL
  `all_to_all_single`。反向使用交换 split sizes 的对称 All2AllV，并在 collective 前后用 device CSR
  合并。DSA 代码不直接混用另一套 GroupCast/GroupReduce 或裸 NCCL API；若以后替换原语，必须作为
  独立性能变更验证 collective DAG、数值和字节数。
- **Q8 / HCA 合同**：数学语义由 DeepSeek-V4 官方论文及固定 revision 的官方 reference code 冻结；
  本仓库 pure-PyTorch CP1 reference 将其转成 packed/training 可执行合同。HCA 以 128 个样本内连续
  token 为一个不重叠 receptive field，用 token-wise gate 加 learnable positional bias 后沿 token
  维 softmax 加权；训练中不足 128 的 sample tail 不产生 compressed row，增量推理则缓存到完整块。
  Query `p` 只可见 block IDs `[0,floor((p+1)/128))`，再拼接最近 128 个 causal raw KV；HCA 无
  Indexer、top-k 或 Indexer KL，但与 CSA 一样使用每 Query head 的 FP32 learnable sink。语义层以
  global block IDs 和 `-1` sentinel 表示无效项；生产 kernel 的 range/CSR/explicit-index 物理布局由
  固定 FlashMLA/cuDNN wrapper ABI 冻结，不能反过来改变上述数学语义。
- **Q9 / 正确性**：继承 MSA 已验证门槛并采用 Q13/Q16 修订：sequential、balanced 与 independent
  reference 的 natural Top-K effective length 必须 exact；每个结果自身的有效 global ID 前缀必须非负、
  逐行唯一，padding 必须为负。跨结果的 ID 集合与位序允许因 backend-native exact tie 而不同，只作
  诊断，不再执行 ordered/canonical exact gate。production exact tie 不施加 secondary key；小型
  Indexer 单测必须验证 cuDNN 原始 ID/位序被原样保留、global offset、cardinality/padding，以及严格高于
  cutoff 的候选不遗漏、不选择低于 cutoff 的候选、同分候选可任取补足，且不得以 near-equal 代替
  exact tie。Indexer raw score 与其他 forward 值使用 `atol=rtol=5e-3`，KL、常规输入和
  Compressor/Indexer 参数梯度使用 `atol=rtol=2e-2`；仅确实经过 BF16 distributed reduction 的
  shared/compressed-KV gradient 可使用 `atol=1e-8`、`rtol=5e-2` 和 mismatch ratio `<=0.08`。
  MSA 的 solver 只有预测 `max/mean-1 <= 0.05`，没有实际 score/top-k profile 门槛；Magi-DSA
  另行冻结实际验收：5 个 captured steps 中，score 和 top-k 各自每一步都必须满足
  `relative_rank_range=(max-min)/mean <= 0.05`。`rank_range_ms` 必报但不设硬门槛，sequential
  只作 baseline。
- **Q10 / 范围**：要求 BF16 causal packed/varlen prefill 的 training forward/backward；普通
  `torch.no_grad()` 通过同一 operator 可用，但不另做 inference 专用快路径。decode/KV cache、FP8/FP4、
  TP、CUDA Graph、selected-KV routing 和 attention recompute 均不在 v4 范围内。

冻结 correctness matrix：

| 层级 | 路径 | 必须验证 | deadline boundary |
| --- | --- | --- | --- |
| CP1 | ratio 0/4/128，kernel 对 pure-PyTorch reference | natural forward/backward；raw score tolerance；Top-K effective length exact、各结果有效 ID 唯一/padding 合法；ID 集合/位序仅诊断；LSE、output/KL、输入与模型参数梯度 | 普通测试 10 分钟 |
| CP2 | CSA sequential 与 balanced，对 CP1 reference | natural 完整前后向；两 plan/reference 的 Top-K length exact、各自结构合法；ID 集合/位序仅诊断；raw score/LSE/output/KL/全部梯度满足 Q9 | setup/JIT/prewarm 在外，实际执行 60 秒 |
| CP8 | release CSA sequential 与 balanced，并覆盖 ratio 0/128 release smoke | natural 完整前后向；沿用 CP2 Top-K/raw-score 合同；通信对称性、调用次数和 Q9 其余数值合同 | setup/JIT/prewarm 在外，correctness execution 60 秒 |
| backend-native Top-K unit | 小型 Indexer score/top-k | cuDNN IDs/位序原样保留、global offset、cardinality、padding/空行、cutoff 合法性、near-equal 反例 | 普通测试 10 分钟 |
| fused V4 RoPE unit | BF16 Indexer-Q、sample-relative 大位置、forward/backward | 对 eager GPT-J trailing RoPE 满足现有梯度门槛；输入不别名/不改写；覆盖 two-inflight、retain-graph 与梯度累积 | JIT/autotune 在计时外，普通测试 10 分钟 |

## 2. 与 Magi-MSA 的继承关系

| 方面 | Magi-MSA | Magi-DSA v4 |
| --- | --- | --- |
| Query ownership | 完整 Q 可按多个不连续 fragment 搬到任意 rank | 主 Q/output 保留在原 token owner；仅 CSA Indexer 工作重分配 |
| K 可见区 | 每个 fragment 的完整 causal prefix | 128-token raw window + ratio 4/128 compressed bank |
| Indexer | 同一 rank 上 packed fragments 共同调用 | CSA worker 处理多个不连续 fragments；HCA 无 Indexer |
| top-k | 产生后始终本地消费 | worker 产生 int32 IDs，再恢复到 Query owner |
| KV 通信 | consumer 所需 prefix 求并集，接收后本地展开 | 四类 typed rows 按 consumer union 静态路由 |
| backward | packed 梯度先 CSR 合并，再反向 collective | 保留同一原则，并按 payload 独立归约 |
| hot/cold path | plan 构建与 kernel 执行分离 | `prepare_execution` 与 `calc_dsa` 严格分离 |

直接继承的不变量包括：fragment 使用 sample-relative `[q_begin,q_end)`；全体 fragments 对非空
Query 精确覆盖一次；每个物理 row 对同一 consumer 最多发送一次；destination map 完整；
forward route 必有可验证的 reverse map；rank 0 求解并广播，warm path 不再碰 Python plan。

必须改变的一点是 distributed input 语义。MSA 的 `dispatch(x_global)` 假定每个 rank 已持有
完整 global tensor；模型内 DSA 不复制该合同。DSA 直接接收模型已经持有的 owner-local
CP shard，并返回同顺序 owner-local output；仅测试工具可做全局 gather/restore。

## 3. Plan 与区域描述

全局 packed batch 仍由 `cu_seqlens` 描述。plan 逻辑上分为两层：

1. **Owner plan**：记录每个 rank 原有的 `(sample_id, q_begin, q_end, owner_local_begin)`；v4
   保持连续 CP ownership，不因 solver 改变主 Q/output 布局。
2. **Indexer plan（仅 CSA）**：在 owner fragment 之上切出 assignment atoms，并记录
   `(sample_id, q_begin, q_end, owner_rank, owner_local_begin, worker_rank,
   worker_local_begin)`。同一 worker 可拥有多个样本以及同一样本中多个不连续区域。

Indexer dispatch map 把 owner-local `q_idx + weights` 排列到 worker-local buffer；restore map
是严格逆置换，把 `[T_worker,512]` int32 top-k IDs 放回 owner-local Query 顺序。每个 fragment
还携带 sample-relative Query position 和该 sample 的 compressed-key range，使 causal mask 不受
临时 packing 顺序影响。压缩 block 使用 `(sample_id, logical_block_id)` 标识；由包含该完整
block 最后一个 token 的 rank 唯一负责生成，跨 owner 边界所需 hidden rows 通过 overlap route
取得，避免重复压缩。

CSA 的 cost model 必须分别校准 score 与 top-k：score 至少取决于 Query 数和可见 compressed
block 数，top-k 至少取决于 Query 数、候选宽度和 grouped shape；fragment launch/packing、QW
dispatch、Ki route 和 top-k restore 作为容量与次级成本。首要目标不是仅平衡 token 数或一个合并
总成本，而是让每 rank 的单次 `indexer_score` 与单次 `indexer_topk` 都接近相同。128-token 对齐
可作为优先边界，sample 边界和短尾部必须允许更小 atom。

## 4. High-level Architecture

```text
                         collective cold path
config + CP group + cu_seqlens + owner layout
                    │
                    ▼
      owner plan → CSA cost solver → typed routes/restore maps
                    │ validate/hash/broadcast/materialize
                    ▼
             immutable DsaExecutionHandle
                    │
     ┌──────────────┴─────────────── warm calc_dsa ──────────────┐
     │                                                            │
owner-local x/qr/q/latent_kv/sink                                 │
     │                                                            │
window/overlap routes → compressor → compressed KV/Ki routes      │
     │                                      │                     │
     │        CSA: q_idx+weights dispatch → Indexer workers       │
     │                                      │ score/mask/top-k    │
     │                         top-k IDs restore to token owner    │
     └──────── owner packs window + compressed indices ───────────┘
                                      │
                                      ▼
                         sparse attention → O, KL
```

模块边界为：public API、metadata/solver、runtime/handle、typed communication、packing/CSR
kernels、compressor/Indexer、external attention backend 和 pure-PyTorch Magi reference。静态
plan/handle 可缓存；send/recv tensors、CUDA events、collective work 和 native handles 必须
invocation-private，不能被两个 in-flight microbatches 共享。

## 5. Forward 数据流

### 5.1 公共前半段

公共数据边界是 owner-local `x`、`qr`、已由模型侧完成 partial RoPE 的 attention `q`、
`latent_kv`、FP32 `sink` 和 global packed metadata。`MagiDSAInput` 是纯数据对象，不拥有参数；
`MagiDSALayer` 使用自己的 Compressor/Indexer 参数调用 DSA functional，而 parameter-free runtime
仅提供 plan、通信与调度。默认 Indexer projection 直接消费 graph-connected `x/qr`。若模型只想
截断 Indexer→trunk 梯度，应在 layer 内对 Indexer 分支使用 `x.detach()/qr.detach()`，不能 detach
projection 输出，否则会同时切断 Indexer projection 参数梯度。

forward 首先异步启动两类低层路由：

- `WINDOW_KV`：取得每个 owner Query 的 128-token causal raw window；
- `OVERLAP_X`：取得只生成一次 compressed block 所需的跨 owner hidden rows。

接收端按 device map pack，compressor 只处理 sample 内完整 block；不足 ratio 的尾部不生成
compressed row。位置 `p` 可见的完整 compressed block 数按冻结 DSA 合同计算，当前证据为
`floor((p+1)/ratio)`。compressed KV/Ki 使用 global logical block IDs，不能把 receive-buffer
offset 当作模型 ID。

#### 5.1.1 Indexer-Q fused V4 RoPE

CUDA production 的 Indexer-Q projection 使用从
`Megatron-LM@d1384c2d95c4fb18a892c524aa9991441e83b9db` 的 DeepSeek-V4
`megatron/core/fusions/fused_mla_yarn_rope_apply.py` 薄适配的 Triton/autograd 算子。它保持 THD 的
token×head 调度、仅旋转 head 尾部 `rope_dim`、GPT-J adjacent even/odd 配对和 FP32 旋转计算；位置
继续直接使用 Magi
sample-relative `local_q_positions`，YaRN 逆频率继续由当前 `MagiDSAConfig` 公式产生。为避免每 step
生成 angle/cos/sin 表，kernel 在同一次 launch 内读取 position 与小型 FP32 逆频率并计算 sin/cos；
因此 Indexer-Q RoPE forward 的目标物理 launch 数是 1。

适配器不采用 Megatron 原始的原地输出/原地梯度 ABI：forward 分配私有 contiguous 输出并在同一
kernel 完成非旋转前缀复制与尾部旋转，backward 分配私有梯度并在单次 kernel 中应用逆旋转。该边界
保证输入不被改写，且 two-inflight、reentrant backward、retain-graph 和 gradient accumulation
不会共享可变 output/grad buffer。Triton JIT/autotune 必须在 capture 和 60 秒 post-compile watchdog
之外完成；warm path 不生成位置表、不做 host sync。CPU/pure-PyTorch reference 和 Compressor RoPE
继续使用原 eager 公式。

该替换不改变 q/weights 的 shape、dtype、owner-local 行顺序、`INDEXER_QW` payload、route split、
balanced plan、cuDNN Indexer ABI 或任何 Top-K 语义。数值仍由 raw score/LSE/output/KL 与全部梯度的
冻结门槛约束；exact cutoff tie 仍只按 Q16 处理，不能用融合造成的舍入差异放宽其他 gate。

### 5.2 CSA（ratio=4）

1. token owner 从本地 `x/qr` 生成低维 `q_idx` 和 per-head weights，同时 overlap-x 通信与
   compressor 工作继续进行。
2. 静态 query exchange 将 `q_idx + weights` 发给 balanced workers。`COMPRESSED_KI` route
   把每个 worker 所负责 fragments 的 causal key-prefix 并集送到该 worker；重叠 prefix 只传一次。
3. worker 依据 fragment 的 sample ID、`q_begin` 和 key range，在一次 grouped logical backend
   invocation 中完成该 rank 全部 fragments 的 score → causal mask → cuDNN Top-K → backend finalize。
   finalize 只做 global offset、length 和 padding，不执行 tie-break。ragged shape
   由 varlen metadata 表示；不得用 Python per-fragment 或 runtime shape-bucket 循环替代。backend
   内部有界子 kernel 必须另外记录 launch count。
4. top-k IDs、length 和 Indexer LSE/ABI 所需 auxiliary state 通过 inverse-permutation restore
   exchange 回到 token owner。attention Q 从未离开 owner。
5. `COMPRESSED_KV` route 向 attention owners 提供这些 Query **可能**选择到的 compressed rows。
   owner 将 compressed IDs 与 raw-window IDs 组成 kernel 所需索引，调用 sparse attention，输出
   保持 owner-local 顺序。
6. owner 使用恢复的 top-k/Indexer LSE 计算 fused selected-KL：owner 已有 Q、sparse LSE、原始
   `q_idx/weights`，`COMPRESSED_KI` 的 consumer union 同时包含 KL owners 和 selection workers。
   custom backward 在 owner 产生 `dq_idx/dweights/dKi`，保持与 MSA “KL 跟随 sparse-attention
   saved state”相同；non-differentiable selection 不建立跨 rank autograd 图。Ki 一对多流量必须
   profile，但不通过迁移整个 Indexer backward 来掩盖。

top-k 只减少 sparse-attention 计算，不自动减少 v4 的静态网络流量。Ki 必须在选择前可见，而静态
compressed-KV plan 在 top-k 产生前已经确定。selected-KV routing 会增加依赖链和动态通信，明确
排除在 Magi-DSA v4 范围外。

### 5.3 HCA（ratio=128）

HCA 只执行 overlap-x → non-overlap compressor → `COMPRESSED_KV` route。每个 Query 使用 raw
window 与全部 causal-visible compressed rows，不生成 `q_idx/weights/Ki/top-k/KL`，也不执行
Indexer dispatch/restore。compressed prefix 应优先用紧凑 range/CSR 描述；若外部 sparse kernel
只接受显式 indices，则由 device kernel 生成并记录峰值显存，禁止在 Python/host 上展开
`[T,Kmax]` 大矩阵。

### 5.4 Window-only（ratio=0）

只保留 `WINDOW_KV` route 与 sparse/window attention。无 compressor、compressed payload、
Indexer 或 KL。该分支既是模型 layer form，也是排除 compressor/Indexer 问题的 correctness
reference。

## 6. 通信流与反向流

| payload | producer → consumer | forward | backward |
| --- | --- | --- | --- |
| `WINDOW_KV` | raw-KV owner → attention owner | unique-row All2AllV | dKV reverse All2AllV/CSR 回 owner |
| `OVERLAP_X` | hidden owner → compressed-block owner | unique-row All2AllV | dX reverse All2AllV/CSR 回 owner |
| `COMPRESSED_KV` | block owner → attention owners | causal-visibility union | dCompressedKV FP32 合并后回 block owner |
| `COMPRESSED_KI` | Ki owner → CSA workers/KL owners | causal-visibility union | dKi FP32 合并后回 Ki owner |
| `INDEXER_QW` | token owner → CSA worker | 静态 All2AllV/permutation | selection 不可微；KL 在 owner 直接产生输入梯度 |
| `INDEXER_AUX` | CSA worker → token owner | top-k/length/LSE inverse permutation | 保存供 sparse/KL backward，无 collective gradient |
| model parameter grads | local autograd → `MagiDSALayer`/外层 reducer | 无 | runtime 不做隐藏 all-reduce；layer 明确声明所需归约 |

所有数据面交换统一调用 Magi `all2all_v` wrapper；typed row route 与 permutation 的差别只体现在
split sizes 和 device maps，不在 DSA 内直接调用裸 `all_to_all_single`，也不混入第二套
GroupCast/GroupReduce DAG。每种 layer form 在 handle 中冻结独立 collective DAG 和 hash；所有 rank
严格按同一顺序进入，零 row rank 也必须参与合法空 collective。不得依赖 Python 条件导致某 rank
少进一次 collective。

backward 不重新选择 top-k。owner 保存 top-k、O、LSE 及冻结 ABI 要求的最小输入；必要的 raw/
compressed banks 可按同一静态 plan 重新 fetch 和 recompute。sparse backward 产生 dQ、dwindow、
dcompressedKV 和 dSink；CSA selected-KL 产生 dKi 与 Indexer 参数梯度。重复 packed row 先在接收
布局用 CSR 以 FP32 accumulator 合并，再执行 reverse collective，owner 侧再次合并到唯一 row；
最后才转换为公共输入 dtype。由 token/KV 路由导致的输入梯度通信属于 DSA functional 的数学
反向；Compressor/Indexer 参数梯度则返回 `MagiDSALayer`，由 layer 或训练框架按已声明的 CP/DP
ownership 归约。runtime 不得隐藏参数 all-reduce。若 sink 是模型参数，也服从同一规则。

必须验证两个 overlap window：forward 的 window/overlap/compressed fetch 与 Q/Indexer projection，
以及 backward 的 dKi reverse 与 sparse backward。event、buffer 的 `record_stream` 和异常 drain
语义是 correctness 合同，不是纯性能细节。

## 7. 初始化与 warm 调用流程

1. **冻结依赖与语义**：记录 FlashMLA、cuDNN frontend、Quack/CuTe DSL、Magi 及
   submodule 完整 revision；校验 GPU arch 和 backend capability。
2. **建立 layer/runtime**：模型先创建拥有 Compressor/Indexer 参数的 `MagiDSALayer`；所有 rank
   再以相同 config、CP group、layer ratio 和 policy 创建无参数 runtime。初始化先做不触发
   collective 的本地 schema 与 parameter-ownership 校验。
3. **绑定 workload**：收集/校验 global `cu_seqlens`、owner-local counts/capacities 和 layout hash；
   rank 间 hash 不一致时在任何数据面 collective 前失败。
4. **构建 owner plan**：精确覆盖 packed samples，确定 raw token owner、compressed-block owner、
   window 与 overlap 需求。
5. **求解 CSA plan**：仅 ratio=4 在 rank 0 以独立 score/top-k cost model 分配 fragments；
   sequential plan 同时作为 correctness reference 和 5-step profile 的 Magi-DSA 内部基线。
6. **生成并验证通信图**：构建 typed send/receive、query exchange、top-k restore、pack/CSR maps；
   检查 coverage、逆映射、send/recv symmetry、dtype/row width 和全 rank collective order。
7. **广播与 materialize**：广播 immutable host plan/hash，在各 rank 创建 resident CUDA int32 maps、
   kernel specialization 和容量元数据。cache 必须有显式上限/clear，不允许按 workload 无限增长。
8. **dry run/health check**：小输入执行一次 route 与 reverse-route 自检，确认 native handle、stream
   和错误传播；随后返回不可变 `DsaExecutionHandle`。
9. **warm path**：`calc_dsa(input, handle)` 只做本地 type/shape/device/handle identity 检查并执行
   已冻结 DAG；禁止隐式 prepare、solver、object broadcast、host metadata sync 或 map 构建。

批准的 high-level 调用形式如下；精确 Python 命名若需改变，必须同步本文、公共文档和测试：

```python
layer = MagiDSALayer(config)  # owns Compressor/Indexer parameters
runtime = MagiDSARuntimeMgr(config, cp_group, policy="indexer_balanced")  # no parameters
handle = runtime.prepare_execution(packed_meta, device, local_token_capacity=capacity)
output, kl = layer(MagiDSAInput(x, qr, q, latent_kv, sink, packed_meta), runtime, handle)
```

## 8. 5-step Profile 验收

验收只运行 Magi-DSA 的 CSA sequential 与 balanced path，不设置外部实现或训练 E2E 比较项。
workload 固定为单条 131072-token BF16 causal sequence，`cu_seqlens = [0, 131072]`、`ratio=4`、
`world_size=rank_size=cp_size=8` 和 8 张 B300。两种 plan 必须使用完全相同且显式记录的 seed、输入、
配置和软件镜像；各自的 setup、plan materialization、JIT 和 warmup 必须在 capture range 外，并
各自使用 immutable handle 连续捕获恰好 5 个 forward steps。profile 根 seed 固定为 `0`；owner-local
输入使用稳定的 `(seed, tensor_name, rank)` SHA-256 派生子流，模型侧共享的 sink 使用 rank `-1` 子流。
每 rank 保存全部输入和模型参数的 SHA-256，sequential/balanced 对应 rank 的 hash 必须 exact。

用户于 2026-07-20 另行批准一份不替代上述正式 release profile 的 balanced-only
forward+backward 诊断。它复用完全相同的 8×B300、BF16、CSA ratio=4、128K、seed=0 输入和固定依赖，
合同如下：

1. 2026-07-22 用户最终将本诊断收窄为仿照 Magi-MSA 的 DSA-core：sequential 与 balanced 各自在
   capture/prewarm 前执行一次 `TOKEN_LAYOUT(x)` 和一次 deterministic profile projection，把
   `MagiDSAInput.x/qr/q/latent_kv` 分别 detach 为梯度 leaf，再各完成 3 次
   `forward → backward(dout, dkl) → parameter-gradient all-reduce` prewarm；capture 内仅运行 balanced，
   连续执行 5 个 steps，不做逐 step synchronize。每步从同一组 fixed post-projection leaves 重新生成
   `calc_dsa` autograd graph，capture 内不得出现 `TOKEN_LAYOUT.forward/backward` 或
   `magi_dsa::module::model_projection::*`。
2. capture 外用固定 seed 生成一份 source-order global BF16 random `dout`，以
   `1/global_output_elements` 缩放后再按 sequential/balanced 各自的 `local_query_global_rows` 取出
   plan-local contiguous view，确保两个 plan 的 canonical Query 行使用同一、且与原 global-mean loss
   同量级的 backward seed；`dkl` 固定为 FP32 scalar one。capture 内直接执行：

   ```text
   torch.autograd.backward((output, selected_kl), (local_dout, scalar_one))
   ```

   不构造 scalar loss，aggregate trace 中 `magi_dsa::loss` 必须为零。

3. 每 step 的 range 开始后、forward 前对 `MagiDSALayer` 使用 `zero_grad(set_to_none=True)`，并把
   post-projection `x/qr/q/latent_kv` leaves 与 sink grad 设为 `None`；五步互不累积梯度。backward 后由
   模型侧逐项 all-reduce 共享 sink 与全部 layer 参数梯度，保持 runtime 无参数、且不在 runtime 内归约
   模型参数。逐 step profile-control 记录在五步 capture synchronize 后统一落盘，相邻 step NVTX 之间
   不执行文件 I/O。
4. 不运行 optimizer step。capture 停止后使用同一输入/参数运行一次 sequential DSA-core 前后向 shadow；
   Top-K 按 Q16 要求 effective length exact、各结果自身结构合法，集合/位序仅作诊断；canonical-set
   相同行的 output、KL、全部输入梯度及参数梯度继续按既有门槛检查，canonical-set mismatch 行的
   output 只要求两侧有限并保存诊断，BF16 `latent_kv` reduction gradient 沿用
   mismatch-ratio 特例。`dx/dqr/dq/dlatent_kv` 分别按静态 route canonicalize 后比较；projection
   backward、合并后的 `dlocal_x` 与 source-owner `dx` inverse route 不在本隔离 profile 内重复验证，
   继续引用 CP8 natural correctness。
5. 外层 NVTX 为 `$Magi_DSA/capture_five_forward_backward_steps`；每个 step 必须分别包含一次
   `magi_dsa::forward`、`magi_dsa::backward` 和
   `magi_dsa::parameter_gradient_allreduce`。已有 `magi_dsa::indexer_score/indexer_topk` 继续嵌套于
   forward 且各出现一次。所有 phase GPU 时间仍通过 NVTX 内 CUDA runtime launch 的 correlation ID
   关联 CUPTI kernel 求和；每个 rank/step 的 forward/backward 必须分别恰有 4 个
   `ncclDevKernel_SendRecv`，任何 `TOKEN_LAYOUT` 或 model-side profile projection NVTX/kernel 均使该
   run 无效。五步提交后的唯一 `torch.cuda.synchronize()` 位于外层 capture range 内，与 Magi-MSA
   capture loop 对齐。
6. 报告逐 step 保存上述五个 phase 的每-rank原始时间及 min/max/range/relative range；本诊断对
   backward 与参数梯度归约只报告、不新增 5% 硬门槛。正式 forward-only profile 的文件、门槛和
   release 地位保持不变。

用户于 2026-07-23 又批准一份不替代上述诊断的 W+CSA+HCA 合并五步 profile。它把
W(`ratio=0`)、CSA(`ratio=4`) 和 HCA(`ratio=128`) 作为三张独立 Attention autograd graph 合并到
同一个 training-step range，而不是把 balanced CSA output 直接作为 HCA 的跨层输入。每 step 的
forward 顺序为 W→CSA→HCA，backward 顺序为 HCA→CSA→W；三者均在 capture 前准备 fixed
post-projection leaves 与同一 global mean-scaled `dout` 的 plan-local view，capture 内不含 layout、
projection、loss、optimizer 或梯度累积。三次 backward 后统一做一次模型侧 parameter/sink bucket
AllReduce。模式级 SendRecv 硬合同为 W `1F+1B`、CSA `4F+4B`、HCA `3F+3B`，总计 `8F+8B`；
Indexer score/top-k 只在 CSA forward 各出现一次。外层 NVTX 为
`$Magi_DSA/capture_five_attention_suite_steps`，模式子层为
`magi_dsa::attention_suite::<w|csa|hca>::<forward|backward>`；capture 后另跑 CSA sequential
shadow，W/HCA 只验证其 ratio-specific gradient schema 与 finite。

最终证据为
`artifacts/profile/20260723T053951Z-dsv4-flash-128k-attention-suite-precomputed-dout/`。8 ranks 全部
PASS；440 条 logical phase records、7,890 个 kernels 的归因 coverage 为 1.0，未归因为零。模式级
W/CSA/HCA SendRecv 精确为 `1/4/3 F+B`，总计 `8F+8B`，统一 gradient bucket 每 step 只有一个 NCCL
AllReduce；TOKEN_LAYOUT、model projection、scalar loss 均为零。平均 W/CSA/HCA forward 为
`0.915652/46.212248/2.817966 ms`，backward 为 `2.364132/16.399909/12.853463 ms`，总
forward/backward/gradient-allreduce 为 `49.945866/31.617504/0.425274 ms`。CPU/GPU step gap 分别为
`1.550–3.508 us` 与 `0.800–1.504 us`。CSA shadow 的 8 个 exact-cutoff set-mismatch 行只豁免
output，其余行 max-abs 为 `9.765625e-4`；gradient max-abs 为 `4.6147034e-7`，三种 ratio 的全部预期
gradient 均 finite。115 项 SHA-256 manifest 已通过。

该 artifact 使用 2026-07-22 已被后续裁决推翻的完整 sparse-LSE teacher、full-domain predictor
和 dense Indexer backward 数学。其 W→CSA→HCA 调度、8F+8B 通信、NVTX/归因和性能数据仍可作为
历史执行证据，但不能作为 2026-07-23 selected-only/double-LSE 合同的 correctness、release 或正式
profile PASS；新合同必须使用新的 UTC run id 重新验收。

新合同的替代诊断证据为
`artifacts/profile/20260723T090409Z-dsv4-flash-128k-attention-suite-precomputed-dout/`。它使用
FlashMLA 同次返回的 compressed-prefix LSE、selected-only KL 和 sparse Indexer backward，并新增
mode-specific launch-thread NVTX 与 kernel-overlap 资源签名审计。8 ranks × 5 steps 的 440 条 logical
phase records、`1/4/3 F+B` 模式通信、`8F+8B` 总账、7,730 个 kernel 的 100% 归因、CSA shadow
correctness 和 117 项 SHA-256 manifest 全部 PASS。W/HCA 的 FlashMLA 三输出变体资源签名完全相同；
CSA dual-LSE 变体只在 registers/thread 上从 126 专用化为 128，cuDNN backward 四个 core kernel
在三种模式间保持完整资源签名一致。

用户于 2026-07-20 要求正式 capture 按 balanced → sequential 顺序执行，并使 NVTX 结构尽量与只读
Magi-MSA `balanced_5steps(1).nsys-rep` 对齐。该参考报告 SHA-256 为
`01a429b2c6d42532a99c39ba5e7a2e08ee594df56f20a5413eac474a882a288f`，由 Nsight Systems
`2026.3.1.157` 生成；冻结 DSA 镜像的 `2026.2.1.210` 不能向后打开它，因此采用报告内可恢复字符串
与 MSA 源码交叉确认命名，不升级冻结 runtime。DSA 外层使用
`$Magi_DSA/capture_five_training_steps`，step 使用 `<plan>/rank_<rank>/training_step_<step>`，输出层
使用 `<plan>/rank_<rank>/O`，Indexer 父层使用 `Magi_DSA/indexer`。这些父层只增强诊断层级，不参与
正式 GPU phase 计时。

为把 Nsight Systems timeline 中的物理 kernel 或 kernel 组稳定映射回模型计算，自 2026-07-20 起
production warm DAG 还必须提供 `magi_dsa::module::<branch>::<operation>` 层级化诊断 range。它们只在
host launch 周围执行 NVTX push/pop，不允许插入 CUDA event、同步、barrier 或改变 tensor 数学；CPU
reference 路径不发射这些 ranges。主要命名合同如下：

| 逻辑模块 | 稳定 NVTX 路径 |
| --- | --- |
| Main/Indexer Compressor | `compressor::{main,indexer}::{value_projection,gate_projection,overlap_assembly,nonoverlap_assembly,validity_mask,gate_softmax,compression_hadamard,support_reduction}` |
| Compressor RMSNorm | `compressor::<branch>::rms_norm::{variance_reduction,inverse_root,normalize_hadamard,scale_hadamard}`，并保留父 `rms_norm` |
| Compressor/eager reference RoPE | `<scope>::rope::{angle_generation,sincos,rotation_hadamard,output_assembly}`，并保留父 `rope` |
| Indexer projection | `indexer_projection::{query_projection,query_cast,query::rope,query::rope::fused_megatron,weight_projection,weight_scaling}`；CUDA Indexer-Q 的 fused 子层只应关联 1 个 forward kernel |
| Device packing/route | `packing::{cute_row_copy,cute_csr_reduce,<payload>::forward_copy,<payload>::backward_csr_reduce}` 与 `route::<ROUTE>::{forward,backward,restore}::<operation>` |
| Indexer score/Top-K | `indexer::score::{cudnn_backend,logsumexp}`、`indexer::topk::{cudnn_backend,backend_finalize}`；finalize 只做 global offset/length/padding |
| Attention assembly/backend | `attention::kv_bank_assembly`、`attention_indices::<operation>`、`sparse_attention::{forward,flashmla_forward,cudnn_backward}` |
| selected-KL | `selected_kl::{cudnn_indexer_recompute,cudnn_attention_recompute,validity_mask,target_log,predict_log,kl_hadamard,kl_reduction,cudnn_indexer_backward}` |

其中 `<branch>`/`<scope>` 必须由静态 layer form 或 route 名决定，不能包含 token 数、动态 shape、rank
或 step，以免污染 timeline 聚合。新增 ranges 是诊断子层：正式 score/top-k 时间仍只按外层
`magi_dsa::indexer_score`/`magi_dsa::indexer_topk` 内关联到的全部 CUPTI kernel duration 求和，不能用
子 range 排除 backend finalize、LSE 或其他 production 开销。

为使 cuDNN 核心调用不被上述小型前后处理 kernel 淹没，Indexer 还在对应 frontend wrapper 周围嵌套
两个醒目的稳定 range：`magi_dsa::CUDNN_CALL::indexer_score` 和
`magi_dsa::CUDNN_CALL::indexer_topk`。前者对应 CUPTI kernel 名中的 `IndexerForwardSm100`，后者对应
`IndexerTopKKernelVarlenDecode`。这些 range 标记的是无同步的 cuDNN wrapper launch 区间：若固定
frontend 在同一次 wrapper 内发射辅助 kernel，它们仍属于该 call range；不能把 NVTX CPU wall time
当作核心 kernel GPU 时间，核心 GPU 时间仍通过 CUDA runtime correlation ID 精确关联对应 CUPTI row。
forward+backward 诊断还必须提供 `magi_dsa::CUDNN_CALL::sparse_attention_backward` 和
`magi_dsa::CUDNN_CALL::indexer_backward`，分别包住固定 cuDNN DSA sparse-attention backward 与
selected-KL Indexer backward wrapper；同样只表示 launch 边界，不引入同步。

每个 `(plan, step, rank)` 必须分别产生一个逻辑 `magi_dsa::indexer_score` 区间和一个逻辑
`magi_dsa::indexer_topk` 区间。一个逻辑区间可包含 backend 内部必要的子 kernel，但不能把多个
fragment 的 Python 循环累计后伪装成“单次”；报告必须另外保存底层 kernel launch count。计时
以 Nsight SQLite 中 logical NVTX 内的 CUDA runtime launch 为边界，通过 correlation ID 关联所有
CUPTI kernel，再累加 kernel GPU duration；不使用 CPU NVTX wall time，不依赖易变的物理 kernel 名，
也不得在 score/top-k 内插入人为 barrier 改变真实执行。

对 plan `a ∈ {sequential,balanced}`、phase `p ∈ {indexer_score,indexer_topk}`、step
`s ∈ [0,4]`，从全部 ranks 的单次时间 `t[a,p,s,r]` 计算：

```text
rank_min_ms[a,p,s]         = min_r t[a,p,s,r]
rank_max_ms[a,p,s]         = max_r t[a,p,s,r]
rank_range_ms[a,p,s]       = rank_max_ms[a,p,s] - rank_min_ms[a,p,s]
relative_rank_range[a,p,s] = rank_range_ms[a,p,s] / mean_r(t[a,p,s,r])
```

`profile_5step.jsonl` 保存逐 plan/rank 原始值，`rank_ranges.json` 保存上述 20 组范围，
`REPORT.md` 逐 step 并排展示两种 plan 的 score/top-k rank min/max/range/relative range 及
balanced 相对 sequential 的变化。只有同时满足以下条件才可宣称 profile 通过：

1. 两种 plan 的 5 steps 全部完成，输出 finite，warm-path planning counters 保持为 0；
2. sequential 与 balanced 的 Top-K 符合 Q16 length/结构合同，集合/位序差异只作诊断；output、
   Indexer/sparse LSE 与 KL 继续满足既有数值门槛；
3. `2 plans × 5 steps × 8 ranks × 2 phases = 160` 条记录完整、无重复，每个逻辑调用次数恰好为 1；
4. balanced 的 score 与 top-k 在 **每一个** step 都满足 `relative_rank_range <= 0.05`，不能只检查
   五步平均或 median；`rank_range_ms` 必须报告但不设硬门槛，sequential 不承担 5% 门槛。

推荐命令合同为：

```bash
bash scripts/profile/run_5step.sh --world-size 8 --cp-size 8 --case dsv4-flash-128k \
  --plans sequential,balanced --steps 5 --skip-smoke
```

本次重跑按用户 2026-07-20 的明确指令不重复短输入 smoke；release 继续引用已通过且 manifest 完整的
`artifacts/profile/20260720T033324Z-cp8-short-profile-smoke/`，正式 run 的 `WORKLOAD.json` 必须记录
`smoke=skipped_by_user`，不能将跳过伪装成新一次通过。

结果写入全新的 `artifacts/profile/<UTC-run-id>/`。每个 plan 保存一份包含 8 workers 的 aggregate
`.nsys-rep` 及 SQLite export；从同一报告按 rank 导出 NVTX/CUDA runtime/CUPTI kernel 原始 JSONL，
不得为 per-rank 文件重复执行正式 5 steps。`profile_5step.jsonl`、`rank_ranges.json` 和并排报告放在
run 根目录。缺少任意 plan/rank/step、出现 NaN/负时间、逻辑调用次数不为 1、aggregate report 无法
打开或 per-rank Nsight 原始记录不完整时，该次 profile 整体无效。

本次正式结果已经固定为 `artifacts/profile/20260720T044956Z-dsv4-flash-128k/`。其
`SUMMARY.json`、`REPORT.md`、`profile_5step.jsonl`、`rank_ranges.json`、两份 aggregate report/
SQLite 与 SHA-256 manifest 均通过；不得用后续离线改写替换该正式证据。

## 8.1 Release 镜像与复现

Release image 从 clean 40 字符 revision 构建，固定 runtime/backend 依赖不变。构建阶段使用
`versioningit==3.3.0` 生成只包含 DSA 所需 Python runtime 的 `magi_attention` wheel，镜像内同时
保存 clean source snapshot；legacy Magi CUDA extensions 不属于 DSA wheel。镜像必须满足：

1. wheel 版本为 `1.1.1+g<40-char-revision>`，且 import path 位于 Python 安装目录
   （`site-packages` 或 Debian/NVIDIA 等价的 `dist-packages`），不得来自 source mount；
2. OCI label、环境变量、wheel SHA-256 与 source revision 相互一致；
3. 8-rank CP8 natural forward/backward 在 installed-wheel 模式通过，每个 rank 报告相同 revision、
   version 与 package path；
4. release artifact 引用已通过的正式 profile，保存 build/CP8 命令和日志、image ID、环境、硬件、
   submodules、最终报告与 SHA-256 manifest。

多卡 summarizer 的 `SUMMARY.json` 不要求重复写顶层 `result` 字段；release finalizer 必须用既有
schema 的 `case/world_size/result_count`、8 条 rank result、执行时间 `<60s`、rank 0 参数值检查、
phase audit 和 SHA-256 manifest 联合判定通过，再在 release 自身的 `SUMMARY.json` 写 `result=PASS`。

## 9. 主要风险与缓解方向

| 风险 | 影响 | 设计期缓解 |
| --- | --- | --- |
| CSA/HCA、sink、RoPE、完整 block、backend-native Top-K 或 KL target 在 Magi reference/kernel 间不同 | 静默训练偏差 | CP1 reference、sequential 和 balanced 对拍 Top-K length/结构与所有数值门槛；backend 原始 ID/位序和 cutoff 合法性单独小测 |
| layer/runtime ownership 被实现混淆 | optimizer 漏参数、重复归约或梯度链被截断 | runtime 无参数断言；layer state-dict/optimizer、默认梯度链与显式 detach 定向测试 |
| ragged fragments 无单次 grouped operator ABI | launch storm，balance 收益消失 | ABI capability check；一次 logical invocation 接收 varlen metadata，不接受 runtime bucket 或 fragment loop |
| Indexer selection 与 KL 位于不同 rank | 额外 Ki/target/state 通信 | 当前采用 selection-worker、KL-owner；用 consumer union 量化并验证流量 |
| top-k 不减少静态 compressed KV 网络量 | 长序列通信占主导 | 报告真实 bytes；selected-KV routing 明确排除在 v4 范围外 |
| HCA 显式 prefix indices 占用大 | 峰值显存/OOM | range/CSR 优先，device 生成 fallback，并对冻结的最大 workload 记录峰值 |
| collective 顺序或 rank-local 异常不一致 | 多卡死锁 | DAG hash、零长度 participation、60 秒 watchdog、fail-stop 与 fresh process-group 重试 |
| handle 缓存/共享 buffer 生命周期错误 | 泄漏、串批、reentrant 错误 | 有界 cache；plan 静态、work/event/buffer 调用私有；two-inflight/retain-graph 测试 |
| fused RoPE 原地覆盖或大位置 sin/cos 偏差 | 梯度串批、Indexer raw score 漂移 | 私有 forward/backward 输出；覆盖 128K 位置的 eager 对拍；CP1/CP2/CP8 原数值门槛保持不变 |
| SM103、FlashMLA/cuDNN ABI 或 FFA 能力不匹配 | build/runtime 失败或错误 fallback | 固定 image/revision，启动时 hard capability check；不把 head-dim≤128 的 FFA 路径冒充 DSA-512 |
| CuTe/Quack 版本漂移或 specialization key 不完整 | JIT/ABI 失败、错误复用 kernel | 跟随 Magi 精确锁版；升级单独评审；cache key 包含 operation、完整 arch、dtype 与 row width |
| host sync、planning 或 CPU NVTX 时间污染 profile | 虚假均衡结论 | warm counters、CUDA Event 与 Nsight 交叉验证；setup/JIT/warmup 排除在 5 steps 外 |
| 聚合多个 fragment 调用后只报一个 phase 总和 | “单次耗时”不可比较 | 每 step/rank 恰好一个逻辑区间，保存底层 launch count，并拒绝线性 Python fragment loop |
| strided/capacity/valid-length 合同不一致 | 错行或越界 | 公共 schema 明确 contiguous/capacity；device maps 带 row-count 与 plan hash 校验 |

## 10. Review 结论

Q1 已关闭：`MagiDSALayer` 拥有 Compressor/Indexer 参数，`MagiDSARuntimeMgr` 无参数且不隐式
改变 autograd；Indexer→`x/qr` 默认连通，detach 只能由模型侧显式决定。

Q2 已关闭：不另设 QDSL 仓库或 style guide；跟随所选 Magi base revision 的 Quack helpers +
NVIDIA CuTe Python DSL 精确依赖与 `msa_pack.py` 风格。

Q3 已关闭：使用官方 DeepSeek-V4-Flash recipe、冻结 FlashMLA/cuDNN 版本、8 张 B300、
`world_size=rank_size=cp_size=8`，以及单条 `cu_seqlens = [0, 131072]` 的 BF16 CSA workload；
sequential/balanced 使用相同 seed 和输入各捕获 5 steps。Q9 也已关闭：balanced 的 score/top-k
必须逐 step 满足 `relative_rank_range <= 0.05`，`rank_range_ms` 必报但不设硬门槛，sequential
只作 baseline。

Q14/Q15 已作为历史裁决关闭，其诊断证据继续保留。Q16 是当前权威合同：production 直接保留固定
cuDNN wrapper 的 backend-native IDs/位序，只做 global offset、effective length 与 padding；跨
sequential/balanced/reference 的 ID 集合和位序不设 exact gate，但 length 与各自结构必须合法。Indexer
raw score tolerance、LSE/KL/全部梯度与 natural backward 门槛不变；output 按 2026-07-22 补充裁决，
仅 canonical-set mismatch Query 行免除跨结果 gate，其余行仍执行原门槛。

Q4–Q8 与 Q10 已由 MSA/官方 V4 证据收敛并随总体设计批准为第 1 节的冻结合同。当前没有剩余的
workload、接口或均衡门槛问题。设计批准不等同于未请求的实现动作；收到明确实现任务后，先确认
本文与 `AGENTS.md` 一致，再按冻结合同创建代码、测试、profile 和镜像脚本目录。

## 11. 2026-07-22 现行设计补充说明

本文只保留历史实现与证据；现行 CSA 架构以
`docs/design/magi_dsa_v4_design.md` 为唯一权威来源。2026-07-22 用户批准三项新的执行合同：

- 在固定 cuDNN frontend revision 上施加可追溯 no-sync 补丁，删除 dense unit-gradient
  路径的 GPU-to-host `.item()` 同步；2026-07-22 的连续五步 NSYS 诊断进一步证明该补丁还必须令
  `CUstream(0)` 保持 legacy default stream，避免 dK zero/backend/copy 被拆到不同 stream 后发生
  临时量提前 copy 与释放复用；
- CSA forward 的 compressed route 在同一 communicator 上改为
  `COMPRESSED_KI → COMPRESSED_KV`，以便 grouped Indexer 与 compressed-KV 传输重叠。
- backend-native exact-cutoff-tie 可以产生不同 output；canonical Top-K 集合不同的 Query 行只要求
  两侧 output 各自 finite 并记录诊断，不参与跨 plan output gate。非 mismatch 行及 LSE、KL、loss、
  全部梯度门槛保持不变。

上述第一项只作为 2026-07-22 的历史决定保留。2026-08-06 用户明确撤销既有 cuDNN frontend
host-sync/default-stream 补丁；该决定覆盖本节和本文其他位置的旧补丁描述，不得再作为现行合同。
现行 cuDNN frontend 必须使用
`v1.26.0@35fd7b0d0e1d4952b904c79341c5e84e3af0a328` 官方未修改源码并记录
`local_patches=none`。dense `.item()` 补丁不再使用；stream 问题由 Magi-DSA caller 显式选择非零
CUDA stream，并在进入受影响 wrapper 前 hard check 拒绝 stream 0。FlashMLA 条款不受此覆盖决定
影响；完整现行合同以 `docs/design/magi_dsa_v4_design.md` 为准。

2026-07-23 用户另行批准一项不改变数学和 `4F+4B` 账本的 overlap 调度：CSA 在 Indexer Compressor
完成后先提交 `COMPRESSED_KI` forward，再执行 Main Compressor；backward 由组合 attention+KL
autograd 节点先提交一次且仅一次 `COMPRESSED_KI` reverse，并与专用 CUDA stream 上的 sparse
attention backward 重叠。其余三条 reverse 暂不异步化。该变更及后续 profile 事实只以当前权威设计
和新 artifact 为准，本文仅保留时间线说明。

2026-07-24 用户进一步要求尽可能隐藏其余 backward All2AllV，因而上述“其余三条暂不异步化”只保留
为历史状态。现行实现用 handle-owned Main/Indexer/support streams 和 multi-output autograd late join
保持 `COMPRESSED_KI → COMPRESSED_KV → OVERLAP_X → WINDOW_KV` 固定 collective 顺序，同时分别以
sparse-attention、Indexer Compressor 和 Q/weight projection backward 遮挡四条 reverse。诊断
`artifacts/profile/20260724T020200Z-dsv4-flash-128k-forward-backward-dsa-core-precomputed-dout/`
在 40 个 rank-step 上证明四条 reverse 均有实际 GPU compute overlap，且 correctness、`4F+4B` 和
kernel attribution 全部 PASS。该时序的唯一架构事实仍以当前权威设计为准。

2026-07-24 用户继续要求在 W+CSA+HCA 合并诊断中尽量遮挡 W/HCA 通信。现行实现保持
caller-thread
`W F → CSA F → HCA F → HCA B → CSA B → W B → gradient AllReduce` 顺序，为三张独立 graph
配置独立 execution streams，并将 HCA route 固定为 forward
`OVERLAP_X → WINDOW_KV → COMPRESSED_KV`、backward
`COMPRESSED_KV → WINDOW_KV → OVERLAP_X`。HCA 使用 handle-owned high-priority
support-route/main streams；三张图的 completion events 在统一 gradient AllReduce 前汇合。
该跨模式 overlap 只属于独立图 profile，不能推广为真实 Transformer 层间依赖。最终 profile、
逐 route 覆盖与真实 step span 以当前权威设计和新的 UTC artifact 为准。

2026-07-24 用户随后明确推翻上述“跨模式 overlap”口径：W、CSA、HCA 的通信只能由同一 mode、
同一 forward/backward 阶段的自身计算遮挡，不能借用其他 Attention graph 的计算。现行
attention-suite 在四个跨 mode 边界以 CUDA event 建立 GPU happens-before，并将逐 route 审计限定为
mode-local non-route compute；跨 mode CUPTI kernel overlap 必须为 0。W 的 Window forward/reverse
以及 HCA 中受严格生产者/消费者依赖约束的 route 必须如实报告不可遮挡原因，不再追求通过其他 graph
制造覆盖。当前架构事实和新的 profile 证据仍只以权威设计为准。

本文记录的旧 profile/release artifact 均不包含该 frontend 补丁，且属于已被推翻的
Query-worker 架构，因此只作历史证据，不能被引用为当前设计的正式 profile/release PASS。

当前 local-Query 架构的 balanced forward+backward profile 保存在
`artifacts/profile/20260722T065216Z-dsv4-flash-128k-forward-backward/`。该 run 使用 installed wheel、
8×B300、BF16、CSA ratio=4、单条 128K 和五个 captured steps；8 ranks 全部 PASS。aggregate report
包含 5,920 个 runtime-correlated kernels，归因 coverage 为 1.0、未归因数为 0；6 / 131072 个
canonical-set mismatch Query 按上述裁决豁免，非 tie output max-abs 为 `9.765625e-4`，全部梯度
max-abs 为 `4.3772161e-7`。同一 report 的 autograd `process_temporal` 重提取给出 backward 平均
`20.109514 ms`，避免旧提取器仅按主线程 `globalTid` 得到约 `0.001 ms` 的漏计结果。
该 artifact 在上述 2026-07-22 DSA-only 收窄之前生成，每个 captured step 仍含外层
`TOKEN_LAYOUT.forward/backward`，因此只保留为旧口径证据；用户要求的新结果必须使用新的 run id
重新采集，且通过 trace 内 TOKEN_LAYOUT=0、4F+4B 的硬审计。

随后生成的
`artifacts/profile/20260722T075402Z-dsv4-flash-128k-forward-backward-dsa-only/` 已达到
TOKEN_LAYOUT=0 和严格 4F+4B，8 ranks、5,760 个 kernels、全归因与 shadow correctness 均 PASS；但其
梯度边界仍是 final-layout `local_x`，每个 captured step 会重新运行 deterministic profile projection。
用户于 2026-07-22 进一步要求按 Magi-MSA 把边界后移到 post-projection `MagiDSAInput`，因此该 artifact
继续作为上一版 DSA-only 口径证据，不能替代新 DSA-core profile；新 trace 还必须额外证明
`magi_dsa::module::model_projection::*` NVTX/kernel 为零，并消除相邻 step 之间的 profile-control 文件
I/O。

上一版 post-projection DSA-core artifact 为
`artifacts/profile/20260722T083239Z-dsv4-flash-128k-forward-backward-dsa-core/`。其 gradient boundary 为
post-projection `MagiDSAInput`，TOKEN_LAYOUT/model projection capture 均为零，40 个 rank-step 严格
4F+4B；8 ranks shadow correctness、5,040 个 kernel 的 100% 归因和 106 项 SHA-256 manifest 均
PASS。相邻 step 的 CPU NVTX 间隙已从上一版的 `1.611–8.442 ms` 降为 `1.624–5.440 us`，GPU kernel
间隙为 `0.800–1.280 us`，达到 Magi-MSA 式连续提交口径。该 run 仍构造 output-square + selected-KL
scalar loss；2026-07-22 用户进一步要求改为 capture 外预计算 global `dout` 与 unit `dkl`、capture 内
直接 backward，因此该 artifact 保留为上一版证据，不能替代新的 loss-free 诊断。两版均不替代正式
sequential+balanced forward-only release profile。

首次 raw O(1) random `dout` 尝试保存在
`artifacts/profile/20260722T090319Z-dsv4-flash-128k-forward-backward-dsa-core-precomputed-dout/`。五步 capture
本身完成，但 sequential shadow 的 `parameter::compressor.ape`/`input::q` gradient 超出原门槛，因此该
artifact 明确为 FAIL。该结果证明允许不同 canonical Top-K set/位序后，未经 global-mean 缩放的 O(1)
upstream gradient 会放大这些差异；本诊断没有据此屏蔽 tie 行或放宽 gradient gate，而是把预计算
random `dout` 固定为原 loss 的 `1/global_output_elements` 量级。

最终 loss-free artifact 为
`artifacts/profile/20260722T091353Z-dsv4-flash-128k-forward-backward-dsa-core-precomputed-dout/`。其 global
BF16 random `dout` scale 为 `2^-32`、`dkl=1`；8 ranks 全部 PASS，capture 中 scalar loss、TOKEN_LAYOUT
和 model projection 均为零，40 个 forward 与 40 个 backward 均精确 4 次 SendRecv。4,640 个 kernel
全部完成 runtime-correlation 归因，coverage 为 1.0、未归因为零；五步平均 forward/backward/
parameter-gradient-allreduce 为 `46.411299/16.576867/0.350376 ms`，CPU/GPU step gap 分别为
`1.928–11.880 us`/`0.800–1.344 us`。8 个合法 canonical-set mismatch 行只豁免 output；其余 output、
KL 和全部输入/参数梯度按原门槛通过，gradient max-abs 为 `4.6519563e-7`。114 项 SHA-256 manifest
全量校验通过。该诊断仍不替代正式 sequential+balanced forward-only release profile。
