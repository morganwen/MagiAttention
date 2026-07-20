# Magi-DSA v4 High-Level Design

状态：**已批准（2026-07-19），阶段 A–D correctness 与阶段 E profile 已通过；release 镜像收尾中。** Q1–Q10、
Q13 natural-only、Q14 independent-backend Top-K 对齐、Q15 deterministic exact-tie、128K workload
和 5% 均衡门槛构成冻结设计基线。
后续实现必须与本文同步；任何合同变更都要先暂停并重新评审。

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
  pure-PyTorch reference 按后续 Q14 comparator 对齐 natural top-k IDs/length，并保持 Indexer raw
  score/LSE、output、KL、全部输入梯度及 Compressor/Indexer 参数梯度门槛；
- natural top-k 的同-backend ordered exact 与 independent-reference canonical exact 语义由 Q14 精化；
  cutoff tie policy 由 Q15 冻结为 exact score 相等时 canonical global ID 升序，并由小型 Indexer
  单测覆盖，不再用完整 forced backward 改写 selection；
- 与 MSA 一样，plan setup、JIT、kernel 编译和完整 forward/backward prewarm 位于 60 秒 watchdog
  之外。CP2 gate 是预热后的 natural sequential/balanced 完整前后向在 60 秒内完成并与 reference
  对齐；
- 旧 forced backward timeout 与对齐诊断 artifacts 作为历史记录保留，但不再构成当前 blocker 或
  release gate。
- Q13 批准时的下一关是 CP2 natural sequential/balanced 完整前后向；该关随后已取得通过证据。当前
  下一关由 Q14 更新为按两级 Top-K comparator 重跑 CP8 natural backward。

### Q14 / 独立数值实现之间的 Top-K 位序合同（APPROVED，2026-07-20）

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
- **Q9 / 正确性**：继承 MSA 已验证门槛并采用 Q13/Q14/Q15 修订：同 backend、输入和 seed 的 sequential
  与 balanced 必须保持 ordered natural Top-K tensor/length exact；production 与独立 pure-PyTorch
  reference 逐行过滤 padding 后要求 effective length exact、有效 global ID 各自唯一、按 global ID
  排序后的有效 tensor exact，不要求不同 reduction backend 的内部位序 exact。production exact tie
  使用 canonical global ID 升序 secondary key；小型 Indexer cutoff-tie 单测必须验证完整 tie-boundary
  集合重选、cardinality 和 `(score descending, global ID ascending)` 位序，且不得以 near-equal 代替
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
| CP1 | ratio 0/4/128，kernel 对 pure-PyTorch reference | natural forward/backward；raw score tolerance；canonical global Top-K IDs/effective length/uniqueness exact；LSE、output/KL、输入与模型参数梯度 | 普通测试 10 分钟 |
| CP2 | CSA sequential 与 balanced，对 CP1 reference | natural 完整前后向；两 plan ordered IDs/length exact；各 plan 对 reference 按 canonical IDs/effective length/uniqueness exact；raw score/LSE/output/KL/全部梯度满足 Q9 | setup/JIT/prewarm 在外，实际执行 60 秒 |
| CP8 | release CSA sequential 与 balanced，并覆盖 ratio 0/128 release smoke | natural 完整前后向；沿用 CP2 Top-K/raw-score 合同；通信对称性、调用次数和 Q9 其余数值合同 | setup/JIT/prewarm 在外，correctness execution 60 秒 |
| tie unit | 小型 Indexer score/top-k | cutoff 全候选重选、global ID 升序 secondary key、cardinality、padding/空行、near-equal 反例 | 普通测试 10 分钟 |

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

### 5.2 CSA（ratio=4）

1. token owner 从本地 `x/qr` 生成低维 `q_idx` 和 per-head weights，同时 overlap-x 通信与
   compressor 工作继续进行。
2. 静态 query exchange 将 `q_idx + weights` 发给 balanced workers。`COMPRESSED_KI` route
   把每个 worker 所负责 fragments 的 causal key-prefix 并集送到该 worker；重叠 prefix 只传一次。
3. worker 依据 fragment 的 sample ID、`q_begin` 和 key range，在一次 grouped logical backend
   invocation 中完成该 rank 全部 fragments 的 score → causal mask → top-k/tie-break。ragged shape
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

用户于 2026-07-20 要求正式 capture 按 balanced → sequential 顺序执行，并使 NVTX 结构尽量与只读
Magi-MSA `balanced_5steps(1).nsys-rep` 对齐。该参考报告 SHA-256 为
`01a429b2c6d42532a99c39ba5e7a2e08ee594df56f20a5413eac474a882a288f`，由 Nsight Systems
`2026.3.1.157` 生成；冻结 DSA 镜像的 `2026.2.1.210` 不能向后打开它，因此采用报告内可恢复字符串
与 MSA 源码交叉确认命名，不升级冻结 runtime。DSA 外层使用
`$Magi_DSA/capture_five_training_steps`，step 使用 `<plan>/rank_<rank>/training_step_<step>`，输出层
使用 `<plan>/rank_<rank>/O`，Indexer 父层使用 `Magi_DSA/indexer`。这些父层只增强诊断层级，不参与
正式 GPU phase 计时。

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
2. sequential 与 balanced 的 output/top-k 符合 Q14 冻结的同-backend ordered exact 数值等价合同；
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

## 9. 主要风险与缓解方向

| 风险 | 影响 | 设计期缓解 |
| --- | --- | --- |
| CSA/HCA、sink、RoPE、完整 block、tie-break 或 KL target 在 Magi reference/kernel 间不同 | 静默训练偏差 | CP1 reference、sequential 和 balanced 对拍 natural top-k、KL 与全部梯度；tie 单独小测 |
| layer/runtime ownership 被实现混淆 | optimizer 漏参数、重复归约或梯度链被截断 | runtime 无参数断言；layer state-dict/optimizer、默认梯度链与显式 detach 定向测试 |
| ragged fragments 无单次 grouped operator ABI | launch storm，balance 收益消失 | ABI capability check；一次 logical invocation 接收 varlen metadata，不接受 runtime bucket 或 fragment loop |
| Indexer selection 与 KL 位于不同 rank | 额外 Ki/target/state 通信 | 当前采用 selection-worker、KL-owner；用 consumer union 量化并验证流量 |
| top-k 不减少静态 compressed KV 网络量 | 长序列通信占主导 | 报告真实 bytes；selected-KV routing 明确排除在 v4 范围外 |
| HCA 显式 prefix indices 占用大 | 峰值显存/OOM | range/CSR 优先，device 生成 fallback，并对冻结的最大 workload 记录峰值 |
| collective 顺序或 rank-local 异常不一致 | 多卡死锁 | DAG hash、零长度 participation、60 秒 watchdog、fail-stop 与 fresh process-group 重试 |
| handle 缓存/共享 buffer 生命周期错误 | 泄漏、串批、reentrant 错误 | 有界 cache；plan 静态、work/event/buffer 调用私有；two-inflight/retain-graph 测试 |
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

Q14 已关闭：同 backend 的 sequential/balanced 保持 ordered Top-K exact；production 对独立
pure-PyTorch reference 使用过滤 padding 后的 canonical global valid IDs/effective length/uniqueness
exact，同时保留 Indexer raw score tolerance 及全部既有 LSE/output/KL/gradient/tie/backward 门槛，且
不要求不同 reduction backend 的内部位序 exact。Q15 已关闭设计裁决：exact raw-score tie 使用
canonical global ID 升序 secondary key，完整覆盖 cutoff 候选集合与有效输出位序；非 tie 数学、backend
ABI 和全部既有数值门槛不变。

Q4–Q8 与 Q10 已由 MSA/官方 V4 证据收敛并随总体设计批准为第 1 节的冻结合同。当前没有剩余的
workload、接口或均衡门槛问题。设计批准不等同于未请求的实现动作；收到明确实现任务后，先确认
本文与 `AGENTS.md` 一致，再按冻结合同创建代码、测试、profile 和镜像脚本目录。
