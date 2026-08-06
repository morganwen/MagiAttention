# Repository Guidelines

## 当前权威设计与硬性边界

当前范围是官方
`DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1`。架构、tensor/index ABI、
collective 对账、backward、kernel→op 与 release 验收的唯一权威文档是
`docs/design/magi_dsa_v4_design.md`；负载均衡的结构性 cost/dispatch 依据是
`docs/design/README_dsv4_cp_dispatch_structural_balancing.md`。两者之外的 Base、
`shared_greedy`、owner-stable Query-worker/Top-K restore 文档都不得覆盖当前 Pro 合同。
`docs/design/magi_dsa_v4.md` 只保留 2026-07-19 至 2026-07-20 的历史实现、Q13–Q16 裁决和
profile/correctness 证据；与新设计冲突时一律由新设计覆盖，不得继续据此恢复 balanced worker、
`INDEXER_QW`、Top-K auxiliary restore 或 contiguous owner-stable layout。

依赖版本、backend-native Top-K、natural correctness、测试/Profile 证据和结果留存要求在不与新架构
冲突时继续有效。只有在用户明确发起实现任务后才能新增或移植 runtime、kernel、测试、构建系统或
性能优化代码。遇到 tensor schema、kernel ABI、CSA 数学或通信原语不清楚时，先核对当前权威设计、
Magi-MSA 和固定 backend 源码，再向用户提问；不得用隐式 fallback 或未登记的默认值补齐合同。

## 明确目标与验收条件

目标是 Pro 主干 61 层的训练级 hybrid attention：layer id `0..60` 中
`0,1,3,5,...,59` 为 31 个 HCA（`ratio=128`），`2,4,6,...,60` 为 30 个 CSA
（`ratio=4`）。每层拥有独立 Compressor/Indexer/attention 参数。独立 MTP 的
window-only `ratio=0` 仅保留兼容边界，不进入本次主干实现或正式 Pro 性能验收。

方案复用 MSA 的 sample-relative fragment、cold plan、device map、unique-row routing 和
反向 CSR reduction。`structural_balanced` 在进入 61 层前用 packed-global native causal
`AttnSlice.area` + `MinHeapDispatchAlg` 产生一份 ratio-independent 共享 Query layout，只对
pre-projection hidden `x` 执行一次 `TOKEN_LAYOUT` All2AllV。61 层内 Query owner 不再改变，
CSA/HCA 只构建 ratio-specific local/remote K route。一期不写入 B300 经验权重，也不把
CSA-first/HCA-second 或“忽略通信”冻结为 solver 先验。

Pro 训练 ABI 固定为 hidden `7168`、`qr=1536`、attention H128/D512、rope D64、
Indexer H64/D128/Top-K 1024、window 128；activation/线性权重为 BF16，score/LSE/sink/
accumulator 为 FP32，ID 为 INT32。

参数与执行职责已经冻结：`MagiDSALayer` 是模型侧模块，拥有全部 Compressor/Indexer 可训练
参数；`MagiDSARuntimeMgr` 不拥有任何可训练参数，只保存静态 plan、通信 metadata 与执行状态。
runtime 不得隐式 `detach`、注册梯度 hook 或归约模型参数来改变 autograd 语义。Indexer 分支默认
保留到 `x/qr` 的梯度链；若模型需要截断，必须在 `MagiDSALayer`/调用侧显式表达。

CSA 的当前执行合同如下：

- DSA 内 forward 只有 `WINDOW_KV`、`OVERLAP_X`、`COMPRESSED_KI`、`COMPRESSED_KV` 四条，
  CSA 严格按该 collective 顺序发起；Indexer Compressor 完成后立即 start `COMPRESSED_KI`，再执行
  Main Compressor 并 start `COMPRESSED_KV`，使 KI 通信由 Main Compressor 遮挡，随后 grouped
  Indexer 继续与 compressed-KV 传输重叠。backward 是对应四条 reverse All2AllV，其中
  `COMPRESSED_KI` reverse 先异步 start，并与专用 CUDA stream 上的 sparse-attention backward 重叠；
  `COMPRESSED_KV` 由独立 Indexer backward 遮挡；multi-output autograd late join 同时释放
  projection 与 support gradient，使 `OVERLAP_X/WINDOW_KV` reverse 由 Q/weight projection
  backward 遮挡。四条 reverse 的 communicator 顺序固定为
  `COMPRESSED_KI → COMPRESSED_KV → OVERLAP_X → WINDOW_KV`。外层
  `TOKEN_LAYOUT` forward/reverse 和
  model-side parameter/sink AllReduce 分别计时，不计入 4F+4B。
- local Query 直接执行每 rank/step 一次 grouped Indexer score 和一次 grouped Top-K；没有 Query
  worker、`INDEXER_QW`、distributed Top-K 或 Top-K restore。
- production 直接组合固定 FlashMLA 与 cuDNN DSA，不调用 Megatron
  `FusedIndexerSparseAttnFromTopkFunc`。FlashMLA indices 固定为 compressed 1024 列在前、window
  128 列在后，各区域用 `-1` padding，不按 `topk_length` 压紧。CSA 在冻结 924 基线上
  依次应用已审核的 13d dual-LSE 和 b764 Pro-H128/prefix-1024 增量，传
  `indexer_topk=1024`，同次 forward 返回完整 `sparse_lse` 与 compressed-prefix LSE：前者只服务
  sparse backward，后者只服务 KL attention teacher。独立 full-domain `indexer_lse` 仅作诊断，
  不参与 KL。所有 LSE 都是 detached state，不建立 autograd 边。
- KI/KV bank 使用两套独立 global-to-local map，不要求相同物理行序；selected teacher、clipping、
  缩放和 unit-gradient 预计算调度参考固定 Megatron revision，但 teacher/predictor 归一化域以权威
  设计 6.1 节为准。predictor 只在 selected Top-K 内归一化，并调用 sparse Indexer backward；
  未选中 candidate 的梯度必须为 0。
- HCA `ratio=128` 的 DSA 内 route 顺序冻结为 forward
  `OVERLAP_X → WINDOW_KV → COMPRESSED_KV`、backward
  `COMPRESSED_KV → WINDOW_KV → OVERLAP_X`。execution handle 为 HCA 创建逐 handle 的高优先级
  support-route/main-compressor streams；Window 可与 Main Compressor 正反向交叠。backward 使用
  无 gradient hook、无 trunk detach 的 branch-order gate 先提交 Window reverse，再释放 compressed
  分支的 Main Compressor backward；event、route buffer 和 gate state 必须保持逐 invocation 私有。
- Q/K 两侧 Hadamard、sample-relative Q RoPE、weights `1/sqrt(64)` 和 score `1/sqrt(128)` 是
  correctness 合同，不能由 backend wrapper 的实现技巧改变。
- FlashMLA/cuDNN 返回的 raw attention output 末 64 维仍在 RoPE 坐标系；output
  projection 前必须用本层 YaRN frequencies 和 sample-relative Query position 执行
  out-of-place inverse RoPE，backward 使用共轭的正向 RoPE，不得改写 attention saved state。

当前 Pro 模型、backend 和 profile workload 合同已冻结：DeepSeek-V4-Pro revision 为
`b5968e9190ef611bbf34a7229255be88a0e937c1`；FlashMLA 基线为
`9241ae3ef9bac614dd25e45e507e089f888280e0`，依次应用 dual-LSE 来源 revision
`13d173ac48abd8ec88a4e742bcaaa59c5ccf4ece`（SHA-256
`6957dbde516c73066c5911108761325edc1bdcd8f62e15dc0a84f4f290118d4b`）和 Pro
H128/prefix-1024 来源 revision
`b7643bd54521f563b839b98289b5cd048c062ba2`（SHA-256
`c534e13ff432ac1c694cb24981826c11be26a2d9743d7175ddb05f887279461f`）。cuDNN backend 使用
`9.24.0.43`，frontend 使用 `v1.26.0@35fd7b0d0e1d4952b904c79341c5e84e3af0a328`。
2026-08-06 用户明确撤销 cuDNN frontend host-sync/default-stream 本地补丁；frontend 必须是
官方未修改源码，镜像/artifact 记录 `local_patches=none`，不得施加
`docker/cudnn_frontend_dense_indexer_no_host_sync.patch` 或等价修改。受 stream 语义影响的
cuDNN 调用由 Magi-DSA caller 显式提交到非零 CUDA stream，进入 wrapper 前 hard
check 拒绝 stream 0，不修改 frontend 兜底。

正式 Pro-pair workload 固定为单机 8 张 B300（SM103）、
`world_size=rank_size=cp_size=8`、单条 128K causal sequence、
`cu_seqlens=[0, 131072]`、BF16 和固定 seed/input/image。每个 captured step 合并两张互相独立的
post-projection graph：representative layer 2 CSA 与 layer 3 HCA；forward 为
`CSA → HCA`，backward 为 `HCA → CSA`，连续 5 轮。它们只代表两种 DSA-core
workload，不声称执行了具有层间 activation 依赖的 61 层 Transformer。“官方最新”
只在环境冻结时解析一次，后续只使用 artifact 记录的 exact revision/version。

以下 2026-07-20 至 2026-07-22 的 Base balanced-only forward+backward 合同只保留为历史
correctness/NVTX 审计细节，不是当前 Pro release profile 入口。它当时被批准为不替代
release profile 的 balanced-only forward+backward 诊断，
并于 2026-07-22 最终收窄为仿照 Magi-MSA 的 DSA-core capture：workload、seed、输入和依赖保持相同；
sequential 与 balanced 各自在 capture/prewarm 前执行一次 `TOKEN_LAYOUT(x)` 和一次 deterministic profile
projection，将得到的 `MagiDSAInput.x/qr/q/latent_kv` 分别 detach 为显式梯度 leaf，再各完成 3 次
DSA-core 前后向 prewarm。capture 外另以同一固定 seed 生成一份 source-order global BF16 random
`dout`，按 `1/global_output_elements` 缩放后再按各 plan 的 `local_query_global_rows` 排成本地 `dout`；
KL 的上游梯度固定为 FP32 scalar one。capture 内仅
运行 balanced 的 5 个 `forward → backward(dout, dkl) → parameter_gradient_allreduce` steps，不构造
scalar loss，也不得出现 `magi_dsa::loss` range。每次 `calc_dsa` 从同一组 fixed post-projection leaves
重新构建 DSA autograd graph，不能复用已 backward 的 graph；capture 不得包含
`TOKEN_LAYOUT.forward/backward` 或 `model_projection`，DSA forward/backward 必须分别严格只有 4 个
All2AllV。backward 使用 multi-output
`torch.autograd.backward((output, selected_kl), (local_dout, scalar_one))`；每步开始后、forward 前用
`set_to_none=True` 清空四个 DSA activation leaves、共享 sink 和 layer 参数梯度，不做梯度累积；逐 step
profile-control 记录须缓存到 capture synchronize 之后再写，不能在相邻 step NVTX 之间执行文件 I/O。
backward 后在模型侧逐项 all-reduce 共享 sink 和 `MagiDSALayer` 参数梯度，runtime 仍不得归约模型参数。
诊断不包含 optimizer；capture 停止后运行一次 sequential DSA-core 前后向 shadow，按 Q16 检查 Top-K
长度/结构，并按原冻结门槛检查 output/KL、canonicalized `dx/dqr/dq/dlatent_kv`、sink 和 layer 参数梯度；
该诊断不重复验证 projection backward、合并后的 `dlocal_x` 或 source-owner `dx` inverse route，后者继续
由 CP8 natural correctness 覆盖。2026-07-22 用户补充批准：backend-native exact-cutoff-tie 可产生不同 output；
canonical Top-K 集合不同的 Query 行只要求两侧 output 均有限并记录诊断，不参与跨 plan output
`5e-3` gate，canonical 集合相同的 Query 行仍执行原 gate。其余 LSE、KL 和梯度门槛不变。
backward/gradient-allreduce 的逐 rank 时间必须报告，但当前不设新的 5% 硬门槛。

### 当前 installed-wheel release 边界

Release 镜像必须从 clean revision 构建并安装该 revision 的 `magi_attention` wheel，而不是只依赖
source bind mount。Wheel 为已批准 DSA Python runtime 范围，构建时固定
`versioningit==3.3.0` 且跳过与 DSA 路径无关的 legacy Magi CUDA extensions；镜像内版本必须包含
完整 40 字符 source revision，CP8 release 验证必须证明 8 ranks 都从 Python 安装目录
（`site-packages` 或 Debian/NVIDIA 等价的 `dist-packages`）导入该 wheel，而不是从 source mount 导入。

当前 CSA 合同记录在权威设计第 1、5、6、10 节：CP1/CP2/CP8 分别承担 reference、最低多卡回归和
release 验收；一个 rank 可以持有多个不连续 sample-relative fragments，但所有 local Query 必须在一次
grouped logical Indexer score/top-k invocation 中处理，不允许 runtime bucket 或 per-fragment launch。
Top-K 从生成开始就是 Query-owner local order，selected-KL 和 main attention 也在该 rank；所有 K route
统一使用 Magi `all2all_v`、unique receive、local prefix pack 和反向 CSR。范围是 BF16 causal
packed/varlen training prefill 和同路径 `no_grad`；不包含 decode/cache、FP8/FP4、TP、CUDA Graph、
selected-KV dynamic routing 或完整 attention recompute。KL 的 selected-score recompute 属于固定训练
算法，不能误删。HCA 数学仍由官方论文/固定 reference revision 约束。

Natural-only 修订冻结：正式 `MagiDSAInput` 不暴露 `forced_topk_ids` 或 `forced_topk_length`；CP1、CP2
和 CP8 correctness 都执行 natural forward/backward。sequential、`structural_balanced` 与 CP1 reference 的 Top-K
按下述 Q16 backend-native 合同验证；Indexer LSE、output、KL、全部输入梯度和 Compressor/Indexer
参数梯度继续与 CP1 reference 对齐；exact-cutoff-tie 导致 canonical Top-K 集合不同时，output 按下述
逐行豁免合同处理。Top-K backend-native 语义只用小型 Indexer 单测覆盖，不运行完整
forced backward release gate。JIT、kernel
编译和 prewarm 必须像 MSA 一样在 60 秒 post-compile watchdog 之外完成；CP2 的有效门槛是预热后的
natural sequential/`structural_balanced` 完整前后向在 60 秒内完成。历史 forced timeout 只保留诊断记录，不再是
当前 goal blocker。

Q14/Q15 的历史裁决与三次诊断继续作为审计证据，但其 ID 对齐和 exact-tie secondary-key 要求已由
Q16 显式取代。Q16 冻结为：production 直接采用固定 cuDNN `indexer_top_k_wrapper` 返回的 local IDs 与
位序，只执行 sample-local → canonical-global offset、effective length 和 `-1` padding；不得扫描完整
score 行重选 cutoff tie、二次 Top-K、稳定排序，或以 epsilon、量化、score bit 改写和重采样制造次序。
sequential、`structural_balanced` 与 independent reference 可在 exact cutoff tie 上选择不同有效集合和位序；这些
差异只记录诊断，不作为失败条件。`topk_length` 必须 exact，每行有效前缀必须非负且唯一，padding 必须
为负；直接 backend 单测还必须证明 production 输出逐元素等于 cuDNN 原始 IDs 加 global offset，并对
对应 raw score 验证：严格高于 cutoff 的候选全部保留、不选择低于 cutoff 的候选、cutoff 同分候选可任取
补足、非同分 score 保持降序而 exact tie 位序不限制。固定 cuDNN ABI、raw FP32 score 与所有
LSE/KL/梯度门槛不变。2026-07-22 用户批准 exact-cutoff-tie output 例外：canonical Top-K 集合不同的
Query 行允许产生不同 output，不参与 sequential/`structural_balanced` 或 reference 间的 output 数值 gate，但两侧
output 必须各自全部有限并记录 global/local row 与原始 max-abs；canonical 集合相同的行继续使用
`atol=rtol=5e-3`。该例外不得扩展到 raw score、LSE、KL、loss、梯度或非 tie 行。

设计获批后的验收必须同时满足：

- Magi-DSA 必须完成 Pro ratio 4/128 的 natural forward/backward；CP1 reference 与 kernel、
  CP2/CP8 distributed sequential 与 `structural_balanced` 按 Q16 验证 natural Top-K length/结构、
  Indexer raw score/LSE、
  canonical-set 相同行的 output、KL、全部输入梯度及 Compressor/Indexer 参数梯度，并覆盖 packed/ragged、边界 fragment、空 rank
  和多 seed；`ratio=0` 只验证兼容边界，不计入 Pro release profile。
- 参数归属测试必须证明 61 层的 Compressor/Indexer/attention 参数 owner 相互独立，
  optimizer/state dict 只从 `MagiDSALayer` 取得可训练参数，
  runtime 为无参数对象；默认路径的 Indexer 梯度能回到 `x/qr`，显式 model-side detach 只截断
  指定 trunk 梯度而不被 runtime 改写。
- 每个 Query 在前置 layout 中精确覆盖一次；CSA/HCA 必须具有相同的
  `query_layout_hash/query_token_counts/fragments`；`TOKEN_LAYOUT(x)` 及其 `dx` inverse route 可逆，
  CSA 四条、HCA 三条 typed route 收发计数对称，反向梯度归约无遗漏或重复。除 Q16
  exact-tie output 逐行例外外，`structural_balanced` 与 sequential 数值等价。
- KI/KV 双 map 必须允许 bank 行序不同；FlashMLA 固定 1024+128 indices，同次返回的完整
  `sparse_lse` 只用于 direct sparse backward，compressed-prefix LSE 只用于 KL teacher；
  predictor 必须在 selected Top-K 内归一化并调用 sparse Indexer backward。三种 LSE 状态与
  CP1 reference 成对验证，并证明未选中 candidate 梯度为 0。
- sequential/`structural_balanced`/reference 的 effective Top-K length 必须 exact，且每个结果分别满足有效 global ID
  非负唯一和 padding 为负；ID 集合与位序允许因 backend-native exact tie 而不同。小型 Indexer 单测必须
  证明 production 原样保留 cuDNN IDs/位序及 global offset，并验证严格高于 cutoff 的候选全部保留、
  不选择低于 cutoff 的候选、cutoff 同分候选可任取补足、exact tie 不施加 secondary key，且
  near-equal score 不被当作 tie。Indexer raw score、LSE 和 canonical-set 相同行的 output 使用
  `atol=rtol=5e-3`，canonical-set 不同行的 output 只要求两侧有限并记录诊断；KL 和
  常规输入/参数梯度使用
  `atol=rtol=2e-2`；只有 BF16 distributed shared/compressed-KV gradient 可使用 `atol=1e-8`、
  `rtol=5e-2`、mismatch ratio `<=0.08`。任何放宽都必须以 reference 误差证据单独 review。
- 复用 execution handle 的 warm path 不运行 solver、object collective、host layout 构建或
  device map materialization；调用私有 buffer/event 支持 two-inflight、reentrant backward、
  retain-graph 和 gradient accumulation。
- backward 必须分别验证 `detach_indexer_trunk=False/True`；模型侧必须对 replicated
  Compressor/Indexer/sink partial gradients 做 CP AllReduce，runtime 不拥有参数且不执行该归约。

- 正式 Pro-pair profile 只运行 `balanced + structural_balanced`，每 step 依次提交
  `CSA forward → HCA forward → HCA backward → CSA backward`，两个 mode 的边界使用 CUDA
  event 建立 happens-before，GPU 上不允许跨 mode kernel overlap。capture 内不得出现
  `TOKEN_LAYOUT`、model projection、scalar loss、optimizer 或跨 step 梯度累积；两张图的
  backward completion stream join 之后，才能发起一次 model-side parameter/sink AllReduce。
- 每个 rank/step 的 CSA 为 `4F+4B`、HCA 为 `3F+3B`，Pro-pair 总账必须精确为
  `7F+7B`；14 条 route-direction 均须只出现一次，顺序、mode 归属和收发计数是硬门槛。
  所有 route 与另一 mode compute 的 overlap 必须为 0；mode-local 的正 overlap 时长/比例
  完整报告但不设最低门槛，HCA 四条 dependency-bound route 必须记录依赖原因。
- 每个 rank/step 的 Indexer score 和 Top-K 各有一次 logical invocation；两者的
  `relative_rank_range=(max-min)/mean` 在全部 5 steps 均必须 `<=0.05`。CSA/HCA
  FlashMLA forward、cuDNN sparse backward、Compressor 与其他大 kernel 同样逐 step 报告
  `min/max/range/relative_rank_range`，但当前不设 5% 硬门槛。
- 正式汇总必须报告 route pack/reverse CSR、KV-bank assembly、CSA compression gather、
  grouped Indexer K pack 和 Indexer score/Top-K NVTX 窗口中的 CUDA D2D 次数、字节与时间；
  NCCL、`DsaRowCopy`、`DsaRowCsrReduce` 不得充当 compute。同一 aggregate report 的
  CUPTI/runtime correlation 归因覆盖率必须为 1.0，unattributed kernel 必须为 0。
- clean revision 的 installed-wheel 镜像必须复现 CP1、CP2、CP8 natural correctness 和
  5-step Pro-pair profile；四类 artifact 必须对应同一 image ID，源码、依赖、命令、
  dirty status、镜像 digest 与 SHA-256 manifest 可追溯。

### 历史 Base profile 审计细节（不属于当前 Pro release 验收）

以下 2026-07-20 至 2026-08-03 的 sequential/balanced、forward-backward 与 W/CSA/HCA
attention-suite 约束只用于解释历史 Base artifact，不得作为当前 profile 入口、
route 总账或 release 通过条件。

- 冻结 CSA 128K workload 上分别完成 sequential 与 balanced 5-step profile：warmup/JIT 在
  capture 外，每个 `(plan, step, rank)` 对 `indexer_score` 和 `indexer_topk` 各记录一次逻辑调用
  GPU 时间。逐 step 并排报告两种 plan 的 rank `min/max/range` 与
  `relative_rank_range=(max-min)/mean`；balanced 的两项 relative range 在全部 5 steps 中都必须
  `<=0.05`，不能用五步平均掩盖离群。`rank_range_ms` 必须报告但不设硬门槛；sequential 只作为
  Magi-DSA 内部 baseline，不承担 5% 门槛。
- 正式 capture 顺序为 balanced → sequential。NVTX 结构与 Magi-MSA balanced 参考对齐：外层
  `$Magi_DSA/capture_five_training_steps`，step 为 `<plan>/rank_<rank>/training_step_<step>`，输出层为
  `<plan>/rank_<rank>/O`，Indexer 父层为 `Magi_DSA/indexer`；冻结的
  `magi_dsa::indexer_score`/`magi_dsa::indexer_topk` logical ranges 嵌套其中且继续承担计时合同。
  模型、packing、route 和 backend 的细粒度诊断统一使用
  `magi_dsa::module::<branch>::<operation>`；这些子 range 只做无同步的 launch 归因，不能替代或缩小
  外层 score/top-k 正式计时范围。
  Indexer 的两个 cuDNN frontend wrapper 另加醒目的
  `magi_dsa::CUDNN_CALL::{indexer_score,indexer_topk}` 子 range；它们只标记 wrapper launch 边界，核心
  kernel GPU 时间仍通过 CUDA runtime correlation ID 关联 CUPTI kernel，不使用 CPU NVTX wall time。
  forward+backward 诊断另以
  `magi_dsa::CUDNN_CALL::{sparse_attention_backward,indexer_backward}` 标记两个 cuDNN backward wrapper。
- 独立 forward+backward 诊断使用 `$Magi_DSA/capture_five_forward_backward_steps`，每 step 内必须各有
  一个 `magi_dsa::{forward,backward,parameter_gradient_allreduce}` range，且 `magi_dsa::loss` 必须为零；原有
  `magi_dsa::{indexer_score,indexer_topk}` 仍各出现一次。所有范围只用 CUDA runtime correlation ID
  关联 CUPTI kernel，不得插入逐 step synchronize。该诊断只采集 balanced，sequential shadow 在
  capture 停止后运行；aggregate trace 中任何 `TOKEN_LAYOUT` 或
  `magi_dsa::module::model_projection::*` NVTX/kernel 均使结果无效，逐 rank/step 的 forward 与 backward
  必须分别恰有 4 个 `ncclDevKernel_SendRecv`；四条 reverse 的每个 rank/step 都必须与非 route GPU
  compute 有正时间线交集，并逐条报告通信时长、重叠时长和覆盖比例，当前不设最低覆盖比例。
  相邻 step range 之间不得执行同步、梯度清理或 artifact
  文件写入；唯一的 capture synchronize 位于五步提交完成后并处于外层 range 内，与 Magi-MSA 对齐。
- W+CSA+HCA 合并诊断使用 `$Magi_DSA/capture_five_attention_suite_steps`。它不是具有跨层 activation
  依赖的 Transformer 三层，而是在同一个 step range 内合并三张独立 Attention autograd graph：
  forward 固定为 `W(ratio=0) → CSA(ratio=4) → HCA(ratio=128)`，backward 固定反序
  `HCA → CSA → W`。三者分别在 capture 前准备固定 post-projection leaves 和同一 source-order global
  `dout` 的 plan-local view；capture 内无 TOKEN_LAYOUT、model projection、scalar loss、optimizer 或
  跨 step 梯度累积。每 step 只在开始时清空三组 activation/model 梯度，三次 backward 完成后统一执行
  一次模型侧 parameter/sink bucket AllReduce。三种模式各用独立 execution stream，但
  `W F → CSA F`、`CSA F → HCA F`、`HCA B → CSA B`、`CSA B → W B` 四个跨 mode 边界必须
  用 CUDA event 建立 happens-before；caller-thread 调用/NVTX 顺序保持不变，GPU 上不允许跨 mode
  kernel overlap。三次 backward completion event 必须先在
  `magi_dsa::module::attention_suite::stream_overlap::gradient_join` 汇合，随后才能发起 gradient
  AllReduce。模式级
  `magi_dsa::attention_suite::<w|csa|hca>::<forward|backward>` 必须各出现一次；逐 rank/step 的
  SendRecv 数量必须分别为 W `1F+1B`、CSA `4F+4B`、HCA `3F+3B`，外层总账必须为 `8F+8B`。
  由于 PyTorch autograd backward kernel 由 worker thread 发射，正式 caller-thread 模式 range
  之外还必须分别出现
  `magi_dsa::module::attention::<w|csa|hca>::sparse_attention::cudnn_backward` 和
  `magi_dsa::phase::collective_all2all_v::attention::<mode>::<route>.backward` launch-thread
  range；不能只靠同进程时间窗猜测三种模式的 backend 与 route 归属。汇总必须验证三种模式各自
  恰有一次 FlashMLA forward core 和同一组 cuDNN sparse-attention backward core，并单独报告
  ratio-specific kernel 差集。W/HCA 三输出 FlashMLA 变体的完整资源签名必须一致；CSA dual-LSE
  变体允许只在 registers/thread 上专用化，但 kernel 名称、block 和 shared memory 必须与 W/HCA
  一致。cuDNN backward 四个 core kernel 的完整资源签名必须在三种模式间一致。
  汇总还必须对全部 16 个 W/CSA/HCA route-direction 硬校验单次
  `ncclDevKernel_SendRecv` 和固定 runtime launch 顺序，在
  `ATTENTION_SUITE_COMMUNICATION_OVERLAP.json` 逐条报告完整 NVTX path、通信时长、相同 mode 且
  相同 direction 的非 route GPU overlap 与比例，并在 `STEP_KERNEL_SPANS_ATTENTION_SUITE.json`
  报告真实首末 kernel span；NCCL、`DsaRowCopy` 和 `DsaRowCsrReduce` 都不能充当计算，任一 route
  与其他 mode compute 的交集必须为 0，六个 mode phase 的实际 CUPTI kernel span 也必须严格串行。
  overlap 比例只报告、不设硬门槛；无法自身遮挡的 route 必须记录依赖原因。W 因无独立
  Compressor/Indexer，forward/backward `WINDOW_KV` 均不可借用 CSA/HCA 计算；HCA 只有 Window
  forward/reverse 具有 Main Compressor/Compressor-backward 的 mode-local 遮挡条件。
  Indexer score/top-k 只允许出现在 CSA forward 且各一次。CSA 在 capture 停止后另跑 sequential
  shadow 并沿用 Q16/tie-aware output 与梯度门槛；W/HCA 验证 ratio-specific gradient schema 和 finite。
- clean revision 的 installed wheel/镜像能够复现 Magi-DSA smoke 与 5-step profile；源码、
  submodule、依赖、镜像 digest、命令和原始结果均可追溯。

## 事实来源与参考路径

CSA `ratio=4` 的架构、tensor/index ABI、collective 对账、backward 和 kernel→op 事实入口是
`docs/design/magi_dsa_v4_design.md`。旧 `docs/design/magi_dsa_v4.md` 仅用于追溯历史 Q 裁决、测试和
profile artifacts，不能覆盖当前架构。

Magi-MSA 的权威参考位于 `/home/scratch.wewen_gpu/Magi-MSA`：

- API/runtime：`magi_attention/api/msa_attn_interface.py`、
  `magi_attention/msa_runtime_mgr.py`、`magi_attention/functional/dist_msa.py`。
- 通信/packing：`magi_attention/functional/msa_comm.py`、
  `magi_attention/functional/msa_packing.py`。
- 元数据/solver：`magi_attention/meta/collection/msa_meta.py`、
  `magi_attention/meta/solver/msa_dispatch.py`、
  `magi_attention/meta/solver/msa_solver.py`。
- kernel/backend：`magi_attention/kernel/cutedsl/msa_pack.py`、
  `magi_attention/functional/msa_backend.py`、固定 gitlink
  `magi_attention/functional/MM-Sparse-Attention/`。
- 测试/文档：`tests/test_msa/`、`docs/source/user_guide/magi_api.md`。

DeepSeek-V4 数学事实来源为官方 `DeepSeek_V4.pdf` 与
`deepseek-ai/DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py`。
前者冻结 CSA/HCA、window、sink 和训练 CP 语义；后者冻结 Pro tensor 尺寸/公式、完整块/尾块与显式
index sentinel 行为。二者冲突或未覆盖 packed training 细节时必须停下提问，由本仓库 CP1 reference
在 review 后补成可执行合同；不得把 inference cache 行为直接扩展成 training ABI。

DSA correctness 只使用本仓库的 pure-PyTorch/CP1 reference、sequential plan 和
`structural_balanced` plan
形成自洽闭环，不设置外部实现比较项。其他 dirty worktree 或 `experimental/` 代码不是事实
来源，不得直接复制为 production 实现。

## Magi、QDSL 与 FA 编码约束

- Magi Python 使用 4 空格、类型标注、冻结 dataclass 表示静态 plan；通过 Black、isort
  （Black profile）、Ruff、Flake8（127 列）和 MyPy。C++/CUDA 服从仓库
  `.clang-format` 与 clang-format-20。新增源码保留版权头；源码标识符、注释和 docstring
  使用英文，中文仅用于指定的评审文档。
- Q2 已冻结：不引入其他 QDSL 仓库或 style guide；“QDSL”仅指所选 Magi 基线使用的 Quack
  helpers + NVIDIA CuTe Python DSL。Q12 于 2026-07-19 经用户独立批准后，版本精确锁定为
  `nvidia-cutlass-dsl==4.5.0`、`nvidia-cutlass-dsl-libs-base==4.5.0`、
  `nvidia-cutlass-dsl-libs-cu13==4.5.0`、`apache-tvm-ffi==0.1.8.post0`、
  `quack-kernels==0.4.1`；后续升级必须作为独立依赖变更评审，不得在 DSA 实现中顺手升级。
- CuTe kernel 以 `magi_attention/kernel/cutedsl/msa_pack.py` 为直接风格参考：host wrapper 显式
  校验 device、dtype、shape、contiguity/alignment，device map 使用 contiguous CUDA int32；
  `@cute.jit` 负责 specialization/launch，`@cute.kernel` 实现 device body，并透传 caller stream。
  使用 symbolic shape、fake tensor 和 TVM-FFI 编译；DSA specialization 按
  `(operation, major, minor, dtype, feature width)` 有界缓存。warm path 禁止 host sync、plan/map
  构建和逐 token Python 循环；输出或 workspace 的分配/复用策略必须显式并计入 profile。copy
  明确 vector width，归约使用 FP32 accumulator，仅在输出边界转换 dtype。
- FA/kernel 接入采用薄 wrapper 和固定版本 ABI；forward/backward、LSE、sink、top-k sentinel、
  saved-state 与 backend-native selection 语义必须成对测试。不得为了通过测试修改外部 kernel 数学，
  不得静默 fallback。当前证据支持 FlashMLA sparse forward 与 cuDNN DSA backward/Indexer；
  production 直接调用这两类 backend，不通过 Megatron fused autograd wrapper；Megatron 只提供
  Indexer、selected teacher/clipping、缩放和调度参考，KL 归一化域由当前权威设计覆盖。
  现有 FFA IndexAttn 的 head-dim 限制不能被当作 DSA-512 backend。修改扩展签名时同步 `.pyi`、
  public API、设计文档和 reference 测试。

## 测试、Profile、镜像与结果规范

实现获批后，测试放在 `tests/dsa_v4/`，benchmark driver 放在
`benchmarks/dsa_v4/`，测试/profile/镜像脚本分别放在 `scripts/test/`、
`scripts/profile/`、`scripts/image/`，镜像入口为 `docker/Dockerfile.dsa-v4`。当前 Pro 命令为：

```bash
pytest -q tests/dsa_v4/test_plan.py tests/dsa_v4/test_comm.py
pytest -q tests/dsa_v4/test_reference.py tests/dsa_v4/test_kernels.py \
  tests/dsa_v4/test_backend.py tests/dsa_v4/test_cp1_kernel.py \
  tests/dsa_v4/test_profile_contract.py
timeout --signal=TERM --kill-after=5s 60s bash scripts/test/run_multigpu.sh --world-size 2 --case smoke
bash scripts/test/run_cp1.sh --image <release-image>
bash scripts/test/run_multigpu.sh --world-size 2 --case csa-natural-backward \
  --image <release-image> --installed-wheel
bash scripts/test/run_multigpu.sh --world-size 8 --case cp8-natural-backward \
  --image <release-image> --installed-wheel
bash scripts/profile/run_5step.sh --world-size 8 --cp-size 8 \
  --case dsv4-pro-128k --plans balanced --steps 5 --step-mode pro-pair \
  --layout-policy structural-balanced --skip-smoke --image <release-image>
bash scripts/image/run_release.sh --revision <40-char-clean-commit>
pre-commit run --all-files && make format-check
```

全新的正式 release 由 `run_release.sh` 串行编排，不得用不同镜像拼接证据：

1. 从给定 clean 40 字符 revision 构建 installed-wheel 镜像；
2. 在该镜像内完成 CuTe AOT/prewarm；
3. 依次产生 CP1、installed-wheel CP2 和 installed-wheel CP8 natural correctness artifact；
4. 使用同一 image ID 运行 8×B300、5-step `dsv4-pro-128k` Pro-pair profile；
5. 校验 CP1/CP2/CP8/profile 的 image ID、来源 revision、dirty status 与 phase audit 后执行
   `finalize_release.py`。

实现时若命令改变，必须先更新本文件和设计文档。多卡 correctness 的 setup/JIT/prewarm 可在有界
30 分钟 build/JIT deadline 内完成，但必须排除在 60 秒 post-compile execution watchdog 外；若运行
smoke 则仍使用 60 秒总时限。Pro-pair 正式 profile 使用 `--skip-smoke`，但不得跳过同一
release 镜像中的 CP1/CP2/CP8 验证。5-step profile 可显式延长总时限，但必须保留
60 秒“无进展”watchdog。
怀疑 collective 死锁时立即终止整个 worker 进程组，不复用可能损坏的 process group；以
`NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL` 和最小 case 重跑定位。

结果使用全新 UTC run id，分别写入 `artifacts/correctness/<run-id>/`、
`artifacts/profile/<run-id>/` 和
`artifacts/release/<run-id>/`，不得覆盖旧目录。每个目录至少保存 exact command、stdout/stderr、
逐 rank raw records、summary、source/submodule revisions、dirty status、依赖与硬件信息、seed、
镜像 ID/digest 和 SHA-256 manifest。Pro-pair profile 在 `balanced/` 子目录保存一份包含
8 workers 的 aggregate `.nsys-rep` 及其 SQLite export，并在 run 根目录保存
`SUMMARY_PRO_PAIR.json`、`rank_ranges_pro_pair.json`、
`MAJOR_KERNEL_BALANCE_PRO_PAIR.json`、`PRO_PAIR_ROUTE_TIMINGS.json`、
`MODE_SERIALIZATION_PRO_PAIR.json`、`SUPPORT_OVERHEAD_PRO_PAIR.json`、
`INDEXER_D2D_PRO_PAIR.json` 和 `REPORT_PRO_PAIR.md`。每 rank 必须从同一 aggregate report
导出 NVTX/CUDA runtime/CUPTI kernel 原始记录和每个 phase
的逻辑调用次数，不得通过重复执行正式 5 steps 伪造 per-rank 文件。缺 plan/rank/step、重复记录、
非 finite 时间或调用次数不为 1 均使结果无效。正式 phase GPU 时间统一使用 logical NVTX 时间窗内、
同 worker 进程的 CUDA runtime launch 通过 correlation ID 关联的 CUPTI kernel duration 之和，不使用
CPU NVTX wall time；autograd worker thread 的 launch 必须以 `process_temporal` 计入其正式 phase，
不能只查询外层 NVTX 主线程。
该 Pro-pair run 还必须保存 `nsys_kernel_attribution.jsonl`、
`rank<rank>_nsys_kernel_attribution.jsonl` 和 `NSYS_ATTRIBUTION.json`；归因 join 必须同时限制
kernel `globalPid` 与 runtime `globalTid` 的 worker 进程，防止并行 rank 的同号 correlation ID 串接。
正式摘要只有在 `attribution_coverage=1.0` 且 `unattributed_kernel_count=0` 时才能通过；
autograd 跨线程回退到同进程正式逻辑 phase 时必须显式标记 `process_temporal`。

## 设计与实现同步

任何 public schema、模型语义、fragment/route、collective 顺序、saved-state、kernel ABI、
overlap 或验收门槛变化，都必须在同一 commit 更新 `docs/design/magi_dsa_v4_design.md`、相关测试和
用户文档。若变更影响历史验收/Profile 证据，再同步更新 `docs/design/magi_dsa_v4.md` 的历史说明，
不得让旧文档重新成为架构事实来源。实现行为与当前权威设计冲突时不得只修代码；先暂停并请求评审。
