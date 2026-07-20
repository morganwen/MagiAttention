# Repository Guidelines

## 已批准设计与硬性边界

Magi-DSA v4 总体设计、接口方向和验收矩阵已于 2026-07-19 获用户批准，并于同日批准 natural-only
release gate 修订；2026-07-20 又批准 Q14 的独立 reduction backend Top-K 对齐合同与 Q15 的
deterministic exact-tie 合同。权威设计文档是
`docs/design/magi_dsa_v4.md`。后续实现必须遵守该冻结基线，且只有在用户明确发起实现任务后才能
开始新增或移植 runtime、kernel、测试、构建系统或性能优化代码。遇到需求、tensor schema、kernel
ABI、CSA/HCA 数学语义或通信原语不清楚时，先在设计文档登记问题，先与 Magi-MSA 的设计和实现
比较，再向用户提问；不得用“合理默认值”补齐合同。任何偏离已批准设计的修改必须先暂停并复审。

## 明确目标与验收条件

目标是提供 DeepSeek v4 hybrid attention 的训练级实现：`ratio=0` 为 window-only，
`ratio=4` 为 CSA，`ratio=128` 为 HCA。方案复用 MSA 的 sample-relative fragment、cold
plan、device map、unique-row routing 和反向 CSR reduction，但保持主 attention 的 token
owner/output 布局稳定；CSA 只临时平衡 Indexer，HCA 不经过 Indexer。

参数与执行职责已经冻结：`MagiDSALayer` 是模型侧模块，拥有全部 Compressor/Indexer 可训练
参数；`MagiDSARuntimeMgr` 不拥有任何可训练参数，只保存静态 plan、通信 metadata 与执行状态。
runtime 不得隐式 `detach`、注册梯度 hook 或归约模型参数来改变 autograd 语义。Indexer 分支默认
保留到 `x/qr` 的梯度链；若模型需要截断，必须在 `MagiDSALayer`/调用侧显式表达。

Q3 的模型、环境与 profile workload 合同已冻结：模型配置使用官方
`DeepSeek-V4-Flash-Base` revision
`8855555deef230a27a21a8d6f294b7b7497759b6`；FlashMLA 使用官方 `main` revision
`9241ae3ef9bac614dd25e45e507e089f888280e0`；cuDNN backend 使用 `9.24.0`（CUDA 13 wheel build
`9.24.0.43`），cuDNN frontend 使用 `v1.26.0` tag commit
`35fd7b0d0e1d4952b904c79341c5e84e3af0a328`。该组合于 2026-07-19 由用户批准的 Q11 方案 1
冻结。目标机器为单机 8 张 B300（SM103），`world_size=rank_size=cp_size=8`。
Profile workload 固定为单条 128K causal sequence，`cu_seqlens = [0, 131072]`、BF16、CSA
`ratio=4`、`world_size=rank_size=cp_size=8`。sequential 与 balanced 必须使用完全相同且显式记录的
seed、输入、配置和软件镜像，各运行 5 个 captured forward steps；运行中不得重采样或改变 layout。
“官方最新”只在环境冻结时解析一次，随后必须记录 exact revision/version，禁止 profile 时追踪
浮动分支。
Release 镜像必须从 clean revision 构建并安装该 revision 的 `magi_attention` wheel，而不是只依赖
source bind mount。Wheel 为已批准 DSA Python runtime 范围，构建时固定
`versioningit==3.3.0` 且跳过与 DSA 路径无关的 legacy Magi CUDA extensions；镜像内版本必须包含
完整 40 字符 source revision，CP8 release 验证必须证明 8 ranks 都从 Python 安装目录
（`site-packages` 或 Debian/NVIDIA 等价的 `dist-packages`）导入该 wheel，而不是从 source mount 导入。

Q4–Q10 的冻结合同记录在设计文档第 1、10 节：production 使用 contiguous owner-local CP shard，
CP1/CP2/CP8 分别承担 reference、最低多卡回归和 release 验收；每 rank/step 只有一次 grouped logical
Indexer score/top-k invocation，不允许 runtime bucket 或 fragment loop；selection 在 balanced worker，
top-k/length/Indexer LSE 返回 owner，selected-KL 在 owner；所有 route 统一使用 Magi `all2all_v` 和
本地 CSR。HCA 数学由官方论文/固定 reference revision 冻结，物理 prefix ABI 由固定 backend wrapper
冻结。范围是 BF16 causal packed/varlen training prefill 和同路径 `no_grad`；不包含 decode/cache、
FP8/FP4、TP、CUDA Graph、selected-KV routing 或 attention recompute。上述合同已随总体设计获得
批准；实现不得静默扩大范围或替换通信、KL placement 与 grouped Indexer 语义。

Natural-only 修订冻结：正式 `MagiDSAInput` 不暴露 `forced_topk_ids` 或 `forced_topk_length`；CP1、CP2
和 CP8 correctness 都执行 natural forward/backward。sequential、balanced 与 CP1 reference 的 Top-K
按下述 Q14 两级 comparator 对齐；Indexer LSE、output、KL、全部输入梯度和 Compressor/Indexer 参数梯度
继续与 CP1 reference 对齐。cutoff tie policy 只用小型 Indexer 单测覆盖，不运行完整 forced backward release gate。JIT、kernel
编译和 prewarm 必须像 MSA 一样在 60 秒 post-compile watchdog 之外完成；CP2 的有效门槛是预热后的
natural sequential/balanced 完整前后向在 60 秒内完成。历史 forced timeout 只保留诊断记录，不再是
当前 goal blocker。

Q15 已由用户批准方案 1 并解除裁决 blocker：production Top-K 的主键仍为原始 FP32 Indexer score
降序；仅当 raw score 逐元素 exact 相等时，使用 canonical global compressed-block ID 升序作为
secondary key，较小 ID 优先。该规则必须覆盖 cutoff tie 的候选集合重选和有效 Top-K 内部位序，不能
只排序 cuDNN 已返回的 K 项。不得用 epsilon 扰动、量化或重采样制造次序；near-equal-but-non-tied
score 的顺序不变。固定 cuDNN score/top-k wrapper ABI、raw score、Indexer LSE、output/KL/梯度门槛
和 Q14 跨 reduction-backend comparator 均保持不变。历史三次诊断继续作为审计证据。

Q14 Top-K 对齐合同冻结：相同 backend、输入和 seed 下的 sequential 与 balanced 必须保持 ordered
Top-K tensor exact。production 与独立 pure-PyTorch reference 比较 canonical global IDs：逐行按
`topk_length` 过滤 padding 后，effective length 必须 exact，有效 ID 必须各自唯一，按 global ID 排序后的
有效 tensor 必须 exact；不同 reduction backend 之间不要求 Top-K 内部位序 exact。Indexer raw score
继续执行 Q9 预先冻结的 `atol=rtol=5e-3` 数值检查；Indexer LSE、output、KL、全部梯度、cutoff tie 和
natural backward 门槛均不变。Q14 原本冻结的非 tie 数学与 backend ABI 不变；Q15 仅显式修订 exact
tie 的 production selection/order 行为为 global ID 升序 secondary key。

设计获批后的验收必须同时满足：

- Magi-DSA 自身能够完成 ratio 0/4/128 的 natural forward/backward；CP1 reference 与 kernel、
  CP2/CP8 distributed sequential 与 balanced 按 Q14 对齐 natural top-k IDs/length、Indexer raw score/LSE、
  output、KL、全部输入梯度及 Compressor/Indexer 参数梯度，并覆盖 packed/ragged、边界 fragment、空 rank
  和多 seed。
- 参数归属测试必须证明 optimizer/state dict 只从 `MagiDSALayer` 取得 Compressor/Indexer 参数，
  runtime 为无参数对象；默认路径的 Indexer 梯度能回到 `x/qr`，显式 model-side detach 只截断
  指定 trunk 梯度而不被 runtime 改写。
- 每个 Query 精确处理一次，dispatch/restore 是可逆映射，typed route 收发计数对称，反向
  梯度归约无遗漏或重复；balanced 与 sequential 数值等价。
- 同 backend 的 sequential/balanced natural top-k ordered tensor 与 length 必须 exact；production 对
  independent reference 的 canonical global valid-ID tensor、effective length 和逐行唯一性必须 exact，
  不要求不同 reduction backend 的内部位序 exact。cutoff tie 的小型 Indexer 单测必须证明：严格高于
  cutoff 的候选全部保留、cutoff 同分候选按 canonical global ID 升序补足、有效结果按 score 降序且
  exact tie 按 global ID 升序排列，并且 near-equal score 不被当作 tie。Indexer raw score 和其他
  forward 值使用 `atol=rtol=5e-3`，KL 和
  常规输入/参数梯度使用
  `atol=rtol=2e-2`；只有 BF16 distributed shared/compressed-KV gradient 可使用 `atol=1e-8`、
  `rtol=5e-2`、mismatch ratio `<=0.08`。任何放宽都必须以 reference 误差证据单独 review。
- 复用 execution handle 的 warm path 不运行 solver、object collective、host layout 构建或
  device map materialization；调用私有 buffer/event 支持 two-inflight、reentrant backward、
  retain-graph 和 gradient accumulation。
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
- clean revision 的 installed wheel/镜像能够复现 Magi-DSA smoke 与 5-step profile；源码、
  submodule、依赖、镜像 digest、命令和原始结果均可追溯。

## 事实来源与参考路径

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
`deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py`。
前者冻结 CSA/HCA、window、sink 和训练 CP 语义；后者冻结 Flash tensor 公式、完整块/尾块与显式
index sentinel 行为。二者冲突或未覆盖 packed training 细节时必须停下提问，由本仓库 CP1 reference
在 review 后补成可执行合同；不得把 inference cache 行为直接扩展成 training ABI。

DSA correctness 只使用本仓库的 pure-PyTorch/CP1 reference、sequential plan 和 balanced plan
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
  saved-state 与 deterministic 语义必须成对测试。不得为了通过测试修改外部 kernel 数学，
  不得静默 fallback。当前证据支持 FlashMLA sparse forward 与 cuDNN DSA backward/Indexer；
  现有 FFA IndexAttn 的 head-dim 限制不能被当作 DSA-512 backend。修改扩展签名时同步 `.pyi`、
  public API、设计文档和 reference 测试。

## 测试、Profile、镜像与结果规范

实现获批后，测试放在 `tests/dsa_v4/`，benchmark driver 放在
`benchmarks/dsa_v4/`，测试/profile/镜像脚本分别放在 `scripts/test/`、
`scripts/profile/`、`scripts/image/`，镜像入口为 `docker/Dockerfile.dsa-v4`。拟冻结命令为：

```bash
pytest -q tests/dsa_v4/test_plan.py tests/dsa_v4/test_comm.py
pytest -q tests/dsa_v4/test_reference.py tests/dsa_v4/test_kernels.py \
  tests/dsa_v4/test_backend.py tests/dsa_v4/test_cp1_kernel.py \
  tests/dsa_v4/test_profile_contract.py
timeout --signal=TERM --kill-after=5s 60s bash scripts/test/run_multigpu.sh --world-size 2 --case smoke
bash scripts/test/run_multigpu.sh --world-size 2 --case csa-natural-backward
bash scripts/test/run_multigpu.sh --world-size 8 --case cp8-natural-backward
bash scripts/test/run_multigpu.sh --world-size 8 --case cp8-natural-backward \
  --image <release-image> --installed-wheel
bash scripts/profile/run_5step.sh --world-size 8 --cp-size 8 --case dsv4-flash-128k \
  --plans sequential,balanced --steps 5 --skip-smoke
bash scripts/image/build.sh --revision <40-char-commit>
bash scripts/image/run_release.sh --revision <40-char-commit> \
  --profile-artifact artifacts/profile/<formal-run-id> \
  --cp1-artifact artifacts/correctness/<cp1-run-id> \
  --cp2-artifact artifacts/correctness/<cp2-run-id>
pre-commit run --all-files && make format-check
```

实现时若命令改变，必须先更新本文件和设计文档。多卡 correctness 的 setup/JIT/prewarm 可在有界
30 分钟 build/JIT deadline 内完成，但必须排除在 60 秒 post-compile execution watchdog 外；若运行
smoke 则仍使用 60 秒总时限。2026-07-20 用户明确要求本次正式 profile 不重复 smoke，沿用已通过的
CP8 correctness 与 `artifacts/profile/20260720T033324Z-cp8-short-profile-smoke/` 证据。5-step profile
可显式延长总时限，但必须保留 60 秒“无进展”watchdog。
怀疑 collective 死锁时立即终止整个 worker 进程组，不复用可能损坏的 process group；以
`NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL` 和最小 case 重跑定位。

结果使用全新 UTC run id，分别写入 `artifacts/correctness/<run-id>/`、
`artifacts/profile/<run-id>/` 和
`artifacts/release/<run-id>/`，不得覆盖旧目录。每个目录至少保存 exact command、stdout/stderr、
逐 rank raw records、summary、source/submodule revisions、dirty status、依赖与硬件信息、seed、
镜像 ID/digest 和 SHA-256 manifest。5-step profile 在 `sequential/` 与 `balanced/` 子目录分别
保存一份包含 8 workers 的 aggregate `.nsys-rep` 及其 SQLite export，并在 run 根目录保存合并的
`profile_5step.jsonl`、`rank_ranges.json`、`REPORT.md`；报告须逐 step 并排比较两种 plan。每 rank
还必须从该 plan 的同一 aggregate report 导出 NVTX/CUDA runtime/CUPTI kernel 原始记录和每个 phase
的逻辑调用次数，不得通过重复执行正式 5 steps 伪造 per-rank 文件。缺 plan/rank/step、重复记录、
非 finite 时间或调用次数不为 1 均使结果无效。正式 phase GPU 时间统一使用 logical NVTX 内 CUDA
runtime launch 通过 correlation ID 关联的 CUPTI kernel duration 之和，不使用 CPU NVTX wall time。

## 设计与实现同步

任何 public schema、模型语义、fragment/route、collective 顺序、saved-state、kernel ABI、
overlap 或验收门槛变化，都必须在同一 commit 更新 `docs/design/magi_dsa_v4.md`、相关测试和
用户文档。实现行为与设计冲突时不得只修代码；先暂停并请求评审。
