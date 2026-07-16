# Magi DSA 发布验收

## 稳定公共接口

Magi DSA 从稳定的 `magi_attention.api` 命名空间发布。新代码只应导入以下
公共符号：

```python
from magi_attention.api import (
    DsaOverlapConfig,
    DsaPackedMeta,
    MagiDSAConfig,
    MagiDSAInput,
    MagiDSARuntimeMgr,
    MagiDSAYarnConfig,
    calc_dsa,
)
```

生产 wheel 与公共 API 不得包含或解析出 `magi_attention.experimental`；其他历史
worktree 或备份中的 prototype 在该安装边界之外，本次发布无需物理删除。
`MagiDSAV4Config` 和 `MagiDSAV4YarnConfig` 继续作为兼容别名保留，但新的序列化
配置使用 `MagiDSAConfig` 和 `MagiDSAYarnConfig`。

一个 `MagiDSARuntimeMgr` 持有一种 layer form 的参数和不可变
`compress_ratio`：ratio 0 是纯 window，ratio 4 是 CSA，ratio 128 是 HCA。
张量合同、分布式初始化、checkpoint 和完整示例见
[Magi DSA 用户指南](../../docs/source/user_guide/magi_dsa.md)。

## 可运行的 CP=1 示例

下面的示例通过安装后的稳定 API 在一张 CUDA 卡上执行 forward 和 backward。
CP=1 是同输入数值/梯度 oracle，不是分布式发布出口。

```python
import torch

from magi_attention.api import (
    DsaPackedMeta,
    MagiDSAConfig,
    MagiDSAInput,
    MagiDSARuntimeMgr,
    calc_dsa,
)

torch.cuda.set_device(0)
device = torch.device("cuda", 0)
config = MagiDSAConfig(
    compress_ratio=4,
    hidden_size=8,
    q_lora_rank=8,
    softmax_scale=512**-0.5,
    backend="reference",
)
runtime = MagiDSARuntimeMgr(config).to(device).train()


def trainable(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)


tokens = 4
dsa_input = MagiDSAInput(
    x=trainable(tokens, config.hidden_size),
    qr=trainable(tokens, config.q_lora_rank),
    q=trainable(tokens, config.num_heads, config.kv_dim),
    latent_kv=trainable(tokens, config.kv_dim),
    sink=trainable(config.num_heads, dtype=torch.float32),
    packed_meta=DsaPackedMeta(torch.tensor([0, tokens], dtype=torch.int32)),
)
output, indexer_kl = calc_dsa(dsa_input, runtime)
(output.float().square().mean() + indexer_kl).backward()

assert output.shape == (tokens, 64, 512)
assert output.dtype == torch.bfloat16
assert indexer_kl.dtype == torch.float32
assert dsa_input.q.grad is not None
assert dsa_input.latent_kv.grad is not None
assert dsa_input.sink.grad is not None
```

发布矩阵还会从源码 checkout 之外的 `/tmp` 执行
`tests/test_dsa/installed_package_smoke.py --cuda`。该检查覆盖 ratio 0、4、128，
并拒绝解析到归档仓库而非已安装 wheel 的 import。

## 不可变生产镜像

生产镜像由 `agents/release/magi-dsa-v4/build_image.sh` 使用
`agents/release/magi-dsa-v4/Dockerfile.production` 构建。builder 先把请求的 Git
revision 解析并归一化为完整 40 位 commit，只导出该 commit 的已提交 Git 对象，
把冻结的 submodule 对象展开到临时 context，构建并安装带版本的 wheel，构建
native 扩展，并把哈希记录到 `/opt/magi-dsa-build-manifest.json`。虽然 builder
可以归一化其他无歧义 Git revision，发布命令仍显式传入完整 commit；调用方的
dirty worktree 永远不会被复制到镜像中。

Dockerfile 通过 registry digest
`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`
固定 B300 基础镜像，并安装 `nvidia-nvshmem-cu13==3.6.5`。镜像 tag 是可变别名；
报告和命令必须使用内容寻址的 `sha256:...` image ID。

用一条命令构建干净 revision 并执行有序最终矩阵。`REPO` 必须指向已初始化的
disposable integration worktree，且递归 submodule 已 checkout 到 release image
source revision 记录的 gitlink；不能指向开发 worktree：

```bash
REPO="<ABSOLUTE_INITIALIZED_DISPOSABLE_INTEGRATION_WORKTREE>"
RELEASE_IMAGE_SOURCE_REVISION="<FULL_40_CHAR_RELEASE_IMAGE_SOURCE_GIT_SHA>"
RELEASE_OUTPUT="<ABSOLUTE_RELEASE_ARTIFACT_DIRECTORY>"

"${REPO}/agents/release/magi-dsa-v4/build_image.sh" \
  --revision "${RELEASE_IMAGE_SOURCE_REVISION}" \
  --tag "magi-dsa-v4-b300:${RELEASE_IMAGE_SOURCE_REVISION:0:12}" \
  --output-dir "${RELEASE_OUTPUT}" \
  --run-final-matrix

B300_IMAGE=$(<"${RELEASE_OUTPUT}/image_id.txt")
test "${B300_IMAGE}" = \
  "$(docker image inspect --format '{{.Id}}' "${B300_IMAGE}")"
```

验收只 bind-mount 冻结 pack、performance/profile 输出和 final-result artifact
目录。不得挂载源码目录或 Megatron checkout，也不允许 editable install、
镜像内开发 checkout 或退回未安装 package。`RELEASE_OUTPUT` 下的
`installed-wheel-smoke.log`、`image_inspect.json`、构建日志、矩阵结果和 SHA256
manifest 是权威证据。

## CP=8 native 与零 skip 出口

分布式出口严格定义为一台 8× NVIDIA B300 SXM6 AC 节点（SM103）上、固定使用
GPU 0–7 的一个 `world_size=cp_size=8` 进程组。CP=2 只保留兼容回归，不能替代
该出口。

`final_matrix.json` 和 `run_final_matrix.py` 在不可变镜像内依次强制执行以下
case：

- 在仓库外通过已安装公共 API 执行 forward/backward smoke。
- CP=1 API、solver、packing kernel 和固定 Megatron oracle 覆盖。
- CP=8 native transport 和完整 forward/backward，覆盖所有 ratio 和 policy、
  梯度、2×2 overlap 矩阵、并发/reentrant 调用、空路线/空 rank，以及协调可恢复
  故障。
- 隔离的单 rank 故障，随后由 launcher 清理，并在全新进程组中执行 native
  健康检查。

time-boxed sampled closure 不会抽样缩减这个矩阵：重建后的 final image 仍须完整
执行上述四个有序 case。Run `b300-cp8-final-f8ad2e5d-20260713T110210Z` 已将四项
全部通过并封存，因此 sampled closure 下的步骤 9 已完成；这不能把 sampled 性能
证据升级为正式步骤 8 验收或 formal release qualification。

首次步骤 9 run `b300-cp8-final-f88ca22d-20260713T103532Z` 使用旧 f88 candidate
镜像 `sha256:4bd9b55dd1303043256fd6accf79a5362f630b85448a5425958c42a2467d0bea`。
Public smoke 在 `8.679 s` PASS，CP1 `93/93` 在 `99.657 s` PASS，CP8 在
`7.728 s` 得到 `4 passed, 14 failed` 后按序停止。根因是 pytest importlib mode
下从 `/tmp` 启动的 spawn 子进程报 `ModuleNotFoundError: tests`，不是 DSA
correctness、OOM 或 watchdog 故障。单项 diagnostic 改为
`--import-mode=append` 后，native 8-rank test 为 `1 passed`（`36.61 s`），且
`magi_attention` 仍从 installed-wheel `site-packages` 导入。失败 run 与 outer
manifest 文件 SHA256 分别为
`81f8f21a524f4d2dd050ca591358234b133f222de8e03523c75b083caf44757d` 和
`0818fe00f144b63d7718e8d38f3a775a91da4acef824419b639491b3a59107b3`。
修复 commits 为 DEV `e8f50e97896fec4eee5646988e6797e9c5a8b76c`、integration
`f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`；重建后的 final Step9 candidate
是下文已封存的 f8 镜像。旧 f88 镜像仅作为 sampled timing 与失败矩阵前身保留；
runtime、kernel、solver 行为未改，因此不重跑 sampled timing。

CP=8 case 设置 `MAGI_ATTENTION_NATIVE_GRPCOLL=1`、
`MAGI_ATTENTION_HIERARCHICAL_COMM=0` 和 `CUDA_DEVICE_MAX_CONNECTIONS=8`，并
unset `NVSHMEM_SYMMETRIC_SIZE`。冻结的 `GrpCollConfig` 使用 `num_sms=20`、
`num_nvl_bytes=1073741824`（每 buffer 1 GiB）和 `num_rdma_bytes=0`。计时前必须
dry-allocate 四类 typed payload buffer 与 replicated-gradient buffer，并实例化真实
`GrpCollIntraHandle`。DSA 通信使用 native GroupCast/GroupReduce；不接受直接 A2AV
或 hierarchical fallback。缺少 native 扩展、NVSHMEM、固定依赖、测试数据或任何
必需测试都属于硬失败。每个 JUnit case 必须是零 skip、零 error、零 failure；
timeout、OOM、非有限张量、native fallback、测量区内 JIT miss、
revision/artifact 不匹配都会使本轮无效。

A2AV 只在步骤 1–7 的历史开发矩阵中作为兼容覆盖执行。步骤 8/9 正式出口只
接受实际 `GrpCollIntraHandle` native 路径；A2AV 和 hierarchical fallback 都是硬失败。

ratio 4 的每个 owner-local dispatch fragment 直接作为一个 query tile，并在
Indexer projection、top-k、forward KL 和 backward KL recompute 中一致复用。
tile 不跨 sample 或 ownership 边界，也不再细分成 128-row launches，从而消除小
GEMM 和 Indexer kernel 的 launch storm。不同 dispatch policy 可能改变 BF16 GEMM
的 row grouping，因此跨 policy/CP parity 按冻结数值容差验收，不要求 bitwise 等价。

cuDNN top-k launch 对短 sample 仍保持固定配置宽度 K=512，超过实际
compressed-key 数的槽位填 `-1`。不得把 launch K 特化为 473 等奇数短宽度，
否则会违反冻结 CuTe 内核两元素 vector-store 分支的整除要求。

ratio-4 kernel 路径把这 512 个 compressed slots 放在 128 个 window slots
之前，并以 `indexer_topk=512` 调用 FlashMLA。返回的 `lse_indexer` 供整 rank
合批的 cuDNN KL-target recompute 使用，不再 gather selected KV。API 测试用
8-valid/504-invalid 与 512-valid 两种 prefix 对拍显式 selected-QK target，并覆盖
全 invalid rows；CP8 full-backward 测试同时验证 sparse backward 使用同一顺序。

## 性能验收

步骤 8 由 `agents/benchmarks/magi-dsa-v4-balance/` 驱动；其中的
[README](../../agents/benchmarks/magi-dsa-v4-balance/README.md) 是可执行的
workload 与门槛规范。seed-42、20-pack 文件只在 calibration 前生成一次并封存文件
哈希。production 构建流程先生成一个 clean、不可变的 installed-wheel calibration
镜像；该镜像用于生成 cost 系数，但不是最终 measure/release 镜像。系数经 review、
冻结和提交后，同一构建流程再生成一个独立的最终不可变镜像，供正式
measure/profile 和发布矩阵使用。正式测量复用 calibration 的已验哈希文件，绝不再次
调用 `run.sh packs`。wrapper 只挂载 pack、performance 和 profile artifact 目录：

```bash
REPO="<ABSOLUTE_INITIALIZED_DISPOSABLE_INTEGRATION_WORKTREE>"
DEVELOPMENT_REPO="<ABSOLUTE_DEVELOPMENT_WORKTREE>"
export PACKS_DIR="<ABSOLUTE_FROZEN_PACK_ARTIFACT_DIRECTORY>"
export PACKS_FILE="${PACKS_DIR}/packs-seed42.json"

# 只在 calibration 前生成并封存一次 pack 文件。
export MAGI_DSA_IMAGE="<IMMUTABLE_CALIBRATION_IMAGE_ID>"
export RUN_ID="calibration-run-id"
export PERF_DIR="<ABSOLUTE_PERFORMANCE_ARTIFACT_ROOT>/${RUN_ID}"
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" packs
(
  cd "${PACKS_DIR}"
  sha256sum "${PACKS_FILE##*/}" > "${PACKS_FILE##*/}.sha256"
)
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" calibrate

python3 "${DEVELOPMENT_REPO}/agents/benchmarks/magi-dsa-v4-balance/freeze_calibration.py" \
  --input "${PERF_DIR}/calibration.json"
# 在 DEVELOPMENT_REPO review 并提交 dsa_calibration.py，把该原子 commit
# cherry-pick 到 REPO，再从 clean integration HEAD 构建 B300_IMAGE。

# 正式 measure/profile 复用完全相同的已封存文件。
(
  cd "${PACKS_DIR}"
  sha256sum --check "${PACKS_FILE##*/}.sha256"
)
export MAGI_DSA_IMAGE="${B300_IMAGE}"
export RUN_ID="final-performance-run-id"
export PERF_DIR="<ABSOLUTE_PERFORMANCE_ARTIFACT_ROOT>/${RUN_ID}"
export PROFILE_DIR="<ABSOLUTE_PROFILE_ARTIFACT_ROOT>/${RUN_ID}"

"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" measure
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" profile
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" validate
```

validator 从原始逐 pack、逐 rank、逐 iteration 记录重新计算所有 timing 和
load-balance 统计；在计时前验证 candidate 正确性；从 Nsight CUDA interval 证明
两个 overlap window；并把 performance/profile 两个 root 封装进同一个 SHA256
manifest。任何 skip 或不完整运行都不能通过。

### Time-boxed sampled closure（非正式）

2026-07-13，用户选择在约三小时内以 sampled 口径先收尾。这不会删除或放宽上面的
正式 20-pack 合同。已停止的正式 calibration attempt
`b300-cp8-calibration-66d69258-20260713T054451Z` 使用 revision
`66d69258fc9df1ee96f4c91df5d8225173b7fae2` 和 image
`sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`。
其进度为 `57/120`：ratio 0 的 40 个单元全部完成，随后完成
`r4-sequential-00` 的 packs 0–16。它只是 partial 诊断证据，不能 resume 为正式
calibration，也不构成正式验收。容器在 `2026-07-13T07:45:12Z` 收到外部
`SIGTERM`；它不是 OOM，这次 stopped run 也不是 fail-stop 用例。仅保留其
progress 和正确性记录作为诊断证据。Run 目录
`agents/perf/magi-dsa-v4-balance/b300-cp8-calibration-66d69258-20260713T054451Z`
内 `progress.json` 的文件 SHA256 为
`6f5520099807c290452b1a68da479e95240dbb1c9dfb19e54cc8908c025c528f`；
`agents/perf/magi-dsa-v4-release/calibration-logs` 下对应
`magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.docker-events.jsonl`
和 `magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.termination.json`
的文件 SHA256 分别为
`308245df4cdf2cf8db521960fb6ab5f1a3b9cab7e4823909ca20f3eb4ee81ce7` 和
`927974b501330e4339a9e2793b5236697eb92d46294dc8733d0e1bd148c50870`。

非正式 calibration 在六个冻结 calibration cases 上只运行 packs `[0, 1]`：
ratio 0/4/128 各自的 `sequential:00` 和 `balanced:00`。这 12 个单元生成的
`sample_calibration.json` target 为 `sampled-b300-sm103`，并记录
`formal=false`、`scope=sampled_non_formal`；freeze 命令必须显式传入
`--allow-sampled-non-formal`。

该 calibration 已完成为 run
`b300-cp8-sample-calibration-20260713T083426Z`，revision
`66d69258fc9df1ee96f4c91df5d8225173b7fae2`，image
`sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`，
进度 `12/12`。它包含 correctness `12`、plans `96`、raw timing `960` 条记录。
冻结系数 ID 为
`df625f1805024f6c7c78e2bdcb07476c8be94347c3e762a80006a87aea0e882d`。
证据目录
`agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-calibration-20260713T083426Z`
中，`sample_calibration.json` SHA256 为
`5d42c3ea327cf866f7faed7db3334d5d8c7cf56e44d1b60624c219d608e4c0e3`，
`artifact_manifest.sha256` 文件 SHA256 为
`82ccd78736f7972b4160e442350ea8fee63a591bfe2597ed46d427c74e742883`。
ratio 4 的 bounded least-squares 拟合为 `R²=0.923198`、`RMSE=83.118 ms`；它只含
两个独立 packs，且 window/overlap 特征严格共线
（`window_rows = 31.75 * overlap_rows`，design rank `6/7`），因此两项系数不可分别
识别。这只是 sampled diagnostic 拟合，不能证明正式模型拟合质量。

重建 final image 后，sampled 性能目标是以下 9 个 case-pack 单元：

- Pack 0：ratio 4 的 `sequential:00` 加 balanced `00/01/10/11`，以及 ratio 128
  的 `sequential:00` 加 `balanced:11`。
- Pack 16：ratio 4 的 `sequential:00` 加 `balanced:11`。

这些单元仍执行一次 compile、两次 warm-up、十次计时，以及逐元素 forward/全部梯度
对拍。summary 必须保持 `formal_acceptance=false` 和
`formal_gates_evaluated=false`。各 run 的 pack 集合必须分开：stopped attempt 的
ratio 0 覆盖全部 20 packs，ratio-4 sequential 覆盖 packs 0–16；sampled
calibration 选 packs 0 和 1；final-image sampled measure 选 packs 0 和 16。合并这些
记录不会形成新的 sampled pack 集合，也不能作为 20/20 candidate imbalance 或
speed gate 证据。精确 `sample` 命令合同见 benchmark README。

Sampled/diagnostic profile 未运行，也未产生。因此正式 ratio-4 Nsight overlap
门槛仍未评估。

## Runtime 与故障边界

CP=1 runtime 支持完整 module deepcopy 和序列化。分布式 `ProcessGroup` 是外部、
不可 pickle 的状态，因此 CP runtime 通过 `state_dict` 保存：先在目标 group 上
构建新 runtime，再加载 state。

正常退出和协调可恢复异常会 drain 已发起 work。rank-local compute/kernel 或
collective-wait 故障会 best-effort 请求 process-group abort。已经阻塞在
CUDA/NCCL/NVSHMEM stream wait 中的 peer 可能无法返回，因此不可恢复分布式故障
采用 fail-stop：外部 supervisor 必须在 60 秒 watchdog 内终止全部八个 worker，
再创建全新进程组；不承诺同进程恢复。

Magi DSA 本轮发布提供 packed attention primitive 及其 CP runtime。接入真实
DeepSeek V4 模型、映射模型权重、选择逐层 layer form，以及执行端到端模型训练或
推理，仍不属于本轮发布范围。

## 最终 artifact 记录

以下值均仅从封存后的 build、benchmark 和 matrix artifact 回填：

- 正式步骤 8 状态：**未完成**；冻结的 20-pack 门槛保持不变，sampled closure
  没有对它们进行正式评估。
- 已停止的正式 calibration attempt：revision
  `66d69258fc9df1ee96f4c91df5d8225173b7fae2`，image
  `sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`，
  run `b300-cp8-calibration-66d69258-20260713T054451Z`，进度 `57/120`；
  `2026-07-13T07:45:12Z` 被外部 `SIGTERM` 终止，非 OOM，也不是 fail-stop
  用例。
- Sampled calibration：run `b300-cp8-sample-calibration-20260713T083426Z`，
  revision `66d69258fc9df1ee96f4c91df5d8225173b7fae2`，image
  `sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`，
  进度 `12/12`，记录数 correctness/plans/raw timing `12/96/960`，target
  `sampled-b300-sm103`，scope `sampled_non_formal`，`formal=false`，冻结系数 ID
  `df625f1805024f6c7c78e2bdcb07476c8be94347c3e762a80006a87aea0e882d`。
  `sample_calibration.json` SHA256 为
  `5d42c3ea327cf866f7faed7db3334d5d8c7cf56e44d1b60624c219d608e4c0e3`；
  `artifact_manifest.sha256` 文件 SHA256 为
  `82ccd78736f7972b4160e442350ea8fee63a591bfe2597ed46d427c74e742883`。
- Predecessor f88 sampled timing 已完成全部 `9/9` 个选定 correctness 单元，产生
  `72` 条 plan 和 `720` 条 raw timing 记录。三个源 run
  `b300-cp8-sample-measure-a-f88ca22d-20260713T093515Z`、
  `b300-cp8-sample-measure-b-f88ca22d-20260713T101533Z` 和
  `b300-cp8-sample-measure-c-f88ca22d-20260713T103247Z` 的
  `artifact_manifest.sha256` 文件 SHA256 分别为
  `8ec095b475228012a2029a49066595692ae7c431fb12fe0a33981c952a7948f5`、
  `a508503306049cc5460640b6babdb1ef69be2a942ea3bd210ba5c37e7013741e` 和
  `7423cfc56ad63ee8f2f9f6c4716eff7f9f6164ca7890e9e8a76f8c19dcf6d64c`。
  聚合证据目录为
  `agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-measure-f88ca22d-20260713T103500Z-aggregate`；
  `sampled_measure_diagnostic.json` SHA256 为
  `47fb9e1ab4e852be90046c03438ac29ef8c29c552193cfe68e9ff36e262cded5`，
  `artifact_manifest.sha256` 文件 SHA256 为
  `955e17348d4ab00c11a582be2c2dfb4b7bcb6abe79bd9d8eadb6d89dfc93a62b`。
  Artifact 记录 `formal=false`、`formal_acceptance=false`、
  `formal_gates_evaluated=false` 和 `all_gates_pass=null`。ratio 4 两包诊断的
  baseline/candidate E2E 为 `28602.9863/29245.3057 ms`（`+2.245637%`，
  candidate 更慢），Indexer 为 `474.7959/493.6558 ms`（`+3.972199%`，
  candidate 更慢）；两个 sampled candidate pack 的 E2E rank imbalance 均不超过
  5%。ratio 128 pack 0 的 E2E 为 `321.7919/320.7076 ms`（`-0.336954%`，
  candidate 更快），candidate rank imbalance 为 `0.10824%`。因此
  `sampled_performance_observation_pass=false`；这只是 sampled observation，
  不是正式 gate 结果。
- Sampled/diagnostic profile：未运行且未产生；正式 ratio-4 Nsight overlap
  门槛仍未评估。
- Sampled-closure 最终矩阵结果：run
  `b300-cp8-final-f8ad2e5d-20260713T110210Z`，`status=passed`、`error=null`；
  四个有序 case 全部通过。这只完成 sampled closure 下的步骤 9；正式步骤 8 仍未
  完成，sampled ratio-4 性能方向仍为失败。
- Sampled-closeout candidate 源 revision：
  `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`
- 已安装 candidate package 版本：`1.1.1+dsa.f8ad2e5d4f8b`
- 不可变 sampled candidate image ID：
  `sha256:12075f8738456a417cac72f39a328378bb1bb1ee313c5fd7c34c92c6d0bf5c37`
- Candidate 构建证据：
  `agents/perf/magi-dsa-v4-release/sampled-final-f8ad2e5d-20260713T104133Z`；
  镜像内 `/opt/magi-dsa-build-manifest.json` SHA256 为
  `c07f71865a118786e43a54d3c980b14ffaa2db0484ec5e8094ed671ae57e7519`；
  `artifact_manifest.sha256`、`image_id.txt`、`package_version.txt`、
  `installed-wheel-smoke.log`、`image_inspect.json` 和 `docker-build.log`
  文件 SHA256 依次为
  `fa69dad9035c326cea00f6dc8ec7865ec3ea9f8bc568cf6a54a64fc7dfd13eff`、
  `27f7f3362179b16d6faf7541b55db5f86bce8506cab0af8164293b2a29b55985`、
  `965323aa062b7753d197ccf7b0920ad546316c6da7182a5c59224f8976768cce`、
  `eebb12ff3aa2518e74cbc3b9dad5cb04e0b3a2d75877228758eaf6bc534951c5`、
  `7417ec701e0b6f592be477df8269dbbf58ce9c7204c0c6ea8a4d778177f1c674`、
  `15d5a5be9c7b496b930fc02b236496e840f3fed01f1dfbfc6e4efc3635fc02a8`。
  Installed-wheel smoke 已通过。runtime、kernel、solver 行为相对 f88 未变，
  已封存的 9-unit sampled timing 仍归属于旧 f88 镜像且不重跑；f88 同时作为失败
  矩阵前身保留。新镜像只是 sampled-closeout candidate，不是 formal
  production/release image，也不满足 PLAN 步骤 8。
- Calibration artifacts/status：正式 calibration 未完成；run
  `b300-cp8-calibration-66d69258-20260713T054451Z` 停止于 `57/120`。
  抽样、非正式 calibration 证据为上文记录的 run
  `b300-cp8-sample-calibration-20260713T083426Z`。
- 正式性能 run 与门槛摘要：未产生 `validation.json`；正式 20-pack
  validation 未运行。
- Profile run 与 overlap 摘要：因 profile 未运行，未产生 profile/overlap
  JSON；正式 ratio-4 overlap 门槛仍未评估。
- 最终矩阵 cases：installed public-API smoke `8.629054 s`；CP1
  `100.860094 s`，JUnit `93/93`，failure/error/skip 均为 0；CP8 native
  `550.507443 s`，JUnit `18/18`，failure/error/skip 均为 0；隔离 fail-stop
  `47.943240 s`。
- 隔离 fail-stop：通过。fault rank 3 以 rc `86` 退出；7 个 peers 均已发起 native
  GroupCast，全部 workers 在 `3.064738 s` 内回收，`survivors=[]`。Fresh health 的
  8/8 ranks 均 rc `0`，handle 均为 `GrpCollIntraHandle`，配置为 NVL
  `1073741824`、RDMA `0`、`num_rdma_ranks=1`、`num_sms=20`。
- 封存证据根：
  `agents/perf/magi-dsa-v4-release/final-matrix-f8ad2e5d`。Inner/outer
  `artifact_manifest.sha256` 文件 SHA256 分别为
  `fdc9a4cd0a96a5f69d554d6b89c898f6c83de1a26e81ab53de7e22285ae20c11` 和
  `b045d7fb03db5e03f47c16c43a5dfde862e64545791bc3ad58c68eda47977d11`；
  `final_result.json`、`fault_result.json` 和 inspect 文件 SHA256 分别为
  `bb83d46d7f1e0b2a2db4591b4cbe8f45dffafaf3b5adc8d5dc5a7859e5cf1687`、
  `77cfa8a1a937a8281bc590cd3c6b7fdf2098e68796adfe52c2133ef3c24783e1` 和
  `6420dcac04ca064a2bbe4d9ec239e52a9e550fbd154de2d0247273807ff59449`。
  只读 verify 对 80 个 artifacts 全部通过。Container inspect 记录 exit `0`、
  OOM false、image 精确匹配、仅一个 artifact mount、无 source bind，且未设置
  `NVSHMEM_SYMMETRIC_SIZE`。
- 矩阵测试的 integration HEAD：
  `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`；对应 DEV commit：
  `e8f50e97896fec4eee5646988e6797e9c5a8b76c`。
- 开发/integration 文档编辑前的 committed tree 一致：均为
  `38846f45dc5c10b30d0679f1f6decc252e78c1b2`。最终文档 commit IDs 在交付回复中
  记录，以避免 commit 自引用。
- 冻结的递归 gitlinks：
  - `magi_attention/csrc/cutlass`：
    `81a43e6d92cdd8c20d22392f9579604ed5f710a1`
  - `magi_attention/functional/flash-attention`：
    `ee1d15159cda6f3f97bfab9e487da146a8254970`
  - `magi_attention/functional/flash-attention/csrc/composable_kernel`：
    `e8709c24f403173ad21a2da907d1347957e324fb`
  - `magi_attention/functional/flash-attention/csrc/cutlass`：
    `b1d6e2c9b334dfa811e4183dfbd02419249e4b52`
