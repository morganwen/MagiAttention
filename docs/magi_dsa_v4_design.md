# Magi_DSA V4 设计文档

本文与仓库根目录 `PLAN.md` 共同冻结 Magi_DSA V4 的实现口径；发生冲突时以 `PLAN.md` 的“冻结合同”为准。步骤 1–7 已在 8× NVIDIA B300 SXM6 AC 节点（SM103）上的单个 `world_size=cp=8` 进程组完成复验，固定使用 GPU 0–7。CP=1 只作为同权重、同全局输入的数值/梯度 oracle，不是分布式出口；CP=2 只保留兼容回归，也不能替代 CP=8 出口。此前把 8 卡拆成四个 CP=2 pair 的结果已废弃。

## 可复现基线

- MagiAttention 基线：`529fb0a4e273b3557a56d8afd60b74da46688095`。
- DSA 计算原型：`98e043cafebbdcbce6835e26849d96624136ef70`。
- Megatron dsv4 数值参考：`c6449f0b23be397449f21c0967c5fc90785e55ea`。
- DeepSeek 官方 HF 配置：`deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`；使用该 revision 的 `config.json` 和 `inference/config.json`。
- FlashMLA nv_dev：`b7643bd54521f563b839b98289b5cd048c062ba2`。
- NVIDIA cudnn-frontend：`f00538322e9d3d439fe8c5f3144644e58ee66823`（安装包 `nvidia-cudnn-frontend==1.27.0`）。
- fast-hadamard-transform：`e7706faf8d1c3b9f241e36860640ad1dac644ede`。
- CUTLASS 子仓：`81a43e6d92cdd8c20d22392f9579604ed5f710a1`；FA4 子仓：`ee1d15159cda6f3f97bfab9e487da146a8254970`；`nvidia-cutlass-dsl==4.5.2`。
- 步骤 1–7 的 B300 开发镜像来自主 checkout 中的历史配方 `/home/scratch.wewen_gpu/MagiAttention/agents/docker/magi-dsa-b300-step5/Dockerfile`（目录名保留了历史步骤号）。它直接固定 `nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`（NGC build `337426143`，build ref `d557151f4c7ddca284cb5e8d5ce78cee4d80f7e5`）；它不是发布镜像的权威构建入口。
- 该配方把 fast-hadamard-transform 构建到 compute capability 10.3，并使用 `FLASH_MLA_DISABLE_SM90=1` 只构建 FlashMLA SM100-family 对象；B300 上由 SM103 路径运行。本轮步骤 1–7 测试 tag 为 `magi-dsa-b300-step7:dev`，image id 为 `sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da`；tag 可变，报告以 image id 为准。
- 上述开发配方固定 FlashMLA、cudnn-frontend、fast-hadamard-transform 和 CUTLASS DSL revision，但仍不等于步骤 8/9 的 clean、不可变 native 镜像。本轮开发容器安装 `nvidia-nvshmem-cu13==3.6.5`，并挂载当前源码与扩展：`magi_attn_ext` SHA256 为 `0427073e7a0f16450bade229528638bfd1c9f610bf31d00f04d7f7899ffa0eaf`，`magi_attn_comm` SHA256 为 `ca98aa8439007b647c52b8aa4e18b6b0beb631593846445bd877528346c87030`。CP8 native 用例已断言实际 handle 为 `GrpCollIntraHandle`，不是 A2AV fallback。每个 `(group, buffer_name)` 固定 `GrpCollConfig.num_nvl_bytes=1073741824`（1 GiB）；四个 typed payload buffer 加一个 replicated-gradient buffer含 workspace 后，每 GPU 静态下限为 `5,536,482,080` B（约 5.15625 GiB），native full backward 已实际通过。单节点全 NVLink 且 `num_rdma_bytes=0` 时 `NVSHMEM_SYMMETRIC_SIZE` 保持 unset，报告为 N/A；步骤 8 仍须在最终 packs 上独立 dry-allocation。
- 步骤 8/9 的 production 权威入口现为 `agents/release/magi-dsa-v4/Dockerfile.production` 与同目录的 `build_image.sh`。已构建的 clean installed-wheel calibration image 来自 revision `66d69258fc9df1ee96f4c91df5d8225173b7fae2`，package `1.1.1+dsa.66d69258fc9d`，image ID `sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`；它不是最终 release image。仅含 matrix 修复的 clean revision `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9` 已重建当前 sampled-closeout candidate，package `1.1.1+dsa.f8ad2e5d4f8b`，不可变 image ID `sha256:12075f8738456a417cac72f39a328378bb1bb1ee313c5fd7c34c92c6d0bf5c37`，证据目录 `agents/perf/magi-dsa-v4-release/sampled-final-f8ad2e5d-20260713T104133Z`。嵌入的 `/opt/magi-dsa-build-manifest.json` SHA256 为 `c07f71865a118786e43a54d3c980b14ffaa2db0484ec5e8094ed671ae57e7519`；目录内 `artifact_manifest.sha256`、`image_id.txt`、`package_version.txt`、`installed-wheel-smoke.log`、`image_inspect.json` 和 `docker-build.log` 的文件 SHA256 依次为 `fa69dad9035c326cea00f6dc8ec7865ec3ea9f8bc568cf6a54a64fc7dfd13eff`、`27f7f3362179b16d6faf7541b55db5f86bce8506cab0af8164293b2a29b55985`、`965323aa062b7753d197ccf7b0920ad546316c6da7182a5c59224f8976768cce`、`eebb12ff3aa2518e74cbc3b9dad5cb04e0b3a2d75877228758eaf6bc534951c5`、`7417ec701e0b6f592be477df8269dbbf58ce9c7204c0c6ea8a4d778177f1c674`、`15d5a5be9c7b496b930fc02b236496e840f3fed01f1dfbfc6e4efc3635fc02a8`；installed-wheel smoke 已通过。修复未改变 runtime/kernel/solver，9-unit sampled timing 仍归属于旧 f88 镜像且不重跑；旧 f88 镜像同时保留为失败矩阵前身。新镜像只用于 sampled-closeout final matrix，不是 formal release image，也不满足正式步骤 8；其 run `b300-cp8-final-f8ad2e5d-20260713T110210Z` 的 public smoke、CP1、CP8 native 和隔离 fail-stop 四项已全部通过并封存，构成 sampled-closure 步骤 9 证据。
- 步骤 8 使用同一 production 构建流程生成两个不同的 clean、不可变 installed-wheel 镜像：先用 calibration image 生成系数，再在系数 review、冻结并提交后重建最终 measure/release image。calibration image 不是最终生产或发布镜像，不得用它回填最终 image 或 final-matrix 证据。

## 数值与模块边界

- compressor 输入 hidden `x`，不是投影后的 latent KV。内容投影和门控投影都从 `hidden_size` 出发。
- Indexer 的 Qi、Ki、Weights 由 runtime 内部从 `qr` 和 `x` 投影生成；Indexer 分支 detach 输入，KL 不向 trunk `x/qr` 回传。
- cuDNN sparse backward 原生返回 `d_sink`。sink 等价于 value 为零的附加 softmax 条目，但正式实现不通过补算代替原生梯度。
- Q/KV 最后 64 维已经由模型侧做 partial RoPE。压缩条按 logical block id 使用压缩 RoPE；官方配置使用 `compress_rope_theta=160000`、YaRN factor 16、original max positions 65536。
- 压缩器只处理完整块，sample 尾部不足 ratio 的 token 不生成压缩条。ratio=4 使用重叠 compressor，ratio=128 使用非重叠 compressor。
- 位置 `p`（sample 内 0-based）只能看到 `floor((p+1)/ratio)` 个完整压缩块。
- Indexer score 为 `sum_h weights[q,h] * relu(Qi[q,h] dot Ki[k])`。top-k 按 score 降序、同分按小 logical block id；合法项在前，其余填 `-1`。
- KL target 来自主 Q 与 compressed KV 的逐头注意力分布；predict 来自 Indexer。CP 下可微返回值为 `local_kl_sum / global_query_tokens`。

## 公共 API 与 tensor schema

应用代码的唯一正式导入面是 `magi_attention.api`：

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

公共定义位于 `magi_attention/api/dsa_attn_interface.py`；稳定实现包为
`magi_attention/dsa/`，runtime 和 autograd/通信编排分别位于
`magi_attention/dsa_runtime_mgr.py`、`magi_attention/functional/dist_dsa.py` 和
`magi_attention/functional/dsa_comm.py`。Production wheel/public API 不包含也不解析
`magi_attention.experimental`；其他历史 worktree 或备份中的 prototype 在该边界之外，
无需为本次发布物理删除。`forward_cp` / `forward_cp_packed` 同样不是公共接口。

- 新代码使用冻结 dataclass `MagiDSAConfig` 和 `MagiDSAYarnConfig`。`MagiDSAV4Config` 与 `MagiDSAV4YarnConfig` 仅是同一 class object 的兼容别名，供已有调用方平滑迁移；新建配置、文档和序列化记录必须使用无 `V4` 后缀的稳定名称。
- `DsaPackedMeta` 保存全局 packed sample 边界 `cu_seqlens`：1-D contiguous int32，首项为 0，单调不减，至少包含 `[0, T]`。CP=1 时末项必须等于本地 token 数；CP>1 时它仍描述全局 sample layout，本 rank 行按冻结 dispatch plan 的 fragment 顺序排列。
- `MagiDSAInput` 具名保存 `x`、`qr`、`q`、`latent_kv`、FP32 `sink` 和 `packed_meta`。行张量均为 packed THD：`x [T_local,hidden_size]`、`qr [T_local,q_lora_rank]`、`q [T_local,64,512]`、`latent_kv [T_local,512]`，`sink [64]`。前四者必须是 CUDA BF16，`sink` 必须是同一 CUDA device 上的 FP32。
- `calc_dsa(input, runtime_mgr)` 返回 owner-local `O [T_local,64,512]` 和可微 FP32 标量 `kl_loss`。CP 时每 rank 的返回值是 `local_kl_sum / global_query_tokens`；各 rank detached 值之和等于全局 KL，日志汇总不得改变反向图。
- 一个 `MagiDSARuntimeMgr` 只服务其 config 固定的一种 ratio，持有 compressor/Indexer 参数以及可缓存的不可变 host/device plan；每次调用的 tensor、work、event 和 native handle 不进入该缓存。`dispatch_policy` 只接受 `sequential` 或 `balanced`，`DsaOverlapConfig` 独立控制两个 overlap 开关。CP=1 是同权重 oracle；正式分布式出口是单个 CP=8 进程组。
- ratio=0 不构建 compressor/Indexer；ratio=128 只构建 compressor；ratio=4 同时构建 compressor 和 Indexer。

V1 固定配置为：Hq=64、Hkv=1、D=512、rope dim=64、window=128、Hidx=64、Didx=128、topk=512，主路径 BF16，sink FP32。`hidden_size`、`q_lora_rank` 和 `softmax_scale` 由模型配置提供。

## Kernel 接入边界

- sparse forward 复用 FlashMLA `flash_mla_sparse_fwd`。
- sparse backward/d_sink、Indexer forward/top-k、score recompute 和 Indexer backward 复用 cudnn-frontend `cudnn.deepseek_sparse_attention`。
- 统一 wrapper 位于稳定实现包 `magi_attention/dsa/kernels.py`；只做 ABI 接线，不重写、不 fork 外部 kernel。compressor、Indexer projection、reference oracle 和 RoPE 实现同样归入 `magi_attention/dsa/`。
- reference backend 只用于数值对拍；正式 kernel backend 的输入为 CUDA BF16，top-k 保持 device resident。
- 唯一计划内新增的计算辅助 kernel 是 `kernel/cutedsl/dsa_pack.py`，用于 packing、remap 与 FP32 CSR reduction。public frontend 和 device mapping schema 不绑定具体 minor arch，编译 cache key 包含设备的 `(major, minor)`、dtype、feature width 和 operation；因此 B300 使用独立 SM103 cache entry。kernel 仅使用 SM100-family 可用的通用 global copy、寄存器 FP32 累加和 128-bit vector copy。

ratio=4 的 kernel 路径采用与 Megatron Path C 相同的 fused pipeline
边界，而不是新增一个包含 radix top-k 的单体 CUDA kernel。Indexer 仍先用 raw BF16
weights 做冻结的 exact top-k/tie-break；随后 attention indices 按
`[compressed-prefix, window]` 排列，并以 `indexer_topk=512` 调用 FlashMLA，使同一次
sparse forward 返回 compressed prefix 的 `lse_indexer`。训练时 cuDNN
`sparse_attn_score_recompute_wrapper` 直接从该 LSE 生成 KL target，predict 则继续使用
BF16 缩放后的 sparse Indexer score recompute，以保持既有数值语义。一个 rank 上所有
owner fragments 的投影和 global logical block ids 合并为一次 KL 调用，且不再构造
`[local_rows, topk, kv_dim]` selected-KV 临时张量。sparse backward 使用相同的
compressed-first 索引顺序；KL backward 仍按保存的 exact top-k 重计算，因此 9-tensor
saved-state ABI 不变。

## CP plan 与通信

- dispatch/transfer plan 以 sample-relative logical positions 和可非连续 fragments 表示；128 对齐只约束可切分边界，不假设 fragment owner 相邻。
- compressed KV、compressed Ki、window KV、compressor overlap x 是四种独立 typed payload，分别持有 metadata、buffer slot 和 work handle。
- compressed KV/Ki 静态发送到全部 CP peers；window KV 和 overlap x 按 sample/fragment transfer table 只发送 unique rows。
- 正式数据交换只通过 `group_cast` / `group_reduce`。DSA 生产代码不直接调用 torch P2P、all-gather、all-reduce 或 `all2all_v` 替代。
- backward 使用 forward 的对称 GroupReduce；窗口与压缩条落到同一原 token 的梯度先用 FP32 accumulator 合并。
- collective 进入顺序在全部 rank 一致；空路线仍传合法零长度 metadata。

步骤 2 已建立不可变的 host plan：`DsaFragmentSpec` 使用 sample-relative
坐标；完整 plan 显式保存每 rank fragments、compressed block owner、分段
restore map，以及 window KV / compressor overlap x 的 unique-row transfer
table。sequential baseline 按 128 对齐连续切分；balanced solver 先最小化
ratio=4 最慢 rank 的 Indexer scan cost，再在允许 slack 内按 token、通信、
fragment overhead 和显存预测选择完整 E2E 最小的候选。rank 0 确定性求解后
广播 plan，runtime 按 packed layout 与 policy 缓存。

ratio=4 的每个 owner-local dispatch fragment 直接作为一个 query tile；同一组
fragment tiles 一致用于 Indexer projection、top-k、forward KL 和 backward KL
recompute。tile 不跨 sample 或 ownership 边界，但不再在 fragment 内按 128 rows
二次切分，从而避免小 GEMM 和 Indexer kernel 的 launch storm。不同 dispatch policy
可能改变 BF16 GEMM 的 row grouping，因此跨 policy/CP 的结果按冻结精度容差验收，
不要求 bitwise 等价。
cuDNN top-k launch 同样保持固定配置宽度 K=512；短 sample 的不足项由内核写成
`-1`，语义处理前再裁到实际 compressed-key 数。不得把 launch K 特化为 473 等
大于单 CTA 且为奇数的 key 数，否则冻结 CuTe 内核的两元素 vector-store 编译分支
会触发整除断言。

步骤 3 在 `functional/dsa_comm.py` 中实现了通信层。`DsaCommPlan` 为四类
payload 各自生成一个 `GroupCollectiveArg`：window KV 与 overlap x 根据
transfer table 将相同 owner row 的 destination 合并后只 pack 一次；
compressed KV/Ki 则把 owner-local compressed block 静态发送给所有 peer。
每条 source/destination 空路线也保留显式零长度 split，因此所有 rank 能按
固定顺序进入 collective。

步骤 4 保留 torch `index_select`/`index_copy_` reference map 仅用于测试对拍；
production forward 使用静态 device-resident int32 destination→source map 和 CuTe
DSL row-copy kernel 打包 owner-local unique rows，再调用 `group_cast` 得到带稳定
logical row id 的 typed receive buffer。backward 复用同一元数据和 payload-private
native handle 调用对称 `group_reduce`，GroupReduce 初始 owner rows 也由 row-copy
kernel 提取；返回结果按 inverse CSR map 以固定顺序、无 atomic 写回 owner-local
FP32 accumulator，最后只做一次目标 dtype 转换。device map 可作为静态 plan state
共享，send/receive buffer、work 和 native handle 仍按调用独占。

`dsa_pack.py` 同时提供 logical block/token id→packed-local row 的 int32 remap LUT；
`-1` sentinel 和 unavailable logical id 保持 `-1`，动态 Indexer top-k 全程留在
device。CSR kernel 支持重复 source、空 destination row、极端 fan-in，以及覆盖
非空 row或累加到已有 FP32 accumulator两种模式，为步骤 5/6 的 KV bank
packing 和多路径梯度合并提供固定接口。

native grpcoll 的每一条 transport row 都按 dtype 查询对齐要求，并只在内部
send/receive/reduce buffer 右侧补零：例如 BF16 compressed Ki 从逻辑宽度 128
补到 256，FP32 hidden=64 的 GroupReduce 从 64 补到 128；wait 后统一裁回并恢复
原 logical row shape。A2AV、logical row id 和对外 tensor schema 均不改变。
replicated 参数或 sink 梯度若展平为超宽 FP32 row，会按不超过 4096 elements 的
对齐 row 分块再做 GroupReduce，避免 native scratch 随单 row 宽度膨胀；归约后按
原 numel/shape 裁回。

`DsaCommBufferSlot`、`DsaGroupCastWork`、`DsaGroupReduceWork` 和 native handle
dictionary 都按单次调用、单 payload 创建，不进入 runtime 共享静态对象。生产
DSA 通信源码只调用 `group_cast` / `group_reduce`；A2AV 是该 primitive 的内建
fallback，不在 DSA 层直接调用。

![步骤 2 负载均衡改前与改后](assets/dsa_step2_load_balance.svg)

步骤 3 修改前：

![步骤 3 通信修改前](assets/dsa_step3_comm_before.svg)

步骤 3 修改后：

![步骤 3 通信修改后](assets/dsa_step3_comm_after.svg)

## Saved-state 与并发

- CP=1 和 CP>1 共用高层 custom-autograd saved-state ABI。saved-tensor hooks 只能观察到 9 项：5 个 owner-local 原始输入 `x/qr/q/latent_kv/sink`、`O`、FP32 `LSE`、int32 `topk_idx` 和 int32 `topk_length`。ratio=4 的 `topk_idx` 宽度为 512，ratio=0/128 为零宽占位。
- 不保存 compressed KV/Ki、最终 sparse indices、remote/packed tensor、collective work、native handle 或 CUDA event。backward 按静态 plan 重新发起四类 GroupCast，重算两个 compressor，并用已保存 top-k 做 Indexer score recompute/backward；不重算 Indexer top-k 或 sparse forward。
- work、event、remote buffer 和 saved-state 都属于单次 autograd 调用，不放入 runtime 共享静态状态。
- 输入梯度合同为：`q` 和 owner-local `latent_kv` 接收主 attention 梯度；`x` 只在 ratio=4/128 的主 compressor 路径接收梯度，ratio=0 为 `None`；Indexer 输入分支对 `x/qr/q` 和主 compressed KV detach，因而 KL 不向 trunk 输入传播，`qr.grad` 为 `None`。窗口和压缩条对同一 owner token 的梯度先在 FP32 CSR accumulator 中合并，最后转回目标 dtype。
- `d_sink` 与 runtime 内所有 replicated 参数梯度由 DSA 内部使用 FP32 GroupReduce 求和，并在公共 tensor/parameter 上标记 `_magi_dsa_cp_reduced=True`；外层不得沿同一 CP group 重复归约。owner-local 输入梯度不是 replicated 梯度。
- 步骤 7 的 `DsaOverlapConfig` 独立控制 compressed GroupCast/Indexer projection 与 dKi GroupReduce/sparse backward 两组 overlap。步骤 1–7 的 CP=8 历史兼容复验同时检查 2×2 开关矩阵、两个 in-flight microbatch、reentrant/retain-graph backward、gradient accumulation 和空 rank，并分别执行了 A2AV 与 native；A2AV 仅属于这一历史兼容覆盖。步骤 8/9 的正式出口只接受实际 `GrpCollIntraHandle` native 路径，A2AV 或 hierarchical fallback 均是硬失败。旧 CP=2 对真正嵌套 backward 的通过结果只保留为兼容历史，不能证明 CP=8 的 payload-private 状态与 grpcoll stream 串行语义。
- CP=1 runtime 支持完整 module deepcopy/`torch.save`，反序列化时重建锁并清空 device/process-group 绑定的 forward-plan cache。CP>1（包括 CP=8）的 `ProcessGroup` 是 PyTorch 不可 pickle 的外部状态，只支持保存 `state_dict`，再在目标进程组上新建 runtime 并加载。
- 正常退出和协调可恢复异常会 drain 已发起 work；collective wait 或 rank-local compute/kernel 失败会在公共前缀 drain 后 best-effort 请求 `ProcessGroup.abort()`（不可用时尝试 destroy），然后向上抛出。真实 B300 探针表明进程内 abort 不能保证解除 peer 已进入的 CUDA stream wait，因此不可恢复 CUDA/NCCL/NVSHMEM 故障采用 fail-stop 语义：外部 launcher 必须在 60 秒 watchdog 内终止全部 8 个 worker，然后只能新建进程组；不承诺原进程组恢复。步骤 9 的隔离子进程程序位于 `agents/release/magi-dsa-v4/fault_supervisor.py` 和 `fault_worker.py`；最终封存 case 已通过：fault rank 3 rc `86`，7 个 peers 已发起 native GroupCast，`3.064738 s` 内回收且 `survivors=[]`，随后 fresh 8-rank native health 全部 rc `0`。

## Production 镜像与 installed-wheel 边界

- `agents/release/magi-dsa-v4/build_image.sh` 会把请求的、可由 Git 无歧义解析的 revision 归一化为完整 40 位 commit；发布流程策略仍要求调用方显式传入完整 release-image source commit。构建命令中的 `REPO` 必须指向一次性的 integration worktree，并已在该 revision 上把递归 submodule 初始化到记录的固定 gitlink，不能指向开发 worktree。builder 从 Git object 递归 archive 根仓与冻结 gitlink，生成不含 `.git`、ignored `.so`、本地 build cache 或 dirty worktree 文件的临时 Docker context，并写入 `.magi-source-revision` 与 `.magi-submodules`。因此未提交文件绝不能偶然进入发布镜像。
- `agents/release/magi-dsa-v4/Dockerfile.production` 固定 NGC base digest、FlashMLA/cudnn-frontend/fast-hadamard-transform/Megatron commits、CUTLASS DSL 和 NVSHMEM 版本。NVSHMEM 必须在 MagiAttention wheel 之前安装，以保证 `magi_attn_comm` 不会以 `DISABLE_NVSHMEM` 编译。源码在镜像内只用于构建 wheel 和携带冻结测试；runtime 必须从安装后的 wheel 解析 `magi_attention`。
- 构建会校验 wheel metadata 版本、`magi_attention.__version__`、OCI revision/version/base labels、native extension 导入、`pip check`、SM100-family cubin 证据与固定子仓指针，并在 `/opt/magi-dsa-build-manifest.json` 记录 revision、依赖、版本和产物 SHA256。发布证据以 content-addressed `sha256:...` image ID 和 artifact manifest 为准，可变 tag 不是验收身份。
- installed-wheel smoke、性能验收和 final matrix 均不得 bind-mount 主机源码或 Megatron checkout，不得从 `/opt/MagiAttention` 的 source tree shadow 已安装包；容器只可挂载冻结 packs、performance/profile 输出和 final-result 等 artifact 目录，禁止挂载任何源码目录。公共导入 smoke 要在 source checkout 之外运行，并同时证明 8 块 B300 的 `(10, 3)` capability 与 native module 可用。
- CP8 native 固定设置 `MAGI_ATTENTION_NATIVE_GRPCOLL=1`、`MAGI_ATTENTION_HIERARCHICAL_COMM=0` 和 `CUDA_DEVICE_MAX_CONNECTIONS=8`，并 unset `NVSHMEM_SYMMETRIC_SIZE`。`GrpCollConfig` 固定 `num_sms=20`、`num_nvl_bytes=1073741824`（每 buffer 1 GiB）和 `num_rdma_bytes=0`；四类 typed payload buffer 与 replicated-gradient buffer 必须在计时前 dry-allocate，随后实际 handle 必须是 `GrpCollIntraHandle`。
- `agents/release/magi-dsa-v4/final_matrix.json` 是安装包验收顺序：public API smoke、CP=1 oracle、单个 GPU0–7 CP=8 native 合同和隔离 fail-stop。`run_final_matrix.py` 对每个 case 使用独立 process session、超时 TERM/KILL、JUnit 计数和零 skip 检查，并封存报告与 SHA256 manifest。最终 run `b300-cp8-final-f8ad2e5d-20260713T110210Z` 的四个 case 全部通过：smoke `8.629054 s`、CP1 `93/93`、CP8 `18/18`、fail-stop `47.943240 s`。
- 矩阵测试的 integration HEAD 为 `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`，对应 DEV commit 为 `e8f50e97896fec4eee5646988e6797e9c5a8b76c`；两者文档编辑前的 committed tree 均为 `38846f45dc5c10b30d0679f1f6decc252e78c1b2`。最终文档 commits 由交付回复记录以避免 commit 自引用。四个冻结的递归 gitlink 分别为：`magi_attention/csrc/cutlass` `81a43e6d92cdd8c20d22392f9579604ed5f710a1`、`magi_attention/functional/flash-attention` `ee1d15159cda6f3f97bfab9e487da146a8254970`、其 `csrc/composable_kernel` `e8709c24f403173ad21a2da907d1347957e324fb` 和 `csrc/cutlass` `b1d6e2c9b334dfa811e4183dfbd02419249e4b52`。

## 2026-07-13 sampled/non-formal 收尾状态

用户选择在约三小时内先完成明确标记的抽样收尾；这没有修改正式步骤 8 的冻结
合同。正式 calibration attempt
`b300-cp8-calibration-66d69258-20260713T054451Z` 在上述 calibration image 上停止于
`57/120`：ratio 0 的两个 cases × 20 packs 共 40 个单元完成，随后
`r4-sequential-00` 完成 packs 0–16。容器在 `2026-07-13T07:45:12Z` 收到外部
`SIGTERM` 后终止，非 OOM，也不是 fail-stop 用例。该 run 只作为
external-SIGTERM stopped-run 的 progress+正确性诊断证据保存，不得
resume、拟合正式系数或通过正式 validator。20/20 packs、逐 pack
`max/mean-1 <= 5%`、正式 speed gates 和 Nsight overlap 证据仍未满足，也没有被
放宽。Run 目录
`agents/perf/magi-dsa-v4-balance/b300-cp8-calibration-66d69258-20260713T054451Z`
内 `progress.json` 的文件 SHA256 为
`6f5520099807c290452b1a68da479e95240dbb1c9dfb19e54cc8908c025c528f`；
`agents/perf/magi-dsa-v4-release/calibration-logs` 下对应
`magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.docker-events.jsonl`
和 `magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.termination.json`
的文件 SHA256 分别为
`308245df4cdf2cf8db521960fb6ab5f1a3b9cab7e4823909ca20f3eb4ee81ce7` 和
`927974b501330e4339a9e2793b5236697eb92d46294dc8733d0e1bd148c50870`。

抽样 calibration 只选择 packs `[0, 1]`，运行 ratio 0/4/128 各自的
`sequential:00`/`balanced:00` 六个 cases，共 12 个 case-pack 单元。输出 target
固定为 `sampled-b300-sm103`，metadata 必须包含 `formal=false`、
`scope=sampled_non_formal`，冻结时必须显式使用 `--allow-sampled-non-formal`。已完成的
run 为 `b300-cp8-sample-calibration-20260713T083426Z`，revision
`66d69258fc9df1ee96f4c91df5d8225173b7fae2`，image
`sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`，进度
`12/12`，记录数 correctness `12`、plans `96`、raw timing `960`。冻结系数 ID 为
`df625f1805024f6c7c78e2bdcb07476c8be94347c3e762a80006a87aea0e882d`；证据目录
`agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-calibration-20260713T083426Z`
中 `sample_calibration.json` SHA256 为
`5d42c3ea327cf866f7faed7db3334d5d8c7cf56e44d1b60624c219d608e4c0e3`，
`artifact_manifest.sha256` 文件 SHA256 为
`82ccd78736f7972b4160e442350ea8fee63a591bfe2597ed46d427c74e742883`。首次抽样运行因 calibration
image 早于 `sample` 子命令，只读 bind-mount 了已提交的 benchmark `driver.py`；被测
`magi_attention` 仍必须来自 installed wheel，但该 driver bind 进一步确认这只是
non-formal diagnostic，不能充当 clean formal run。
ratio 4 的 bounded least-squares 拟合为 `R²=0.923198`、`RMSE=83.118 ms`；它只含
两个独立 packs，且 window/overlap 特征严格共线
（`window_rows = 31.75 * overlap_rows`，design rank `6/7`）。因此两项系数不可分别
识别，该拟合只适用于 sampled diagnostic target，不能证明正式模型拟合质量。

从抽样系数重建 final image 后，性能诊断目标为 9 个 case-pack 单元：pack 0 运行
ratio 4 `sequential:00` 和 balanced `00/01/10/11`，再运行 ratio 128
`sequential:00`/`balanced:11`；pack 16 运行 ratio 4
`sequential:00`/`balanced:11`。每单元仍使用 1 compile、2 warm-up、10 measure 和
逐元素 forward/全部梯度对拍；summary 必须保持 `formal_acceptance=false`、
`formal_gates_evaluated=false`。三类 run 的 pack 集合不得合并：stopped attempt 的
ratio 0 覆盖全部 20 packs，ratio 4
sequential 覆盖 packs 0–16；sampled calibration 选 packs `[0, 1]`；sampled measure 选
packs `[0, 16]`。这不是新的“五-pack measure”，不得描述为正式性能覆盖。

Predecessor f88 sampled timing 已完成 9/9 个 case-pack 单元的 correctness，产生 plans `72`、
raw timing `720`。三个源 run
`b300-cp8-sample-measure-a-f88ca22d-20260713T093515Z`、
`b300-cp8-sample-measure-b-f88ca22d-20260713T101533Z` 和
`b300-cp8-sample-measure-c-f88ca22d-20260713T103247Z` 的
`artifact_manifest.sha256` 文件 SHA256 分别为
`8ec095b475228012a2029a49066595692ae7c431fb12fe0a33981c952a7948f5`、
`a508503306049cc5460640b6babdb1ef69be2a942ea3bd210ba5c37e7013741e` 和
`7423cfc56ad63ee8f2f9f6c4716eff7f9f6164ca7890e9e8a76f8c19dcf6d64c`。聚合证据目录为
`agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-measure-f88ca22d-20260713T103500Z-aggregate`；
其中 `sampled_measure_diagnostic.json` SHA256 为
`47fb9e1ab4e852be90046c03438ac29ef8c29c552193cfe68e9ff36e262cded5`，
`artifact_manifest.sha256` 文件 SHA256 为
`955e17348d4ab00c11a582be2c2dfb4b7bcb6abe79bd9d8eadb6d89dfc93a62b`。Artifact 明确记录
`formal=false`、`formal_acceptance=false`、`formal_gates_evaluated=false`、
`all_gates_pass=null`。ratio 4 packs 0/16 两包诊断的 baseline/candidate E2E 为
`28602.9863/29245.3057 ms`，candidate 慢 `2.245637%`；Indexer 为
`474.7959/493.6558 ms`，candidate 慢 `3.972199%`，因此
`sampled_performance_observation_pass=false`，但两包 candidate E2E rank imbalance 均不超过
5%。ratio 128 pack 0 的 E2E 为 `321.7919/320.7076 ms`，candidate 快
`0.336954%`，rank imbalance 为 `0.10824%`。这些只是 sampled observations，不是正式 gate
结果。
Sampled/diagnostic profile 未运行，也未产生 profile artifact；正式 ratio-4
Nsight overlap 门槛未评估。

步骤 9 在 sampled closure 中仍要求 final image 的四个有序 case 全量运行：安装包
公共 API smoke、CP1、单个 CP8 native 组和隔离 fail-stop 均不能抽样或 skip。最终
f8 run 已完整通过并封存，因此 sampled-closure 步骤 9 已完成；这仍不能把步骤 8 的
sampled evidence 升级为 formal acceptance，也不构成 formal release qualification。

首次步骤 9 run `b300-cp8-final-f88ca22d-20260713T103532Z` 在旧 f88 candidate
`sha256:4bd9b55dd1303043256fd6accf79a5362f630b85448a5425958c42a2467d0bea`
上运行：public smoke `8.679 s` PASS，CP1 `93/93` 在 `99.657 s` PASS，CP8 在
`7.728 s` 得到 `4 passed, 14 failed` 后停止。根因是 pytest importlib mode 下从
`/tmp` 启动的 spawn 子进程报 `ModuleNotFoundError: tests`，不是 DSA correctness、
OOM 或 watchdog；单项 diagnostic 改用 `--import-mode=append` 后，native 8-rank test
为 `1 passed`（`36.61 s`），且 `magi_attention` 仍从 installed-wheel
`site-packages` 导入。失败 run/outer manifest 文件 SHA256 分别为
`81f8f21a524f4d2dd050ca591358234b133f222de8e03523c75b083caf44757d` 与
`0818fe00f144b63d7718e8d38f3a775a91da4acef824419b639491b3a59107b3`。修复 commits
为 DEV `e8f50e97896fec4eee5646988e6797e9c5a8b76c`、integration
`f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`；重建后的 final Step9
candidate 是上文已封存的 f8 镜像。旧 f88 镜像仅保留为 sampled timing 与失败矩阵
前身。

最终 run `b300-cp8-final-f8ad2e5d-20260713T110210Z` 使用 revision
`f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`、package
`1.1.1+dsa.f8ad2e5d4f8b` 和 image
`sha256:12075f8738456a417cac72f39a328378bb1bb1ee313c5fd7c34c92c6d0bf5c37`，
最终 `status=passed`、`error=null`。Public-API smoke 用时 `8.629054 s`；CP1 用时
`100.860094 s`，JUnit `93/93`、fail/error/skip 均为 0；CP8 native 用时
`550.507443 s`，JUnit `18/18`、fail/error/skip 均为 0；隔离 fail-stop case 用时
`47.943240 s`。fault rank 3 rc `86`，7 个 peers 均已发起 native GroupCast，并在
`3.064738 s` 内全部回收，`survivors=[]`；fresh health 的 8/8 ranks 均 rc `0`、
handle 为 `GrpCollIntraHandle`，配置为 NVL `1073741824`、RDMA `0`、
`num_rdma_ranks=1`、`num_sms=20`。

Container inspect 记录 exit `0`、OOM false、image 精确匹配、仅一个 artifact mount、
无 source bind，且 `NVSHMEM_SYMMETRIC_SIZE` 未设置。证据根为
`agents/perf/magi-dsa-v4-release/final-matrix-f8ad2e5d`；inner/outer
`artifact_manifest.sha256` 文件 SHA256 分别为
`fdc9a4cd0a96a5f69d554d6b89c898f6c83de1a26e81ab53de7e22285ae20c11` 和
`b045d7fb03db5e03f47c16c43a5dfde862e64545791bc3ad58c68eda47977d11`；
`final_result.json`、`fault_result.json` 和 inspect 文件 SHA256 分别为
`bb83d46d7f1e0b2a2db4591b4cbe8f45dffafaf3b5adc8d5dc5a7859e5cf1687`、
`77cfa8a1a937a8281bc590cd3c6b7fdf2098e68796adfe52c2133ef3c24783e1` 和
`6420dcac04ca064a2bbe4d9ec239e52a9e550fbd154de2d0247273807ff59449`；
只读 manifest verify 对 80 个 artifacts 全部通过。

## 范围边界

本轮交付是 MagiAttention 库级、可安装、可从稳定公共 API 调用的
`MagiDSARuntimeMgr + calc_dsa` 旁路。真实 DeepSeek 模型层接入、训练脚本接线和
drop-in 替换仍明确不在本轮范围内；本文不声称已完成 full-layer/model
integration。同样 out of scope 的还包括 dense warm-up、TP、FP8/FP4、decode/cache、
attention recompute、CUDA graph、单次调用混合 ratio 和 top-k 驱动的动态 KV 通信。

## 步骤 1–7 历史开发复验证据

以下 source-bind 命令只记录步骤 1–7 当时如何从开发 worktree 的 `tests/test_dsa/` 复验，不能作为步骤 8/9 的发布命令或 installed-wheel 证据。宿主机开发 worktree 当时是 `/home/scratch.wewen_gpu/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll`；不再从旧 native 镜像的 `/tmp/native-test/test_dsa_cp.py` 或主 checkout 的同名文件运行。CP=1 命令只生成 oracle 证据，分布式开发出口由同一个 8-rank 组完成；所用增量 native 容器和 source bind 均不是最终不可变 native 镜像。步骤 8/9 只遵循上节的 clean installed-wheel 边界。

```bash
WORKTREE=/home/scratch.wewen_gpu/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll
B300_BASE_IMAGE=sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da
NATIVE_CONTAINER=magi-dsa-b300-cp8-native
GPU=0
GPU_GROUP=0,1,2,3,4,5,6,7
GRPCOLL_NUM_NVL_BYTES=1073741824  # 1 GiB；test config 必须使用同一固定值

# CP=1 只作为数值/梯度 oracle，不计为分布式出口。
docker run --rm --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES="${GPU}" \
  -v "${WORKTREE}:/workspace/MagiAttention" \
  -v /home/scratch.wewen_gpu/megatron-lm:/ws/megatron-lm:ro \
  -e MAGI_DSA_MEGATRON_PATH=/ws/megatron-lm \
  -w /workspace/MagiAttention "${B300_BASE_IMAGE}" \
  timeout 1200 pytest -q -rs \
    tests/test_dsa/test_dsa_api.py \
    tests/test_dsa/test_dsa_dispatch.py \
    tests/test_dsa/test_dsa_megatron.py \
    tests/test_dsa/test_dsa_solver.py \
    tests/test_dsa/test_dsa_pack_kernel.py

# 步骤 3/5/6/7 的唯一分布式出口：一个 GPU0..7、world_size=cp=8 组。
# 单节点 NVLink 路径使用 num_rdma_bytes=0；NVSHMEM_SYMMETRIC_SIZE 必须 unset/N/A。
docker exec \
  -e CUDA_VISIBLE_DEVICES="${GPU_GROUP}" \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29617 \
  -w /workspace "${NATIVE_CONTAINER}" \
  bash -lc 'unset NVSHMEM_SYMMETRIC_SIZE; \
    test -z "${NVSHMEM_SYMMETRIC_SIZE+x}"; \
    timeout 5400 pytest -q -rs tests/test_dsa/test_dsa_cp.py'
```

当时的 `test_dsa_cp.py` 已不再是 transport-only：CP8 目标矩阵包含 A2AV/native transport、reference/kernel forward、reference/kernel backward/saved-state，以及步骤 7 的 overlap、并发、reentrant/retain-graph、空 rank、协调异常 drain/reuse 和 abort-request 用例。其中所有 `test_native_grpcoll_*` 用例都必须在安装了 `magi_attn_comm` 和 NVSHMEM 的 B300 镜像内实际执行；`-rs` 输出中出现 skip 即为验收失败。CP=2 节点只算兼容回归。`test_dsa_megatron.py` 同样必须加载只读挂载且 HEAD 精确为 `c6449f0b23be397449f21c0967c5fc90785e55ea` 的 checkout，不能以 skip 代替 parity。编译后的单 kernel 测试使用 30 秒 watchdog；CP8 transport、完整路径/并发和首次 sparse-backward 冷编译分别使用 180、600–900 和 1200 秒预算，隔离故障的 launcher fail-stop 硬期限仍为 60 秒。

2026-07-12 在 canonical tiling 修复前完成的历史开发矩阵为：API `32 passed`，dispatch+solver `36 passed`，packing+Megatron `18 passed`，单个八卡 `test_dsa_cp.py` `27 passed, 0 skipped`（637.08 秒），当时正式目录合计 `113 passed, 0 skipped`；旧回归为单卡 `17 passed`、CP8 `2 passed`。修复后新增两项 policy-invariant tile 回归，dispatch+solver 独立复验为 `38 passed`；本节不把不同 revision 的结果相加伪造新总数，最终 installed-wheel 总数只从步骤 9 的同一不可变镜像矩阵回填。历史 CP8 文件覆盖 A2AV/native、7168-wide reverse、三 ratio reference/kernel forward/backward、全部梯度、7 空 rank、2×2 overlap、并发/reentrant、retain-graph 和协调异常 drain/reuse。Black、isort、Ruff 0.12.5、compileall 与 `git diff --check` 通过。

本节上述数字只是步骤 1–7 的历史开发证据，不得被解读为 formal step-8/9 qualification。当前步骤 8 的正式状态仍是未完成；`57/120` partial attempt 与本轮 sampled calibration/measure 的区别和限制以上一节为准。Sampled calibration 已产生并冻结为上述 content-addressed 证据；sampled profile 未运行，正式 overlap 门槛未评估。Sampled final build 已以上述 revision、不可变 image ID 和证据目录封存，但只构成 non-formal candidate。Sampled measure 已封存在 `agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-measure-f88ca22d-20260713T103500Z-aggregate`，聚合 JSON SHA256 为 `47fb9e1ab4e852be90046c03438ac29ef8c29c552193cfe68e9ff36e262cded5`；它的 `sampled_performance_observation_pass=false` 不是正式 gate 结论。Sampled-closure 步骤 9 已由上述 f8 matrix 的四个全通过 case 和 content-addressed manifests 完成；矩阵测试的 integration/DEV commits 分别为 `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9` 与 `e8f50e97896fec4eee5646988e6797e9c5a8b76c`，pre-documentation committed tree 均为 `38846f45dc5c10b30d0679f1f6decc252e78c1b2`。这不改变正式步骤 8 未完成及不具备 formal release qualification 的结论。

## grpcoll 探针保存

原型 worktree `agents/worktrees/magi-dsa-v4` 的未提交 grpcoll 探针没有被覆盖或 reset；完整可应用补丁保存在 `agents/backups/magi-dsa-v4-grpcoll-probe-20260709.patch`，SHA256 为 `b6a6b8fadb48a38dc2c9b38bb1abd1fab8d5d86f27fedb67d298bded836416ee`。
