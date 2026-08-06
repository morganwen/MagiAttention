# [历史] Magi-DSA Flash-Base 概览

> **状态：历史 Base overview，不是当前 Pro 实现或 release 事实源。**
>
> 本文的 hidden `4096`、Indexer Top-K `512`、`[512 compressed + 128 window]`、
> Indexer-cost Query 分配和 Base profile 口径均为历史记录。当前 DeepSeek-V4-Pro 合同是
> hidden `7168`、H128/D512、Indexer H64/D128/Top-K `1024`、
> `[1024 compressed + 128 window]` 以及 CSA/HCA 共享的 `structural_balanced` Query layout。
> 请以 [`magi_dsa_v4_design.md`](./magi_dsa_v4_design.md) 和
> [`README_dsv4_cp_dispatch_structural_balancing.md`](./README_dsv4_cp_dispatch_structural_balancing.md)
> 为准；下文只用于理解历史 Base 实现。

## 1. 概览
1.1 MagiAttention
MagiAttention 是面向超长上下文训练的分布式注意力系统。它以 Context Parallel 为基础，将一个 attention workload 的 Query 和 Key/Value 划分到多个 GPU rank，并通过计算切分、按需通信和计算通信重叠提升多卡扩展效率。
Magi 的核心能力包括：
细粒度负载均衡：将 attention mask 表示为可切分的计算区域，由 solver 把不同成本的区域分配给各个 CP rank；
按需通信：只把某个 Query 分片实际需要的 K/V 发送到对应 consumer，避免无差别复制完整activation；
前后向对称路由：forward 获取 remote K/V，backward 通过反向通信和 CSR reduction 将梯度归还token owner；
计算通信重叠：用 CUDA stream、event 和异步 collective 隐藏网络延迟；
冷/热路径分离：solver、route 和 device map 在执行前构建，训练热路径只执行冻结的计算通信 DAG。
通用 MagiAttention 主要解决 heterogeneous mask 下的分布式 dense/sparse attention。Magi-DSA 不直接
套用其通用 attention runtime，而是复用 Magi 的 fragment、solver、All2AllV、packing/CSR、异步生命周期
和 autograd 基础设施，为 DeepSeek-V4 DSA 建立一条独立执行路径。
1.2 Magi-DSA
DeepSeek-V4 DSA 同时包含 raw window、compressed KV、Indexer 和 Top-K，计算结构与普通 attention
不同。Magi-DSA 的核心思路是：在进入 DSA 前按 Indexer 成本重新分配 Query；进入 DSA 后 Query 保持
local，只按静态 plan 获取 remote KI/KV，并在本地完成 Indexer、sparse attention 和 KL。
Magi 分布式基础设施
+ DeepSeek-V4 DSA 数学
+ Indexer-aware Query 负载均衡
+ FlashMLA / cuDNN 执行后端
当前支持三种互斥模式：
ratio
模式
Key 范围
Indexer
0
Window Attention（W）
最近 128 个 raw KV
无
4
Compressed Sparse Attention（CSA）
512 个 selected compressed KV + 128 个 raw KV
有
128
Hybrid Compressed Attention（HCA）
全部 causal-visible compressed KV + 128 个 raw KV
无
本文重点说明计算和通信最复杂的 CSA（ratio=4）。
2. 问题
2.1 Indexer 负载天然不均衡
对 sample 内位置 p 的 Query，可见 compressed-K 数量随位置增长：
visible_k(p, ratio) = floor((p + 1) / ratio)
如果按 packed token 的连续位置平均切分，每个 rank 的 token 数虽然相近，但越靠后的 rank 需要扫描越长的
K prefix。Indexer score 和 Top-K 的成本因此显著不均，训练 step 被最慢 rank 拖住。
[图片]
2.2 Query 与 K 的所有权不同
均衡后，一个 rank 可以持有来自多个 sample、多个不连续区间的 Query；而 raw KV 和 compressed K
仍由各自 token/block owner 产生。系统必须满足：
- Query 的位置和因果边界始终相对各自 sample；
- 每个 Query 只计算一次；
- local K 直接使用，remote K 通过静态路由获取；
- 重复 K prefix 只传输一次，接收后再本地展开；
- backward 将多个 consumer 的 K 梯度无遗漏、无重复地归还 producer。
2.3 Top-K 引入两套 K 视图
CSA 使用 compressed-KI 做 Indexer score，使用 compressed-KV 做主注意力。两类 bank 可以采用不同的
consumer-local 行序，因此同一份 global Top-K ID 必须分别映射到 KI bank 和 KV bank，不能假设二者
local index 相同。
3. 总体设计
Magi-DSA 将执行分为冷路径和热路径：
冷路径（plan）：根据全局 `cu_seqlens` 和 source layout 生成 Query fragments、负载均衡方案、block owner、四类 typed route、pack/CSR map，并将其物化为不可变的 device plan。
模型边界：在进入 DSA 前，对 pre-projection hidden state `x` 执行一次 `TOKEN_LAYOUT` All2AllV；随后在最终 Query rank 上生成 qr/q/latent_kv。
热路径（execute）：Query 不再移动，只获取 local Query 所需的 remote KI/KV，在本地完成grouped Indexer、Top-K、FlashMLA sparse attention 和 selected KL。
反向路径：先在 consumer 侧用 CSR 合并重复行梯度，再按 forward 的 route 反向 All2AllV，最终归还 token/block owner。
4. 负载均衡设计
4.1 Sample-relative fragment
planner 用以下三元组描述 Query 区间：
DsaFragmentSpec(sample_id, q_begin, q_end)
q_begin/q_end 始终是 sample-relative 坐标。所有 fragments 必须精确覆盖每个非空 sample：
不能重叠、不能留洞。一个 rank 可以接收多个不连续 fragments，并按 plan 给出的 local order 拼接。
4.2 成本模型
CSA 同时估算 Indexer score 和 Top-K 成本。实现按 cuDNN tile 对可见 K 数取整：
score_cost(p) = max(128, round_up(visible_k(p, 4), 128))
topk_cost(p)  = max(256, round_up(visible_k(p, 4), 256))
planner 默认以 128 个 Query 为 assignment atom，按“重任务优先”的贪心策略分配。候选 rank 的选择目标
依次考虑：
1. score/top-k 归一化负载的较大值；
2. 两种归一化负载之和；
3. 绝对负载、fragment 数和 rank ID。
同时，每个 rank 的 Query 数被约束为 floor/ceil(total_tokens / cp_size)；必要时拆分 atom，以精确满足容量。因此该设计不会以严重 token imbalance 换取 Indexer 均衡。
[图片]
4.3 前置 TOKEN_LAYOUT
planner 生成 source token 到最终 Query owner 的全局双射：
source-owner x
  -> TOKEN_LAYOUT All2AllV(x)
  -> final-Query-owner x
  -> local projection
  -> x / qr / q / latent_kv
  -> DSA
只路由一次 [Tsource, 4096] 的 BF16 x，而不是分别路由所有 projection 输出。进入 DSA 后 Query
始终 local，不存在 Query worker、Query 广播、distributed Top-K 或 Top-K restore。
5. 前向
[图片]
CSA 前向的主要步骤如下：
1. 启动基础路由
- WINDOW_KV：获取 local Query 所需的 128-token raw window；
- OVERLAP_X：获取本 rank 生成 compressed row 所需的跨 owner hidden support。
2. 本地 Indexer projection
- 由 local x/qr 生成 q_indexer 和 per-head weights；
- Q 侧使用 sample-relative RoPE 和 Hadamard，weights 使用 1/sqrt(64) 缩放。
3. 生成 compressed K
- Indexer Compressor 生成 128 维 compressed-KI；
- Main Compressor 生成 512 维 compressed-KV；
- 每个完整 4-token block 只由一个 block owner 生成。
4. 路由 compressed K
- 先启动 COMPRESSED_KI，再执行 Main Compressor 并启动 COMPRESSED_KV；
- KI 通信可被 Main Compressor 遮挡，KV 通信可与后续 grouped Indexer 重叠。
5. Grouped Indexer 与 Top-K
- 将 unique KI 按 fragment 展开成 varlen prefixes；
- 每个 rank、每个 step 只执行一次 grouped cuDNN score 和一次 grouped Top-K；
- Top-K 从产生开始就是 Query-owner local order，只做 global offset、有效长度和 -1 padding。
6. 构造 attention indices
- 同一份 global Top-K ID 分别映射到 KI/KV 两套 local bank；
- FlashMLA indices 固定为 [compressed: 512, window: 128]，无效项填 -1。
7. Sparse attention 与 selected KL
- FlashMLA 一次前向返回 output、完整 sparse LSE 和 compressed-prefix LSE；
- 完整 LSE 仅供 sparse attention backward；
- compressed-prefix LSE 用于 selected teacher；
- cuDNN 对 selected compressed K 重算 predictor/teacher，并预计算单位 KL 梯度；
- 未选中的 candidate 梯度严格为 0。
CSA 的 Indexer score 定义为：
score(q, k) = sum_h(
    relu(dot(q_indexer[q, h], compressed_ki[k]) / sqrt(128))
    * weights[q, h]
)
5.1 前向通信
顺序
Route
Payload
作用
1
WINDOW_KV
512 BF16
remote raw window
2
OVERLAP_X
4096 BF16
remote compressor support
3
COMPRESSED_KI
128 BF16
remote Indexer K
4
COMPRESSED_KV
512 BF16
remote attention K
因此 CSA 的 DSA core 前向固定为 4 次 All2AllV。外层 `TOKEN_LAYOUT` 单独计账，不属于这 4 次通信。
7. 反向
[图片]
反向不重新计算 Top-K，也没有 Top-K backward。主要流程为：
1. cuDNN sparse attention backward 产生 dQ、dSink、dWindowKV 和 dCompressedKV；
2. selected-KL backward 缩放 forward 保存的单位梯度，产生
dQIndexer/dWeights/dSelectedCompressedKI；
3. packed prefix 中重复的 KI/KV 梯度先在 consumer 侧用 CSR 合并；
4. 四类梯度通过 reverse All2AllV 返回各自 producer，再在 owner 侧做 CSR 合并；
5. dCompressedKV/dCompressedKI 分别进入 Main/Indexer Compressor backward；
6. late join 等待两条 Compressor support gradient 以及 dQIndexer/dWeights 就绪，再释放
projection 和 support 两侧反向；
7. DSA 返回 local dX/dQR/dQ/dLatentKV，模型侧完成 projection backward 和 dx_local 汇合；
8. 模型边界通过 reverse TOKEN_LAYOUT 将 dx_local 返回 source owner；
9. Compressor、Indexer 和 sink 的 partial parameter gradients 由模型侧统一 CP AllReduce。
6.1 通信与计算重叠
四条 reverse 共用同一 communicator，固定提交顺序为：
COMPRESSED_KI
  -> COMPRESSED_KV
  -> OVERLAP_X
  -> WINDOW_KV
不同 CUDA stream 不改变 collective 的全 rank 顺序，只用于把通信隐藏在独立计算后面：
- COMPRESSED_KI reverse 与 sparse attention backward 重叠；
- COMPRESSED_KV reverse 与 Indexer Compressor backward 重叠；
- OVERLAP_X、WINDOW_KV reverse 与 Query/weight projection backward 重叠。
event 表达 producer-consumer 依赖，record_stream 保证异步 tensor 生命周期；热路径不使用
.item()、host synchronize 或隐式 fallback。
6.2 反向通信账本
CSA 的 DSA core backward 固定为 4 次 reverse All2AllV。因此完整 DSA core 为：
4 forward All2AllV + 4 backward All2AllV
外层 TOKEN_LAYOUT forward/reverse 以及模型参数/sink 的 CP AllReduce 均单独计账。
