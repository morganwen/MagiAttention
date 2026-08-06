# Magi-DSA CP 负载均衡设计

> 通信部分的结构照 `extensions/magi_attn_extensions/MSA` 的已落地实现设计（required-K 前缀 + consumer 内去重 + 复用 `DynamicAttnSolver` range planner），本文标注 DSA 与 MSA 的差异处。
>
> 当前冻结范围是官方 `DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1`：
> 主干 61 层包含 31 个 HCA 与 30 个 CSA，另一个 `ratio=0` 的 MTP attention 只保留接口边界，
> 不进入本次主干实现或正式性能验收。训练 attention ABI 使用 BF16 activation/gradient、FP32
> score/LSE/sink/accumulator 和 INT32 index。主干支持 61 份相互独立的层参数；正式性能验收使用
> 代表性 CSA+HCA 连续执行 5 轮 forward/backward，不用两层代表权重冒充 61 层参数支持。

---

## 1. 问题定义

一个 CP group 含 $P$ 张 GPU。输入 $S$ 个长度不等的 doc

$$\ell_1,\ldots,\ell_S,\qquad T=\sum_i\ell_i$$

packing 成一条序列，mask 为块对角 causal。序列按固定位置切成
$n=\lceil T/b\rceil$ 个 chunk（不看 doc 边界，最后一个允许不足 $b$），分配到 $P$ 张卡。
分配后只在主干入口执行一次 `TOKEN_LAYOUT(x)`，全部
$H=H_{\mathrm{CSA}}+H_{\mathrm{HCA}}$ 层共用同一 Query 布局；中间 hidden/residual 保持该布局，
不逐层重排。

DSV4-Pro 结构与取值：

$$r=4,\quad r'=128,\quad \kappa=1024,\quad \sigma=128,\quad H_{\mathrm{CSA}}=30,\quad H_{\mathrm{HCA}}=31$$
$$n_h=128,\quad c=512,\quad n_h^I=64,\quad c^I=128$$

**CSA 层**三个分量，**HCA 层**一个分量：

| 分量 | 内容 | 每 query 可见 entry 数 | 饱和 |
|---|---|---|---|
| lightning indexer | 对全部前序 entry 打分（无 softmax、无 value），排序取 top-$\kappa$ | $\lfloor t/r\rfloor$ | 否 |
| CSA 核心注意力 | 只算选中的 entry + 滑窗 | $\min(\lfloor t/r\rfloor,\kappa)+\min(\sigma,t)$ | **是**（$t\ge r\kappa$ 后恒定） |
| HCA 密集注意力 | 重压缩 entry 上的稠密 causal + 滑窗 | $\lfloor t/r'\rfloor+\min(\sigma,t)$ | 否 |

单位成本（每 $(\text{query},\text{entry})$ 对的 FLOPs，$1$ MAC $=2$ FLOPs）：

$$\alpha_{\mathrm{idx}}=2n_h^Ic^I=16{,}384,\qquad
\alpha_{\mathrm{core}}=\alpha_{\mathrm{hca}}=4n_hc=262{,}144$$

indexer 只打分故无 PV 项；核心注意力含 $QK^\top$ 与 $PV$ 两遍，故为 $4n_hc$。

核心矛盾：

1. token 开销取决于**doc 内**位置，packing 后在每个 doc 边界重置 —— 负载曲线是**锯齿波**而非斜坡
2. 布局全层共用，同一 rank 的过载**逐层累加、不互相抵消**
3. 通信量依赖布局（§5）

---

### 1.1 官方 attention 输出坐标边界

本文的 cost 和 route 虽以 Query/KV 可见面积建模，但 backend 返回的原始 attention 输出还不是
可直接进入模型 output projection 的坐标。官方 Pro 合同要求每个 head 的末 64 维在返回前做：

```text
O_model[..., :448] = O_backend[..., :448]
O_model[..., 448:512] = RoPE^-1(
    O_backend[..., 448:512],
    sample_relative_query_position,
    layer_yarn_frequencies,
)
```

该操作只使用已在最终 Query order 中驻留的 sample-relative position，不新增 route，不改变
§3 的负载均衡 cost。实现必须 out-of-place，避免覆写 sparse-attention backward 保存的
`O_backend`；autograd backward 使用相反方向的正向 RoPE。ratio 0/4/128 共用当前 layer config
的同一组 YaRN frequencies；Pro 61 层正式实现和验收仍只含 30×CSA + 31×HCA，ratio 0
仅作 MTP/window-only 兼容边界。

---

## 2. 决策变量

$$y_{cp}\in\{0,1\},\qquad \Pi_p=\{c\mid y_{cp}=1\}$$

$b$ 不是决策变量但是关键超参（§4.3）。

---

## 3. 计算 Cost 模型

### 3.1 完整闭式

定义 $O(1)$ 前缀和：

$$F_\rho(x)=\sum_{t<x}\Big\lfloor\frac{t}{\rho}\Big\rfloor=\rho\frac{q(q-1)}{2}+q\theta,\qquad q=\lfloor x/\rho\rfloor,\ \theta=x\bmod\rho$$

$$G_\sigma(x)=\sum_{t<x}\min(\sigma,t)=
\begin{cases}x(x-1)/2,&x\le\sigma\\ \sigma(\sigma-1)/2+\sigma(x-\sigma),&x>\sigma\end{cases}$$

对覆盖 doc 内位置 $[u,v)$ 的 slice：

$$\boxed{
\begin{aligned}
A&=F_r(v)-F_r(u)\\
B&=F_{r'}(v)-F_{r'}(u)+G_\sigma(v)-G_\sigma(u)\\
D&=F_r\big(\min(v,r\kappa)\big)-F_r\big(\min(u,r\kappa)\big)+\kappa\max\big(0,v-\max(u,r\kappa)\big)+G_\sigma(v)-G_\sigma(u)
\end{aligned}}$$

$$w_c=H_{\mathrm{CSA}}\big(\alpha_{\mathrm{idx}}A_c+\alpha_{\mathrm{core}}D_c\big)+H_{\mathrm{HCA}}\alpha_{\mathrm{hca}}B_c$$

### 3.2 但实测表明 native 面积已经足够

用不同 cost 模型求解，再按**真实三分量成本**评估 $E$（$T=1\text{M}$，$b=512$，$P=64$）：

| 场景 | core 占比 | HCA 占比 | native 面积 | 仅 indexer | 完整三分量 |
|---|---|---|---|---|---|
| 单条 1M | 8.4% | 31.8% | 1.0001 | 1.0001 | 1.0001 |
| $128\times8\text{K}$ 等长 | 79.7% | 14.6% | 1.0000 | 1.0000 | 1.0000 |
| $2048\times512$ 等长 | 59.1% | 39.5% | 1.0000 | 1.0000 | 1.0000 |
| $32\times32\text{K}$ 等长 | 67.7% | 16.4% | 1.0003 | 1.0003 | 1.0003 |
| lognormal $\sigma=1.0$ | 69.5% | 16.4% | 1.0019 | 1.0019 | 1.0019 |
| lognormal $\sigma=1.5$ | 54.6% | 20.0% | 1.0024 | 1.0024 | 1.0014 |
| lognormal $\sigma=2.0$ | 37.5% | 24.3% | 1.0011 | 1.0011 | 1.0011 |
| $1\times512\text{K}+4\times64\text{K}+\cdots$ | 22.5% | 28.3% | 1.0015 | 1.0015 | 1.0006 |

最大差距 $1.0024$ vs $1.0014$，即 **0.1%**。原因：三个分量都是同一个 doc 内位置的单调函数，且

$$A_c\approx\frac{\texttt{area}_c}{r},\qquad B_c\approx\frac{\texttt{area}_c}{r'}+\sigma b$$

在"每 rank 等 chunk 数"约束下常数项被吸收，二者与 native 面积**近乎精确共线**；唯一非共线的 $D$ 因 $\min$ 饱和而动态范围极小（$t\ge r\kappa$ 后恒定），由等 chunk 数约束自动均衡。

$$\boxed{w_c=\texttt{AttnSlice.area}\ (\text{CAUSAL})\quad\text{—— 一期不写新 cost 模型}}$$

MSA extension 用裸 `MinHeapDispatchAlg()` + per-doc `CAUSAL` ranges 已经验证了这条路径可行。3.1 的闭式作为二期储备：若实测发现 $E$ 偏离预期（尤其在 $D$ 占比高的短 doc 批次），再替换 `chunk.area`，`MinHeapDispatchAlg` 本身不用改。

### 3.3 刻意不建模的项

压缩器 $W_{KV},W_Z$、query / index 投影、输出投影、MoE、LayerNorm 均**与位置无关、正比于 token 数**，由 §4.2 的等 chunk 数硬约束自动均衡。纳入 $w_c$ 只会稀释位置相关项。

---

## 4. 约束条件

### 4.1 Chunk 唯一分配

$$\boxed{\sum_p y_{cp}=1,\qquad\forall c}$$

### 4.2 等 Chunk 数

$$\boxed{\sum_c y_{cp}\in\{\lfloor n/P\rfloor,\lceil n/P\rceil\},\qquad\forall p}$$

用 `uneven_shard=True` 允许末尾不整除。§3.3 的 $O(T)$ 项与 §3.2 的共线性论证都靠它成立。

### 4.3 粒度：显式冻结在 structural plan schema

`DsaStructuralLayoutConfig` 显式保存 `chunk_size=512`、`min_chunks_per_rank=16` 和
`uneven_shard=True`；它们进入 plan hash/artifact，不能依赖进程环境中的隐式默认值。resolved size 为：

```text
min(configured_chunk_size, ceil_div(T, min_chunks_per_rank * cp_size))
```

$n/P$ 是每 rank 的自由度，**不是可行性约束而是质量旋钮**：

| $n/P$ | 32 | 16（当前默认） | 8 | 4 | 2 |
|---|---|---|---|---|---|
| $E$ | 1.0006 | 1.003 | 1.01 – 1.03 | 1.11 | 1.24 |

$$\boxed{\texttt{DsaStructuralLayoutConfig(min\_chunks\_per\_rank=16)}}$$

若 $T/P$ 小到需要 $b<\operatorname{lcm}(r,r')=128$，说明 **CP 开得过宽**（每卡 token 太少，通信与 launch 开销占比过高），应减小 $P$、把并行度让给 DP，而不是继续缩 $b$：

$$\boxed{\frac{T}{P}\ \ge\ 16\operatorname{lcm}(r,r')=2048}$$

### 4.4 压缩边界与 `OVERLAP_X`（DSA 独有）

默认 $b=512$，因而 packed-global chunk 边界与 $r=4,r'=128$ 对齐；这能减少边界 support，
但不是 correctness 前提。

这条是 DSA 与 MSA 在 dispatch 层最实质的区别。MSA 的 sparse block 是原始 token 分组；
DSA 的压缩器则读取同一 sample 内连续的 $r$ 个 token。packed-global chunk 可能穿过 sample 边界，
sample-relative 压缩块也可能穿过 Query chunk 或源 rank 边界，因此不能假设“压缩块绝不跨 chunk”。

实现以 sample-relative compressed-block 表为准，为每个 compressed row 选择唯一 producer；若 producer
缺少 support token，就通过 `OVERLAP_X` All2AllV 获取。反向沿同一 map 做 CSR reduction，保证跨边界
support 的 `dx` 不遗漏、不重复。这样无需 doc padding，也不会把 chunk 对齐假设偷偷变成数学合同。

---

## 5. 通信设计

### 5.1 required-entry 前缀 + consumer 内去重

照 MSA 的模式。对 rank $p$ 上属于 doc $d$ 的全部 Q fragments：

$$\text{required}(p,d)=\Big[\,\text{doc\_begin}_d,\ \max_f f.\text{k\_range.end}\,\Big)$$

causal 下 fragment 的 K 范围恒为 $[\text{doc\_begin},\ \text{doc\_begin}+q_{\mathrm{end}})$，故

$$\boxed{K_{pd}=\max_{c\in\Pi_p\cap d}\big(\text{chunk 在 doc }d\text{ 内的最深位置}+1\big),\qquad
V_p=\sum_{d:\Pi_p\cap d\neq\varnothing}K_{pd}}$$

通信 bank 内同一 doc 的多个 fragments 只接收一份 maximal prefix，因此
`COMPRESSED_KI` 的网络流量按 $V_p$ 去重。这里不能直接照抄 MSA backend 的
`fragment_indices`：当前冻结的官方 cuDNN grouped Indexer ABI 只接受每个 grouped fragment
连续的 K segment。执行前仍需用一次 `k_pack` 将 unique bank 展开成

$$P_p=\sum_{f\in p}\left\lfloor f.q_{\mathrm{end}}/4\right\rfloor$$

行；其重复开销为 $P_p-V_p$。因此“required-prefix 通信去重”和“backend 物理 K 不复制”是两件事：
前者已经实现，后者只有在 cuDNN 提供等价的 fragment-prefix reuse ABI 后才能删除，不能通过只改
All2AllV 假装消失。structural cold plan 对每个 rank 固定记录 `unique/packed/duplicate` 三个计数，
但一期不把它们写入 objective；正式 profile 单独报告 `k_pack` 次数、字节和 GPU 时间，作为二期
cost 校准依据。

**为什么必须搬 K 而不是搬 Q**：indexer 要对全部前序 entry 打分才能选 top-$\kappa$，若不实现 distributed TopK merge，则每个 rank 必须持有完整 required 前缀。这从架构上锁定了方向。量级上也一致：Q 侧是 $n_h=128$ 头，KV 侧是 MQA 共享 1 份，搬 Q 贵两个数量级。

### 5.2 压缩域降低前缀通信，但必须按 BF16 实算

前缀在**压缩域**传输：

| 通路 | Pro BF16 载荷 | 折算/作用域 |
|---|---|---|
| CSA `COMPRESSED_KI` | $128\times2=256$ B / entry | $256/r=64$ B / 原 token |
| CSA/HCA `COMPRESSED_KV` | $512\times2=1024$ B / entry | CSA $256$ B / 原 token；HCA $8$ B / 原 token |
| `WINDOW_KV` | $512\times2=1024$ B / token | 只传每个 Query fragment 所需窗口并去重 |
| `OVERLAP_X` | $7168\times2=14336$ B / support token | 只传压缩边界缺失的 hidden，不传完整前缀 |

压缩前缀仍显著小于传输完整原 token 前缀，但旧版按 FP8 推出的“固定便宜 $7\times$”不属于
当前 BF16 训练合同。`OVERLAP_X`、窗口碎片数、split size 和 D2D pack 都会改变真实通信成本，
必须在 B300 profile 中逐 route 计时，不能只凭带宽公式将通信忽略。

### 5.3 CSA 四条、HCA 三条 typed route

```
source/query-layout token
  → required-row 去重与 destination-major pack
  → NCCL All2AllV
  → consumer physical layout（优先直接作为 backend 输入）
  → reverse All2AllV + CSR reduction
```

每条 route 都有独立 global-to-local map、device collective state 和反向 CSR；KI/KV 不要求拥有相同
physical row order。固定 route 集合和发起顺序为：

```text
CSA forward:  WINDOW_KV → OVERLAP_X → COMPRESSED_KI → COMPRESSED_KV
CSA backward: COMPRESSED_KI → COMPRESSED_KV → OVERLAP_X → WINDOW_KV

HCA forward:  OVERLAP_X → WINDOW_KV → COMPRESSED_KV
HCA backward: COMPRESSED_KV → WINDOW_KV → OVERLAP_X
```

代表性 Pro-pair profile 必须按真实执行 DAG 区分“可遮挡”与“依赖受限”，不能要求每条 route
都出现没有因果依据的通信/计算交叠：

| mode / direction | `overlap_capable` | `dependency_bound` |
|---|---|---|
| CSA forward | `WINDOW_KV`、`OVERLAP_X`、`COMPRESSED_KI`、`COMPRESSED_KV` | 无 |
| CSA backward | `COMPRESSED_KI`、`COMPRESSED_KV`、`OVERLAP_X`、`WINDOW_KV` | 无 |
| HCA forward | `WINDOW_KV`（与 Main Compressor forward） | `OVERLAP_X`（Main Compressor 的 support 前置依赖）、`COMPRESSED_KV`（Main Compressor 的输出且 attention 前必须完成） |
| HCA backward | `WINDOW_KV`（与 Main Compressor backward） | `COMPRESSED_KV`（Main Compressor backward 的梯度前置依赖）、`OVERLAP_X`（消费 support gradient 的末端 route） |

`overlap_capable` 表示 DAG 中存在合法的同 mode、同 direction 独立计算窗口，不表示每个
rank/step 的实测交集必须严格大于 0；正 overlap 时长与比例当前均为 report-only，不设硬门槛。
`dependency_bound` 同样只记录通信时长、实际 overlap 和上述依赖原因。所有 route 与另一 mode
的 compute 交集仍必须为 0；NCCL、`DsaRowCopy`、`DsaRowCsrReduce` 不能充当 compute。route 次数、
归属和固定发起顺序仍是结构性硬门槛。

CSA backward 不能依赖 autograd ReadyQueue 恰好把 Window 排在最后：multi-output late join 先释放
projection/support 梯度，随后 branch-order gate 在同一 route stream 上显式提交
`OVERLAP_X → WINDOW_KV`，并把 owner-local 梯度接回原始 source。这样完整 4B 顺序是可执行因果合同，
而不是仅由 profile 事后观察到的顺序。

`COMPRESSED_KI` 只送 Indexer K；`COMPRESSED_KV` 只送主 attention KV；`WINDOW_KV` 送未压缩
窗口 KV；`OVERLAP_X` 送 compressor support hidden。它们可以共享 MSA 的 required-range/consumer
去重思想，但不能合并 tensor schema 或假设 map 相同。

反向对称 group-reduce：跨 consumer 的梯度按预计算 CSR 累加回 producer/source row。BF16 输入的
归约使用 FP32 accumulator，再在输出边界转回目标 dtype；是否增加 deterministic 模式由 correctness
和实测决定，不把未冻结的 atomic 行为写入 ABI。

### 5.4 旧分析估算不是验收证据

下表是旧版在 $T=1\text{M}$、$P=64$、FP8 载荷假设下的分析估算，只用于说明 native-area
布局值得先实现，不能作为 Pro BF16 训练通信占比或 overlap 的实测证据：

| 场景 | 布局 | $E$ | 计算 ms | 通信 ms | 占比 | 耗时 ms |
|---|---|---|---|---|---|---|
| 单条 1M | zigzag / min-heap | 1.0001 | 3532 | 13.2 | 0.4% | **3532** |
| | sequential | 1.8925 | 6684 | 13.2 | 0.2% | 6684 |
| lognormal $\sigma{=}1.5$ | **min-heap** | **1.0017** | **501** | 8.2 | 1.6% | **501** |
| | min-heap $g{=}4$ | 1.0127 | 507 | 4.0 | 0.8% | 507 |
| | zigzag | 1.6867 | 844 | 2.5 | 0.3% | 844 |
| | sequential | 2.3931 | 1197 | 2.4 | 0.2% | 1197 |
| $1{\times}512\text{K}{+}\cdots$ | **min-heap** | **1.0006** | **1150** | 9.8 | 0.8% | **1150** |
| | zigzag | 1.6812 | 1932 | 6.9 | 0.4% | 1932 |

### 5.5 一期不加局部性旋钮，但不忽略通信

min-heap 会把 chunk 打散，导致每个 rank 触及大量 doc、且每个 doc 内都有深 fragment，$V_p$ 趋近 $T$（退化成 all-gather）。可以引入超块参数 $g$（$g$ 个连续 chunk 打包后再跑 min-heap）来换局部性，但：

| 旧估算 | MSA | DSA |
|---|---|---|
| 通信占计算 | **17.8%** | **1.6%** |
| 旧分析建议 | $g{=}4$ | $g{=}1$ |

$$\boxed{g=1（原生\ \text{min-heap}）,\ 不引入局部性目标}$$

一期使用 $g=1$ 原生 MinHeap，不预先增加未经实测的局部性目标；这只是初始策略，不代表通信
可以忽略。若正式 profile 显示 required-range volume、split count、D2D pack 或 NCCL 成为关键路径，
二期应把实测通信项并入 cost 或引入超块，而不是维护“通信必然是噪声”的结论。

**但仍建议实现 launch/wait 分离的 scheduled API**（照 MSA 的 `calc_msa_scheduled`）：单条 1M 时通信绝对值是 $13$ ms/前向，够大到值得藏；且带宽相对算力若恶化，$g$ 作为储备旋钮可随时启用。

---

## 6. 优化目标与求解

### 6.1 目标

每层结束都有 group-cast / group-reduce 与 MoE all-to-all，CP ranks 逐层对齐，故总时间是"每层最慢 rank"之和：

$$\text{Total}=H_{\mathrm{CSA}}\max_pC_p+H_{\mathrm{HCA}}\max_pB^{\mathrm{L}}_p$$

形式上是双目标 makespan，但由 §3.2 的共线性可退化为单标量：

$$\boxed{\min_{\{y_{cp}\}}\ \max_p\sum_{c\in\Pi_p}w_c},\qquad
E=\frac{\max_p\sum_{c\in\Pi_p}w_c}{\overline{\textstyle\sum_{c\in\Pi_p}w_c}}\ \ge1$$

### 6.2 算法

`MinHeapDispatchAlg`：按 $w_c$ 降序，弹出负载最小且未满（上限 $\lceil n/P\rceil$）的桶。$O(n\log n+n\log P)$。非最优（$P{=}2$、$[8,7,6,5,4,2,2,2]$ 得 $19/17$，最优 $18/18$），但 $n/P\ge16$ 时残差 $<0.3\%$。

**曾评估并否决：向量化双目标 min-heap**（贪心最小化 $\sum_\ell\max_p$ 的增量）。自由度充足时无空间可捡；自由度不足时该贪心过于短视（拖延重 chunk、被逼入死角），反而更差：

| 场景 | $n/P$ | 单标量 | 双目标 |
|---|---|---|---|
| $P{=}64$ | 32 | 1.0006 | 1.0006 |
| $P{=}256$ | 8 | 1.0322 | 1.0328 |
| $P{=}512$ | 4 | **1.1130** | **1.1577** |

---

## 7. 计算均衡实测

$T=1\text{M}$，$b=512$，$P=64$：

| batch 构成 | sequential | 全局 zigzag | 逐 doc zigzag | min-heap |
|---|---|---|---|---|
| 单条 1M | 1.8925 | 1.0001 | 1.0001 | 1.0001 |
| $2048\times512$ 等长 | 1.0000 | 1.0000 | 1.1106 | 1.0000 |
| $128\times8\text{K}$ 等长 | 1.0000 | 1.0000 | 1.2263 | 1.0000 |
| $32\times32\text{K}$ 等长 | 1.1609 | 1.0402 | — | 1.0003 |
| lognormal $\sigma=1.0$ | 1.5549 | 1.4287 | 1.0647 | 1.0019 |
| lognormal $\sigma=1.5$ | 2.4684 | 1.7206 | 1.0329 | 1.0014 |
| lognormal $\sigma=2.0$ | 2.8165 | 1.8302 | 1.0130 | 1.0010 |
| $1\times512\text{K}+4\times64\text{K}+\cdots$ | 3.0310 | 1.6812 | 1.0068 | 1.0006 |

### 7.1 收益是长度方差的函数，与长度本身无关

等长时全局 zigzag 仍精确。设每 doc $q$ 个 chunk、$n=Sq$，$j(i)=i\bmod q$，则

$$j(n-1-i)=(Sq-1-i)\bmod q=q-1-j(i)$$

doc 内位置**恰好也翻转**，每对 $\sum j$ 为常数——锯齿波在周期内自我对称。长度一变，周期错位，对称性失效。

$$\boxed{\text{zigzag 正确}\iff\text{长度分布单一}}$$

### 7.2 凹性效应可被分离

$32\times32\text{K}$ 等长按层类型分解：HCA 层 $1.0010$（线性、精确），CSA 层 $1.0479$（$\min$ 的凹性 / Jensen）。理论预测的分量间冲突真实存在，量级 $4\%$，被 §3.2 的等 chunk 数约束吸收。

### 7.3 与 Megatron-LM 的对比

Megatron 有两条 CP 切分路径（`megatron/core/utils.py`）：

- **Pretrain**（`cu_seqlens is None`）：全局 zigzag，`index = [cp_rank, 2·cp_size−cp_rank−1]`
- **SFT / packed**：TE `thd_get_partitioned_indices`，**逐 doc** zigzag（每 doc pad 到 $2P$ 倍数后切 $2P$ 份）

varlen 的真实 baseline 是逐 doc zigzag：

| | 逐 doc zigzag | min-heap |
|---|---|---|
| 需 padding | 每 doc 补到 $2P$ 倍数（$P{=}512$ 时 **4.8%**） | 无 |
| 碎片 / 卡 | $2S$（$70\!-\!4096$） | $n/P$（$16\!-\!32$） |
| 等长分布 | **退化到 1.23**（各 doc 偏置同相叠加，不相消） | 1.0000 |
| 大 CP（$P{=}512$，$b{=}512$） | 1.0862 | 1.1021（需按 §4.3 调 $b$） |

$$\boxed{\text{优势}=\text{免 padding}+\text{碎片少一个数量级}+\text{不依赖长度分布}}$$

而**不是**"均衡度碾压"。

---

## 8. 反向

三个分量的反向 / 正向 matmul 比值一致（core：$2\to4$；indexer：$1\to2$；HCA：$2\to4$），因此正向的权重比例在反向原样成立，同一划分对反向同样最优。$dKV$ 与正向 group-cast 走同一 route，通信对称。

已冻结项与剩余风险如下：

1. **aux loss 覆盖全部 Query。** 每个本地 Query 都进入 selected-KL，且保留
   Indexer→`qr/x` 的梯度链；不得采样 Query、在 runtime 中 `detach`，或只恢复部分 Query。
2. **$dKV$ 的跨 consumer 原子竞争。** "热门" entry 被大量 query 选中时 scatter-add 会串行化。这是内存系统效应，任何面积模型都看不见；热门度分布依赖 indexer 的运行时输出。
3. **压缩器的反向。** $d(\text{entry})$ 要经压缩器投影散回原始 token；跨 rank support 经
   `OVERLAP_X` reverse + CSR reduction 返回，不能假定始终本地完成。

验证方式：正式 profile 对代表性 CSA+HCA 连续跑 5 轮 forward/backward；按 rank、mode、step
分别统计 Indexer score/top-k、FlashMLA forward、cuDNN backward、Compressor 和全部 route/D2D。
若实测不平衡，再按 **forward+backward 合并时间**校准 cost；不能用五轮平均掩盖单步离群。
固定 Pro backend 的 FlashMLA forward 不是旧 generic variant：SM103 上 `h_q=128`、`d_qk=512`，
且 CSA/HCA 的 sparse width 都不超过 1280，`b7643bd...` 的官方 dispatch 因而选择
`Fwd_Sm100_Head128_Small_TopK_Impl`。正式汇总必须对 CSA 与 HCA 的每个 rank/step 精确匹配一次
`sparse_attn_fwd_for_small_topk_kernel`，两种 mode 的名称必须一致；不得用 `sparse_attn_fwd` 等
模糊 substring 接受另一 variant。

---

## 9. 落地路线

模块边界照 MSA 的职责划分，但落在当前 Magi-DSA 目录，不另造一套 extension：

```
magi_attention/
    dsa_config.py                  Pro 层序、尺寸与 structural schema
    meta/solver/dsa_solver.py      shared Query layout 与 typed route cold plan
    functional/dsa_packing.py      device map、consumer layout 与 backend metadata
    functional/dsa_comm.py         typed All2AllV、reverse CSR、launch/wait
    functional/dist_dsa.py         CSA/HCA autograd 与 scheduled overlap
    functional/dsa_backend.py      FlashMLA/cuDNN thin wrapper
    dsa_runtime_mgr.py             单 ratio parameter-free runtime
    dsa_pro_runtime_mgr.py         61 层共享 layout 与 CSA/HCA handle bundle
    dsa_layer.py                   61 份独立模型参数 owner
```

相对 MSA 需要新增的四项：

| # | 内容 | 位置 |
|---|---|---|
| 1 | **两种层类型**：一份 shared Query layout，两份 ratio-specific route/device plan；不得产生两套 Query 排列 | `dsa_solver.py` / `dsa_pro_runtime_mgr.py` |
| 2 | **Q/K 不等长同源分支**：从 sample-relative Query fragments 分别导出 r4/r128 required rows，不做整数除法猜边界 | `dsa_solver.py` |
| 3 | **压缩器置于 compressed-K route 之前**；跨边界输入由 `OVERLAP_X` 获取，不依赖隐式对齐 | `dist_dsa.py` |
| 4 | **typed routes**：窗口、support hidden、compressed KI/KV 分离，保持固定 collective 顺序 | `dsa_comm.py` |

不需要做的：新 cost 模型（§3.2）、新 dispatch 算法（§6.2）、双目标求解器（§6.2）、局部性旋钮（§5.5）、逐层重排。

实施不冻结为 CSA-first 或 HCA-first。两种 mode 共用 Query layout，但各自保留数学必要 route；
以能最早建立端到端 correctness 与可归因 profile 的依赖顺序推进。

---

## 10. 设计边界

- $\alpha_{\mathrm{core}}:\alpha_{\mathrm{idx}}=16:1$ 是 FLOPs 比而非时间比。小维度 BF16 indexer、
  gather 后访存不规则的 MQA、短 KV 的稠密 HCA，三者算术强度差异极大。实测校准优先于写死
  B300 专用经验权重。
- top-$\kappa$ 的实际选中数是运行时量，$D$ 只有上界可静态计算。由 §3.2，不影响均衡质量。
- 重计算 / 激活 checkpoint 策略对 $w_c$ 的影响未建模。
- 动态 top-$\kappa$ 路由本身的通信不确定性（若未来改为分布式 TopK merge，§5.1 的方向锁定失效，需重新评估）。
- FFA / sparse kernel 的 tile 调度对 slice 数量的敏感性。
- CP size 对计算效率的影响 $\eta(P)$。
- 不支持 CUDA Graph capture 与 higher-order gradient（沿用 MSA 的限制）。

---

## 11. 结论摘要

$$\boxed{\text{收益}\approx\big(E^{\text{baseline}}-1\big)\times\frac{\text{attention FLOPs}}{\text{总 FLOPs}}}$$

第二项随上下文长度增长——短上下文 MoE 主导、失衡被摊薄；1M 上下文时 attention 占大头，失衡近乎全额传导。

1. **一期不写 cost 模型。** native 因果面积与三分量的差距是 0.1%（§3.2）。这是相对原计划最大的一处减法。
2. **`MAGI_ATTENTION_MIN_CHUNKS_PER_RANK=16`。** 机制已存在，默认 8 留 1–3% 失衡。
3. **通信照 MSA 做 required-entry/required-row + consumer 内去重**，但 CSA 4 路、HCA 3 路
   分开建图；官方 cuDNN grouped Indexer 暂无 MSA 的 `fragment_indices` 复用 ABI，故 unique KI bank
   后仍保留一次可归因的 grouped `k_pack`。BF16 下不沿用旧 FP8 的固定 $7\times$ 结论。
4. **一期不引入局部性旋钮。** $g=1$ 是起始策略；是否加入通信 cost 由真实 route/D2D profile 决定。
5. **相对 Megatron 的优势是免 padding + 碎片少一个数量级 + 不依赖长度分布**，而非均衡度碾压。
6. **收益区间 = 长上下文 $\times$ 长度混合。** 缺一个则与 baseline 打平。
7. **单 doc 也使用同一可审计 MinHeap 路径。** 是否增加 zigzag 快路径只能由 cold-plan 开销和
   相同 layout/correctness 证据另行决定，当前不做隐式切换。
8. **aux loss 覆盖全部 Query**；剩余反向未知主要是 selected-KV 热点、CSR/atomic 竞争和实际
   backend tile 效率，必须由五轮 CSA+HCA forward/backward profile 验证。

## 12. 官方 backend 冻结

- 模型与 attention 公式来自
  `DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1`。
- FlashMLA 以 `main@9241ae3ef9bac614dd25e45e507e089f888280e0` 为基线，依次审核回移
  `13d173ac48abd8ec88a4e742bcaaa59c5ccf4ece` 的 dual-LSE 与
  `b7643bd54521f563b839b98289b5cd048c062ba2` 的 H128/prefix-1024 增量。同一次 CSA sparse
  forward 接收 `[1024 compressed, 128 window]` indices，并返回 full sparse LSE 与 compressed-prefix
  per-head LSE，不增加独立 prefix-LSE recompute kernel。该增量在 Pro H128/D512、sparse width
  `<=1280` 下固定发射 `sparse_attn_fwd_for_small_topk_kernel`；CSA 的 `indexer_topk=1024` 与
  HCA 的无 prefix-LSE 调用都必须命中这个 exact variant。
- cuDNN backend 冻结 `9.24.0.43`，frontend 冻结
  `v1.26.0@35fd7b0d0e1d4952b904c79341c5e84e3af0a328`。H128/D512/top-k1152 sparse backward 和
  H64/D128/top-k1024 Indexer 必须通过 B300 专项 preflight/correctness 后，才能标记为运行已验证。
