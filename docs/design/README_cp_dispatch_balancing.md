# [历史审计] 如何同时满足 CSA 和 HCA 负载均衡

> **状态：2026-08-03 Flash-Base / `shared_greedy` 历史审计稿，不是当前设计或发布合同。**
>
> 文中“最终算法”、“release 默认”、Flash-Base 尺寸、`shared_greedy`、W/CSA/HCA
> attention-suite 和对应 artifact，只表示 2026-08-03 当时的实现与实测结论。它们已被
> DeepSeek-V4-Pro 的 `structural_balanced` 共享 Query layout 取代，不得用来恢复当前 policy。
>
> 当前 Pro 的唯一事实源为
> [`magi_dsa_v4_design.md`](./magi_dsa_v4_design.md) 与
> [`README_dsv4_cp_dispatch_structural_balancing.md`](./README_dsv4_cp_dispatch_structural_balancing.md)。
> 本文仅保留 `shared_greedy` 决策、CP8 natural correctness、8×B300 profile 和消融数据，
> 供历史差异与回归审计。

## 1. 问题定义

系统包含一个大小固定的 CP group，共有 \(P\) 个 rank。

当前 pack 包含 \(m\) 个样本：

$$L_1,L_2,\ldots,L_m$$

总 Query 数为：

$$N=\sum_{i=1}^{m}L_i$$

样本 \(i\) 内的 Query 位置为：

$$p=0,1,\ldots,L_i-1$$

source layout 已知。本文要为每个 Query 决定最终 owner，并让 W、CSA 和 HCA 共用这份 Query layout。

当前模型只考虑：

- CSA Indexer score、Top-K 和 KI prefix pack；
- HCA sparse-attention backward、`COMPRESSED_KV.backward` 和 owner CSR reduce；
- CSA KI 峰值显存。

其中，Query 唯一分配、layout 合法性和显存是硬约束；CSA 和 HCA 的负载均衡是分层优化目标。
不要求各 rank 的 Query 数做 \(\lfloor N/P\rfloor/\lceil N/P\rceil\) 均分；token 数不均可以存在，只要硬约束满足且目标更优。
W 暂不进入目标函数，只做共享 layout 下的 non-regression 检查。
所有 \(T\) 函数使用同一时间单位，并由固定 backend 和目标机器的 profile 校准。

## 2. 决策变量

对 \(i=1,\ldots,m\)，\(p=0,\ldots,L_i-1\)，\(r=0,\ldots,P-1\)，定义 Query 分配变量：

$$x_{i,p,r}\in\{0,1\}$$

其中 \(x_{i,p,r}=1\) 表示样本 \(i\) 的 Query \(p\) 被分配给 rank \(r\)。

在唯一分配约束（§4.1）下，Query owner 良定为：

$$o_x(i,p)=r\quad\Longleftrightarrow\quad x_{i,p,r}=1$$

\(x\) 决定的是 sample-relative Query owner；`TOKEN_LAYOUT` 是它到 source token 的全局双射。

### 2.1 Final fragment

分配完成后，同一 sample 内属于同一 rank 的极大连续区间称为 final fragment，记为 \(f=(i,\ell,u)\)：\(i\) 为 sample，\(\ell\) / \(u\) 为 sample 内 Query 的起止下标（半开区间\([\ell,u)\)，含 \(\ell\) 不含 \(u\)）。rank \(r\) 的集合记为 \(\mathcal F_r(x)\)。

atom / band 只是 solver 分配时的搬动单位，用来压缩搜索空间；大小由
`MagiDSAConfig.indexer_atom_size` 显式配置，默认值和本文冻结实验值均为 128，并非代码硬编码。
分完后相邻同-owner band 合并为 final fragment。KI pack 与显存按 \(\mathcal F_r(x)\) 计算，
不按未合并的 band 计算。

## 3. Cost 模型

### 3.1 CSA Indexer Cost

CSA compression ratio 为 4。Query \(p\) 可见的 compressed-K 行数为：

$$v(p)=\left\lfloor\frac{p+1}{4}\right\rfloor$$

根据当前 cuDNN tile，score 和 Top-K 的工作量代理为：

$$c_{\mathrm{score}}(p)=\max\left(128,128\left\lceil\frac{v(p)}{128}\right\rceil\right)$$

$$c_{\mathrm{topk}}(p)=\max\left(256,256\left\lceil\frac{v(p)}{256}\right\rceil\right)$$

rank \(r\) 的 score 和 Top-K 工作量为：

$$S_r(x)=\sum_{i=1}^{m}\sum_{p=0}^{L_i-1}x_{i,p,r}\,c_{\mathrm{score}}(p)$$

$$K_r(x)=\sum_{i=1}^{m}\sum_{p=0}^{L_i-1}x_{i,p,r}\,c_{\mathrm{topk}}(p)$$

grouped Indexer 会为每个 final fragment 单独展开 causal KI prefix。对 \(f=(i,\ell,u)\in\mathcal F_r(x)\)，展开长度为 \(\lfloor u/4\rfloor\)（与 \(\ell\) 无关：前缀从 sample 起点到 fragment 终点）。因此 rank \(r\) 的 packed KI 行数为：

$$U_r(x)=\sum_{(i,\ell,u)\in\mathcal F_r(x)}\left\lfloor\frac{u}{4}\right\rfloor$$

使用固定 backend profile 将三项工作量转换为预测时间：

$$C_r^{I}(x)=T_{\mathrm{score}}(S_r(x))+T_{\mathrm{topk}}(K_r(x))+T_{\mathrm{KIpack}}(U_r(x))$$

CSA Indexer makespan 为：

$$\boxed{I(x)=\max_{r=0,\ldots,P-1}C_r^{I}(x)}$$

由于 \(U_r\) 会随 fragment 数变化，fragment cost 已包含在第一层目标中，不能仅作为最后的 tie-break。

### 3.2 HCA Backward Cost

对 HCA Query 位置 \(p\)，attention 可见 K 行数为：

$$k_{\mathrm{HCA}}(p)=\min(p+1,128)+\left\lfloor\frac{p+1}{128}\right\rfloor$$

当前 64-K tile 下，sparse-attention 的工作量代理为：

$$c_{\mathrm{HCA}}(p)=\left\lceil\frac{k_{\mathrm{HCA}}(p)}{64}\right\rceil$$

该代理同时刻画 HCA sparse-attention 的 forward/backward Query 侧计算量；显式目标取 backward critical path（见下），因为它叠加了 reverse All2AllV 的 arrival skew 与 owner CSR。

rank \(r\) 的 HCA attention 工作量为：

$$Q_r^{H}(x)=\sum_{i=1}^{m}\sum_{p=0}^{L_i-1}x_{i,p,r}\,c_{\mathrm{HCA}}(p)$$

HCA compression ratio 为 128。样本 \(i\) 的完整 compression block 数为：

$$B_i=\left\lfloor\frac{L_i}{128}\right\rfloor$$

记 block \(b=(i,j)\)，\(j=0,\ldots,B_i-1\)，其结束位置为：

$$e_b=128(j+1)$$

block producer 是该 block 最后一个 Query 的 owner：

$$\pi_b(x)=o_x(i,e_b-1)$$

定义 consumer 指示变量：

$$y_{b,r}(x)=\mathbf 1\left[\sum_{p=e_b-1}^{L_i-1}x_{i,p,r}>0\right]$$

\(y_{b,r}=1\) 表示 rank \(r\) 的 causal compressed-prefix union 包含 block \(b\)。

在 `COMPRESSED_KV.backward` 中，consumer rank \(r\) 向 producer rank \(s\) 发送的行数为：

$$n_{r,s}(x)=\sum_{i=1}^{m}\sum_{j=0}^{B_i-1}y_{(i,j),r}(x)\,\mathbf 1[\pi_{(i,j)}(x)=s]$$

完整逐 peer route 矩阵为：

$$\mathbf N(x)=\left[n_{r,s}(x)\right]_{P\times P}$$

rank \(r\) 到达 reverse All2AllV 的时间为：

$$a_r(x)=T_{\mathrm{HCA\text{-}bwd}}(Q_r^{H}(x))+T_{\mathrm{pack}}\left(\sum_{s=0}^{P-1}n_{r,s}(x)\right)$$

arrival vector 为：

$$\mathbf a(x)=\left[a_0(x),a_1(x),\ldots,a_{P-1}(x)\right]$$

rank \(r\) 完成 reverse All2AllV 的时间为：

$$e_r^{H}(x)=T_{\mathrm{collective},r}\left(\mathbf a(x),\mathbf N(x),512,\mathrm{topology}\right)$$

其中 \(a_r\) 和 \(e_r^{H}\) 都相对于同一个 HCA backward 起点计时；512 是 HCA compressed-KV 的 BF16 row width。

block \(b\) 的 gradient fan-in 为：

$$d_b(x)=\sum_{r=0}^{P-1}y_{b,r}(x)$$

当前问题对应的 HCA backward critical-path cost 为：

$$C_r^{H}(x)=e_r^{H}(x)+T_{\mathrm{CSR}}\left(\{d_b(x)\mid\pi_b(x)=r\}\right)$$

HCA makespan 为：

$$\boxed{H(x)=\max_{r=0,\ldots,P-1}C_r^{H}(x)}$$

不能直接把逐 rank NCCL kernel duration 当成 HCA 通信负载。冻结 profile 中，sparse-attention backward 从约 3.0 ms 增长到 10.8 ms，而 CKV NCCL duration 从约 9.5 ms 反向下降到 0.016 ms，且各 rank 几乎同时结束。这说明当前阶梯主要包含 collective arrival skew。

### 3.3 `shared_greedy` v1 的整数代理

首版代码把上述 lookup 固定为可复现的整数 tick；版本号为
`b300_sm103_structural_proxy_v1`：

```text
C_I(r) = 8 * S_r + K_r + 32 * U_r

C_H(r) = 256 * Q_H_r
       + 2048 * (hca_send_rows_r + hca_recv_rows_r)
       + 4096 * hca_max_peer_rows_r
```

`hca_max_peer_rows_r` 同时检查 route matrix 的第 `r` 行和第 `r` 列。这里的 tick 只用于同一 pack
内的确定性候选排序，不解释为 GPU ns，也不替代 Profile。权重冻结在 plan 的
`cost_model_version` 中；若 Profile 证明排序与实测相反，必须发布新版本模型，不能原地改变同名
版本的含义。

### 3.4 `qlen < 4096`

当 \(L_i<4096\) 时，\(c_{\mathrm{HCA}}(p)\) 退化为一张小表；表中区间只取实际存在的 \([0,L_i)\) 部分：

| Query 位置 | \(c_{\mathrm{HCA}}(p)\) |
|---|---:|
| \(p=0,\ldots,63\) | 1 |
| \(p=64,\ldots,126\) | 2 |
| \(p=127,\ldots,L_i-1\) | 3 |

因此，少量短样本仍可能因为 sample 前段和后段分配不均而出现 HCA imbalance。

同时：

- \(L_i<128\) 时 \(B_i=0\)，没有 HCA compressed block；
- \(128\le L_i<4096\) 时有 1 到 31 个完整 block。

不需要为 `qlen < 4096` 单独设计算法；上述 cost 会根据实际 `cu_seqlens` 自动进入同一个优化问题。

## 4. 约束条件

### 4.1 Query 唯一分配

每个 Query 必须且只能分配给一个 rank：

$$\boxed{\sum_{r=0}^{P-1}x_{i,p,r}=1,\qquad \forall i=1,\ldots,m,\ \forall p=0,\ldots,L_i-1}$$

由此自动得到：每个 sample 的 Query 区间无重复、无遗漏。

### 4.2 Layout 合法性

最终分配必须满足：

- final fragment 不跨 sample（由 §2.1 定义直接保证）；
- `TOKEN_LAYOUT` 是 source token 到 Query owner 的全局双射；
- W、CSA 和 HCA 的 route 收发计数对称；
- compression block、尾块和 sentinel 语义保持不变。

**明确排除：** 不把

$$\sum_{i,p}x_{i,p,r}\in\bigl\{\lfloor N/P\rfloor,\lceil N/P\rceil\bigr\}$$

列为硬约束。当前 CSA `ratio=4` 实现仍强制 exact token capacity；本设计模型不继承该约束。

### 4.3 CSA KI 显存约束

rank \(r\) 的 KI 峰值显存为：

$$M_r^{\mathrm{KI}}(x)=2\,\mathrm{bytes}\times128\times U_r(x)+M_r^{\mathrm{map}}(x)+M_r^{\mathrm{workspace}}(x)$$

要求：

$$\boxed{M_r^{\mathrm{KI}}(x)\le M_{\mathrm{budget}}^{\mathrm{KI}},\qquad \forall r}$$

满足全部硬约束的分配集合记为：

$$\mathcal X$$

若 \(\mathcal X=\varnothing\)，solver 必须报告 infeasible，不能静默放宽约束。

方案一代码对这条约束使用可逐项核查的布局相关字节账本。令 `V_r` 为 rank `r` 收到的 unique KI
行数、`F_r` 为 final fragment 数、`N_r` 为 Query 数、`B` 为 pack 的全局 CSA compressed block
数，则：

```text
grouped_ki_bytes = 2 * 128 * U_r

int32_map_bytes = 4 * (
    2 * U_r             # k_pack.source_rows + k_unpack.source_rows
  + (V_r + 1)           # k_unpack.row_offsets
  + B                   # ki_global_to_consumer
  + (3 * F_r + 2)       # q/k cu_seqlens + q_causal_offsets
  + 2 * N_r             # q_sample_block_offsets + seq_lens
)

M_modeled_r = grouped_ki_bytes
              + int32_map_bytes
              + ki_workspace_reserve_bytes
```

grouped Indexer selection 是 `no_grad`，因此账本不虚构一个同形 grouped-K backward gradient。
backend scratch、allocator headroom 和尚未按布局参数化的临时量由调用者显式提供
`ki_workspace_reserve_bytes`。冻结 128K 实验配置暂取 1 GiB budget、256 MiB reserve；两者都会进入
plan hash 和 Profile metadata。该值在实测峰值显存完成前不是新的 release 默认值。

## 5. 优化目标

第一层先求 CSA Indexer 的最优值：

$$I^*=\min_{x\in\mathcal X}I(x)$$

Indexer 最优解集合为：

$$\mathcal X_I^*=\{x\in\mathcal X\mid I(x)=I^*\}$$

第二层只在 Indexer 最优解中优化 HCA：

$$H^*=\min_{x\in\mathcal X_I^*}H(x)$$

等价的总公式为：

$$\boxed{\underset{x\in\mathcal X}{\operatorname{lexmin}}\left(I(x),H(x)\right)}$$

若 \(|\mathcal X_I^*|>1\)，第二层利用 Indexer 最优解中的自由度改善 HCA；若 \(|\mathcal X_I^*|=1\)，严格分层求解没有剩余调整空间。

当前模型不使用未定义的门槛或 slack。只在 \(I(x)\) 和 \(H(x)\) 都相同时，才进一步选择 `TOKEN_LAYOUT` 通信更少、final fragment 更少的方案。

## 6. 冻结求解方案：确定性贪心 + 4 轮局部改善

当前只保留这一种求解方案。它不使用随机数、加权和或未登记的 fallback；完整可行解统一按下列 key
严格词典序比较：

```text
key(x) = (I(x), H(x), R(x), F(x))
```

也就是只有 `I` 相同时才比较 `H`，只有 `I/H` 都相同时才比较 `R/F`。时间 lookup 应输出整数
tick（例如 ns），避免用浮点误差判断 `I` 是否相同。

### 6.1 Band、final fragment 与不可行处理

1. 每个 sample 按 `MagiDSAConfig.indexer_atom_size` 切成不跨 sample 的 band，尾 band 可以更短。
   字段默认值为 128，本文所有 128K correctness/Profile 也显式使用 128；v1 算法本身不要求该字段
   永远等于 128。
2. band 只是搜索移动单位。每次试探和完整解评估都先合并相邻同-owner band，再按 final fragment
   重算 `U_r`、KI modeled bytes、route 和 fragment 数；不能把 band cost 直接累加成最终 cost。
3. 每个候选都检查 §4 的全部硬约束。KI modeled bytes 超过预算的候选立即丢弃，不使用惩罚项软化。
4. 当前 v1 **没有 band 对半拆分逻辑**。若某个 band 对所有 rank 都超过 KI 预算，构造立即抛出包含
   `shared_greedy found no feasible rank for band`、sample id 和 band 区间的 `RuntimeError`；其语义是
   `no_feasible_solution_found`，不是整个优化问题数学上 `infeasible` 的证明。由于没有生成 plan，
   该错误也不是 plan 的 `stop_reason`。只有硬约束下界、穷举或小规模 exact solver 才能证明
   `infeasible`。未来若要加入动态拆分，必须同步修改 solver、plan hash、测试和本文，不能只改文档。

### 6.2 确定性贪心构造

band 排序使用与 v1 cost model 一致的独立整数权重；它只决定构造顺序：

```text
band_weight(b) = 8 * score_cost(b)
               + topk_cost(b)
               + 32 * floor(b.q_end / 4)
```

真正放置时仍必须用已分配 band 合并后的 final fragment 重算试探 `I` 和 KI 显存，不能直接累加
`band_weight`。

1. 按 `indexer_weight`、score cost、Top-K cost 和 band 长度从大到小排序；再以 sample id、sample
   内起点和原始序号作确定性 tie-break。
2. 对当前 band 试探全部 rank，过滤超 KI 预算候选，再按以下构造 key 选择最小者：

```text
greedy_key = (
    max_rank_indexer_cost,
    max_rank_modeled_ki_bytes,
    total_fragment_count,
    destination_rank_indexer_cost,
    destination_rank_id,
)
```

每个候选先执行 1 GiB 硬约束过滤，再按上述 key 比较。`max_rank_modeled_ki_bytes` 是固定的第二排序项，
不提供关闭开关：硬约束只保证 plan 可行，该项用于在同等 Indexer cost 的可行候选中降低最坏 rank 的
显存占用。§8.8 记录了删除该项的一次性历史消融及最终裁决。

3. 全部 band 放置完成后，重算完整 `(I,H,R,F)`；若完整解反而超过预算，立即报错，不返回部分 plan。

### 6.3 4 轮局部改善上限（允许提前收敛）

`DsaSharedLayoutConfig.local_improvement_passes` 在当前选型中固定为 4；代码保留该字段是为了消融和小规模
测试，正式 4 轮配置不会从运行时隐式改变。每轮执行：

1. 先找 `indexer_cost` 等于当前最大值的瓶颈 rank，枚举其单 band move，以及对每个目标 rank 最多
   4 个近似等重 band swap；这一阶段只接受最大 Indexer cost 严格下降的候选。
2. 若没有 Indexer 改善，再对 `hca_cost` 最大的 rank 枚举同一邻域，只接受完整 `(I,H,R,F)` 严格
   变小的候选。
3. 在本轮全部合格候选中选择完整 key 最小者。若没有合格候选，以 `local_optimum` 提前结束；否则接受
   一步并进入下一轮。完成第 4 轮后以 `pass_limit` 停止。

单 band move 会改变相邻 owner boundary；swap 只比较每个目标 rank 中最接近当前 band Indexer/HCA
权重的 4 个候选，因此邻域规模有界且可解释。该算法不保证全局最优，但 §8 的 correctness、正式 Profile
和轮数消融已经证明 4 轮对当前 workload 有效。

### 6.4 缓存、输出与结果含义

相同 `cu_seqlens/source_counts/config` 的共享求解结果使用有界进程内缓存；W/CSA/HCA 仍分别构建各自
ratio-specific route，但不重复执行同一份 cold solver。warm execution handle 不运行 solver。

每次求解至少报告：`solver_scheme`、solver 配置、评估候选数、最终 `key`、各 rank 的
`C_r^I/C_r^H/M_r^KI`、fragment 数与停止原因。

算法没有随机 seed；plan 记录 `solver_scheme`、`cost_model_version`、`candidate_evaluations`、
`improvement_steps`、`stop_reason`、逐 rank cost/显存和 `query_layout_hash`。停止原因目前只有
`local_optimum` 与 `pass_limit`。

启发式输出的 `key(x)` 是当前已找到的词典序 incumbent，不声称已达到 \(I^*\) 或 \(H^*\)。\(I(x)\) 是
\(I^*\) 的上界；在未证明 \(I(x)=I^*\) 之前，\(H(x)\) 不能宣称是 \(H^*\) 的上界或下界。小规模 pack 可用穷举或
CP-SAT/MILP 作为测试 oracle，但它们不进入 production 默认求解路径。

## 7. 当前设计边界

当前模型尚未冻结：

- tile、row 和 route metadata 到实际 GPU 时间的完整校准函数；
- 完整 HCA forward/backward DAG 中的其他 routes 和合法 overlap；
- W/HCA forward 和 `TOKEN_LAYOUT` 边界的端到端 non-regression 上限；
- 非默认 `indexer_atom_size` 在大规模 ragged/multi-sample workload 上的系统 Profile。当前默认值和
  冻结证据均为 128，局部改善轮数固定为 4。

当前代码状态为：

- release 默认路径不变：`ratio=4/indexer_balanced` 使用原 128-token atom Indexer greedy，
  `ratio=0/128` 仍为 source-contiguous；
- 显式实验路径 `shared_greedy` 已实现 §6 的唯一 4 轮方案，不再把各 rank Query 数均分作为硬约束；
- W、CSA、HCA 在实验路径分别生成 ratio-specific plan/route，但必须具有完全相同的
  `query_layout_hash/query_token_counts`，三者都在 DSA 外执行同一 `TOKEN_LAYOUT`；
- HCA structural cost、KI 显存强约束、`(I,H,R,F)` 和求解诊断已进入 cold plan；
- CP8 natural correctness 和正式 B300 attention-suite 已通过；HCA backward 达到预期，HCA forward
  只达到“明显改善但仍有残余不均衡”，且 attention-suite capture 不包含 `TOKEN_LAYOUT`，因此该 policy
  仍不可替换 release 默认值。

## 8. 最终负载均衡与实验结果

### 8.1 可比性与产物

legacy baseline 与 shared-greedy 使用完全相同的 8×B300、BF16、单条 128K、CP8、固定 seed/input、
固定 parameter、5 个 captured attention-suite steps 和 CUPTI attribution 口径。两侧 source revision
均记录为 `0c4b270a9cef07ecea275958849d44bab059d8d2`，cuDNN/FlashMLA/CUTLASS DSL 版本与补丁完全相同；
输入、参数和 config SHA-256 逐项相同。唯一有意差异是 Query layout policy。

```text
legacy baseline:
artifacts/profile/20260724T050521Z-dsv4-flash-128k-attention-suite-precomputed-dout/

shared-greedy final:
artifacts/profile/20260803T031828Z-dsv4-flash-128k-attention-suite-shared-greedy-precomputed-dout/

shared-greedy repeatability run:
artifacts/profile/20260803T025307Z-dsv4-flash-128k-attention-suite-shared-greedy-precomputed-dout/

KI-memory sorting historical ablation:
artifacts/profile/20260803T094307Z-dsv4-flash-128k-attention-suite-shared-greedy-passes4-kimemhardonly-clock1800-attachwarm1-precomputed-dout/

CP8 natural correctness:
artifacts/correctness/20260803T024904Z-cp8-cp8-natural-backward/
```

最终 Profile 为 `PASS`：8 ranks × 5 steps 保持精确 8F+8B，8,000 个 kernel attribution coverage
为 1.0、unattributed 为 0，跨 mode kernel overlap 为 0；capture 内 `TOKEN_LAYOUT`、model projection
和 scalar loss 均为 0。CSA sequential shadow 的非 tie output max-abs 为 `9.765625e-4`，gradient
max-abs 为 `6.6036591e-8`，`latent_kv` gradient mismatch ratio 为 0。

### 8.2 Solver 输出

128K shared plan 的最终结果为：

```text
query_layout_hash = 56c222034fd9960d36d74ad363182ad22e4fd53a7cb2d330d1ff22effe763e46
query_token_counts = [16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384]
key = (2485300480, 48707584, 114560, 898)
candidate_evaluations = 26113
improvement_steps = 4
stop_reason = pass_limit
modeled_ki = 713.831--719.367 MiB/rank
```

W/CSA/HCA 和全部 rank 的 hash、Query count、key、solver metadata 与逐 rank cost ledger 已由最终
summarizer 硬校验；key 由逐 rank ledger 重算，不只信任 planner 写入值。modeled KI 含 256 MiB reserve，
低于 1 GiB budget。`TOKEN_LAYOUT` 共移动 114,560 个 remote rows；这是总 Query 的 87.4%。

### 8.3 逐 rank 结构负载结果

下表来自固定 1800 MHz、4 轮正式 run 的同一份 plan ledger。`relative range` 统一为
`(max-min)/mean`；proxy tick 只用于同一 pack 内排序，不解释成 GPU 时间。

| 结构负载 | min | mean | max | relative range |
|---|---:|---:|---:|---:|
| Query tokens | 16,384 | 16,384 | 16,384 | 0% |
| Indexer proxy tick | 2,484,003,328 | 2,484,761,696 | 2,485,300,480 | 0.0522% |
| HCA proxy tick | 48,643,072 | 48,670,720 | 48,707,584 | 0.1325% |
| HCA Query cost | 171,652 | 171,770 | 171,904 | 0.1467% |
| HCA send rows | 1,010 | 1,018.875 | 1,024 | 1.3741% |
| HCA receive rows | 1,010 | 1,018.875 | 1,024 | 1.3741% |
| HCA max-peer rows | 128 | 128 | 128 | 0% |
| packed KI rows | 1,816,960 | 1,827,092 | 1,838,944 | 1.2032% |
| modeled KI（MiB，含 256 MiB reserve） | 713.831 | 716.383 | 719.367 | 0.7728% |
| final fragments | 112 | 112.25 | 113 | 0.8909% |
| `TOKEN_LAYOUT` remote rows | 14,208 | 14,320 | 14,336 | 0.8939% |

因此 4 轮最终 plan 的主要计算 proxy 已处于 `0.0522%--0.1467%` 的窄范围，HCA 每 peer 最大行数
完全一致；路由行数、packed KI、显存和 fragment 的 spread 也都不超过 1.38%。这解释了后续实测中
HCA sparse core 已接近 1% range，而完整 HCA forward 仍可能受 NCCL arrival jitter 影响。

### 8.4 与 legacy 的 DSA-core 对比

下表的时间是 logical NVTX 内、按 CUDA runtime correlation ID 关联的 GPU kernel duration 之和。
`mean relative range` 先逐 step 计算 `(max-min)/mean`，再对 5 steps 取平均；`shared max` 保留五步中
最差的一步，不能用平均值掩盖离群。

| phase | legacy mean (ms) | shared mean (ms) | mean 变化 | legacy mean relative range | shared mean relative range | shared max |
|---|---:|---:|---:|---:|---:|---:|
| CSA Indexer score | 5.140 | 5.282 | +2.76% | 0.33% | 3.98% | 12.19% |
| CSA Indexer Top-K | 0.621 | 0.621 | -0.03% | 0.81% | 0.54% | 0.81% |
| CSA forward | 18.563 | 18.927 | +1.96% | 4.12% | 6.68% | 27.69% |
| CSA backward | 17.526 | 17.542 | +0.09% | 1.73% | 1.72% | 2.11% |
| HCA forward | 2.882 | 3.338 | +15.81% | 61.48% | 17.52% | 53.35% |
| HCA backward | 12.867 | 8.297 | **-35.51%** | 13.52% | **0.245%** | **0.303%** |

五步平均的 `forward + backward + parameter-gradient AllReduce` phase 和从 `55.584 ms` 降到
`51.975 ms`，即 DSA-core 账面下降 6.49%。这不是端到端训练加速结论，因为 capture 明确排除了
`TOKEN_LAYOUT` 和 projection。

最终 run 的 step 0 出现 GPU 时钟/首 capture step 型离群：Indexer score 的 8 rank kernel 工作项和
launch 数相同，但核心 kernel 为 `4.262--4.900 ms`，导致该步 relative range 为 12.19%；HCA forward
同一步也出现 53.35%。steps 1--4 的 CSA forward mean/max relative range 为 `1.42%/1.81%`，Indexer
score 为 `1.92%/2.24%`，HCA forward 为 `8.56%/11.90%`。独立 repeatability run 中，五步 Indexer
score max 为 3.29%，HCA forward mean/max 为 `11.85%/23.78%`，HCA backward mean/max 为
`0.340%/0.640%`。因此不能声称最终 run 的 CSA score “五步全部 <=5%”；也不能把单个 step 0
离群直接解释成结构性工作量不均。两份 run 一致证明 HCA backward 改善，HCA forward 仍有残余波动。

### 8.5 `TOKEN_LAYOUT` 与显存代价

最终 artifact 在 capture 外对三张独立 Attention 输入各执行一次 `TOKEN_LAYOUT.forward`，CUDA event
逐 rank min/mean/max 为：

| mode | min/mean/max (ms) |
|---|---:|
| W | 0.721 / 3.354 / 6.643 |
| CSA | 1.186 / 1.386 / 1.487 |
| HCA | 0.915 / 1.464 / 2.878 |

这是一次性边界调用，W 又是该进程最先发起的 layout collective，因此该数据包含首调用与时钟爬升，
不应当当作 steady-state 每层固定常数。attention-suite 的三张 graph 相互独立，并同时保留三份
post-projection leaves；其 max allocated memory 从 legacy 的 47,346,056,192 bytes 增至
49,931,553,792 bytes，增加 2.408 GiB。该差值不能直接外推为单层 production 增量。

### 8.6 当前结论

1. **HCA backward：成功。** 绝对时间下降 35.5%，平均 rank range 下降 98.8%，且两份 run 都稳定。
2. **HCA forward：明显改善但未完全解决。** relative range 大幅下降，但绝对时间上升，steps 1--4
   仍约 8.6% relative range。
3. **CSA：steady steps 基本保持均衡，最终 step 0 不稳定。** Top-K 和 CSA backward 无回退；Indexer
   score 的五步硬门槛在最终 run 未通过，不能只引用另一份较好 run 宣称稳定达标。
4. **算法选型：固定为确定性贪心 + 4 轮局部改善。** 当前主要剩余问题不是已证明的局部最优，而是
   proxy 校准、首 step 波动和 layout 边界成本；§8.7 的专门消融进一步否定了“增加局部搜索强度即可
   修复”的假设，§8.8 则否定了删除显存排序项。
5. **发布决策：保持 opt-in。** 在加入端到端 `TOKEN_LAYOUT.forward/reverse` 对比、校准 HCA forward
   proxy，并定义 W/HCA 绝对时间 non-regression 门槛前，不替换 release 默认 policy。

### 8.7 局部轮数与固定时钟/启动后预热消融

本节验证 4 轮是否是足够且有界的最终配置，以及原先首个 capture step 离群是否来自搜索不足。正式
矩阵使用同一诊断镜像、输入、参数和全部 tensor seed，8 张 B300 均锁定 graphics clock 为
1800 MHz。`nsys start` 之后、正式
`$Magi_DSA/capture_five_attention_suite_steps` 之前额外运行一个完整 attention-suite step：forward 为
`W → CSA → HCA`，backward 为 `HCA → CSA → W`，最后统一 parameter-gradient AllReduce。该 step
使用独立 NVTX 和 runtime counter，不进入正式五步；每个 run 仍严格通过 8F+8B、natural sequential
shadow、kernel attribution coverage=1.0 和时钟恢复审计。

```text
cross-run artifact:
artifacts/profile/20260803T054958Z-dsv4-shared-greedy-dispatch-ablation/

runs:
passes=0, 1, 4, 8 各一次；passes=4 共三次
clock=1800 MHz；profiler_attach_warmup_steps=1；formal_steps=5
```

六个 run 的 config/input/parameter/tensor-seed/image identity 逐 rank 全等。下表的 range 均为正式
steps 1--4 中逐 step `(max-min)/mean` 的平均；HCA FWD 由同一 logical NVTX 内经 correlation ID
归因的 CUPTI kernel 分解为 sparse core、NCCL SendRecv 和 other。`W prepare` 包含第一次共享 cold
solver 与 runtime prepare，不能当作纯 solver 微基准。

| run | key | W prepare mean (s) | HCA sparse range | HCA FWD range | NCCL range | DSA-core phase sum (ms) |
|---|---|---:|---:|---:|---:|---:|
| passes 0 | `(2485478144,48744448,114688,906)` | 13.415 | 0.95% | 12.16% | 54.23% | 54.999 |
| passes 1 | `(2485450496,48740352,114688,904)` | 18.511 | 0.72% | 9.62% | 44.01% | 54.635 |
| passes 4 / repeat 0 | `(2485300480,48707584,114560,898)` | 34.571 | 0.70% | 6.02% | 28.23% | 54.751 |
| passes 4 / repeat 1 | 同上 | 34.738 | 0.81% | 4.53% | 22.76% | 54.845 |
| passes 4 / repeat 2 | 同上 | 34.952 | 0.69% | 9.92% | 43.30% | 54.836 |
| passes 8 | `(2484569088,48707584,114304,890)` | 56.054 | 0.78% | 13.51% | 59.10% | 54.887 |

独立 cold-solver 微基准中，passes 0/1/4/8 分别耗时 3.156/8.648/24.577/45.971 秒，候选评估数为
8,193/12,673/26,113/44,033。4→8 轮使该微基准耗时增加 87.05%，但最大 Indexer proxy 只改善
0.0294%，最大 HCA proxy 完全不变。真实 sparse range `0.78%` 落在三次 4 轮重复的
`0.69%--0.81%` 内；完整 HCA FWD 和 NCCL range 反而更差，DSA-core phase sum 也没有可辨识收益。

固定时钟+启动后预热确实削弱了首步离群，但没有消除通信噪声。旧的未锁频/无 attach-warmup 最终 run
中，HCA FWD step 0 的 total/sparse/NCCL range 为 `53.35%/0.90%/112.05%`；三次 4 轮新 run 对应
total 为 `12.82%/23.63%/2.78%`，sparse 为 `0.90%/0.95%/0.62%`，NCCL 为
`53.60%/74.77%/14.37%`。这是一项“固定时钟+等价预热”的组合诊断，不把两者贡献强行拆开；它足以
证明 sparse compute 从首步起就均衡，而完整 FWD 仍随 NCCL/到达时序波动。

因此当前裁决是：

1. 最终算法固定使用 `local_improvement_passes=4`。0 轮已能得到较好的 sparse compute 均衡，但 4 轮
   提供有界、确定性的额外结构改善，且已完成 correctness 和重复 Profile 验证。
2. 8 轮相对 4 轮只改善 0.0294% 的最大 Indexer proxy，真实 sparse compute 落在 4 轮重复噪声内，
   因而没有采用价值。
3. 完整 HCA FWD 的剩余 spread 主要来自 NCCL/到达时序，不能用增加 owner 搜索强度解决；单个完整
   FWD/NCCL 离群也不能推翻 4 轮选型。

### 8.8 KI 显存排序历史消融

该一次性消融在保留 1 GiB 硬约束、`32 * packed_ki_rows`、4 轮和全部其他权重的前提下，只从 greedy
key 删除 `max_rank_modeled_ki_bytes`。结果仍满足 correctness，但 modeled KI rank range 从
`0.773%` 增至 `1.615%`，最大值增加 `3.174 MiB`，cold solver 候选数/时间增加
`15.38%/19.42%`；五步 DSA-core + gradient-AR 为 `54.770 ms`，相对基线 `54.751 ms` 没有收益。
完整 forward 差异主要来自 NCCL 波动，不能解释为确定性性能回退。

因此显存排序不是数学正确性的必要条件，却是低成本且有益的 plan-quality tie-break：硬约束只回答
“是否超限”，它进一步在可行候选中降低最坏 rank 显存。正式 solver 固定保留该项；一次性实验入口已在
裁决后删除。原始证据保留在：

```text
artifacts/profile/20260803T094307Z-dsv4-flash-128k-attention-suite-shared-greedy-passes4-kimemhardonly-clock1800-attachwarm1-precomputed-dout/
```
