# [未来草案] 多 CP Group / DP 间分配

> **状态：独立的未来多-CP-group/DP 分配草案。**
>
> 本文研究多个 DP replica/CP group 之间的样本与 GPU 容量分配，不在当前单机
> 8×B300、`world_size=cp_size=8` 的 DeepSeek-V4-Pro release 范围内，也不改变 CP group 内
> `structural_balanced` Query layout 或 DSA 数学。当前 Pro 合同见
> [`magi_dsa_v4_design.md`](./magi_dsa_v4_design.md) 和
> [`README_dsv4_cp_dispatch_structural_balancing.md`](./README_dsv4_cp_dispatch_structural_balancing.md)。

## 1. 问题定义
系统包含 $$N$$ 张 GPU，其中：
$$N = 2^k,\qquad N > 1024$$
输入为 $$m$$ 个长度不相同的样本：
$$L_1,L_2,\ldots,L_m$$
样本长度分布高度不均，范围约为 1K 到 1M tokens。
所有 GPU 被划分为多个 DP replica，每个 DP replica 是一个 CP group。不同 CP group 可以包含不同数量的 GPU。MagiAttention 负责 CP group 内部的计算与 token 负载均衡；本文关注 CP group 之间的样本分配、计算负载均衡和显存负载均衡。
当前模型只考虑：
- 样本计算 cost：$$L_i^2+L_i$$；
- 样本显存 cost：$$L_i$$；
- 每张 GPU 最多容纳 8K，即 8192 个 token；
- 单个 CP group 最多包含 128 张 GPU；
- CP 通信开销暂不计入 cost 模型 (先寻找计算和显存均衡的解，再找通信解)。
2. 决策变量
CP group 的数量记为 $$G$$。$$G$$ 不预先固定，由分配算法决定。
第 $$j$$ 个 CP group 的 GPU 数量记为：
$$c_j\in\mathbb{Z}_{>0},\qquad j=1,\ldots,G$$
样本分配变量定义为：
$$x_{ij}\in\{0,1\}$$
其中 $$x_{ij}=1$$ 表示样本 $$i$$ 被分配给 CP group $$j$$。
第 $$j$$ 个 CP group 的样本集合为：
$$S_j=\{i\mid x_{ij}=1\}$$
3. Cost 模型
3.1 样本 Cost
样本 $$i$$ 的计算 cost 为：
$$q_i=L_i^2+L_i$$
其中平方项表示 Attention 计算量，线性项表示 MLP 等计算量。
样本 $$i$$ 的显存 cost 为：
$$v_i=L_i$$
3.2 CP Group 原始负载
第 $$j$$ 个 CP group 的总计算量为：
$$Q_j = \sum_{i=1}^{m}x_{ij}(L_i^2+L_i)$$
总显存相关 token 量为：
$$V_j = \sum_{i=1}^{m}x_{ij}L_i$$
3.3 单 GPU 负载
假设 MagiAttention 能够在 CP group 内部实现理想负载均衡，则第 $$j$$ 个 CP group 的单位 GPU 计算负载为：
$$C_j=\frac{Q_j}{c_j}$$
单位 GPU 显存负载为：
$$M_j=\frac{V_j}{c_j}$$
不同 CP group 大小时，应该比较 $$C_j$$ 和 $$M_j$$，而不是直接比较未经 CP size 归一化的 $$Q_j$$ 和 $$V_j$$。
4. 约束条件
4.1 GPU 完整划分
所有 GPU 必须被分配到某个 CP group：
$$\boxed{ \sum_{j=1}^{G}c_j=N }$$
4.2 CP Group 大小
每个 CP group 至少包含一张 GPU，最多包含 128 张 GPU：
$$\boxed{ 1\le c_j\le128 }$$
如果底层实现进一步要求 CP size 为 2 的幂，则增加：
$$c_j\in\{1,2,4,8,16,32,64,128\}$$
因为 $$N>1024$$ 且 $$N$$ 是 2 的幂，所以 $$N\ge2048$$，CP group 数量至少为：
$$\boxed{ G\ge\frac{N}{128} }$$
4.3 样本唯一分配
每个样本必须且只能分配给一个 CP group：
$$\boxed{ \sum_{j=1}^{G}x_{ij}=1,\qquad \forall i }$$
4.4 非空 CP Group
每个有效 CP group 至少包含一个样本：
$$\boxed{ \sum_{i=1}^{m}x_{ij}\ge1,\qquad \forall j }$$
因此：
$$G\le m$$
4.5 单卡 8K Token 显存约束
每张 GPU 最多容纳 8192 个 token，因此大小为 $$c_j$$ 的 CP group 最多容纳 $$8192c_j$$ 个 token：
$$\boxed{ \sum_{i=1}^{m}x_{ij}L_i \le 8192c_j,\qquad \forall j }$$
等价地：
$$\boxed{ M_j=\frac{V_j}{c_j}\le8192 }$$
该约束同时限制了单个样本所需的最小 CP size。如果样本 $$i$$ 被分配给 CP group $$j$$，则必然有：
$$c_j \ge \left\lceil\frac{L_i}{8192}\right\rceil$$
如果 CP size 必须是 2 的幂，则最小 CP size 为：
$$c_{\min}(L_i) = 2^{ \left\lceil \log_2 \left( \left\lceil\frac{L_i}{8192}\right\rceil \right) \right\rceil }$$
4.6 单样本可行性
由于 CP group 最大为 128，单个样本长度必须满足：
$$\boxed{ L_i\le8192\times128=1{,}048{,}576,\qquad \forall i }$$
因此长度为 1M 左右的样本需要接近或等于 CP128。
4.7 全局必要条件
所有样本的 token 总数不能超过所有 GPU 的总容量：
$$\boxed{ \sum_{i=1}^{m}L_i\le8192N }$$
该条件与单样本可行性都是必要条件，但由于样本不可跨 CP group 分配，它们本身不保证一定存在满足所有约束的分组。
5. 优化目标
5.1 理想单位 GPU 负载
全局理想单位 GPU 计算负载为：
$$\overline C = \frac{\sum_{i=1}^{m}(L_i^2+L_i)}{N}$$
全局理想单位 GPU 显存负载为：
$$\overline M = \frac{\sum_{i=1}^{m}L_i}{N}$$
因此，大小为 $$c_j$$ 的 CP group 的理想总负载为：
$$Q_j^*=c_j\overline C$$
$$V_j^*=c_j\overline M$$
5.2 Group 负载误差
第 $$j$$ 个 CP group 的相对计算误差为：
$$e_j^C = \frac{|C_j-\overline C|}{\overline C} = \frac{|Q_j-c_j\overline C|}{c_j\overline C}$$
相对显存误差为：
$$e_j^M = \frac{|M_j-\overline M|}{\overline M} = \frac{|V_j-c_j\overline M|}{c_j\overline M}$$
整体均衡误差定义为所有 CP group、两个 cost 维度中的最大误差：
$$\boxed{ E = \max_{j=1,\ldots,G} \left\{ e_j^C,e_j^M \right\} }$$
5.3 最小最大误差优化
最终优化问题为：
$$\boxed{ \min_{G,\{c_j\},\{x_{ij}\}} E }$$
并满足第 4 节中的全部约束。
等价地，可以引入变量 $$\epsilon$$，求解：
$$\boxed{\min \epsilon}$$
满足对所有 CP group $$j$$：
$$(1-\epsilon)c_j\overline C \le \sum_{i=1}^{m}x_{ij}(L_i^2+L_i) \le (1+\epsilon)c_j\overline C$$
以及：
$$(1-\epsilon)c_j\overline M \le \sum_{i=1}^{m}x_{ij}L_i \le (1+\epsilon)c_j\overline M$$
同时满足：
$$\sum_jc_j=N$$
$$1\le c_j\le128$$
$$\sum_jx_{ij}=1$$
$$\sum_ix_{ij}\ge1$$
$$\sum_ix_{ij}L_i\le8192c_j$$
6. 优化结果的含义
当最优值：
$$\epsilon^*=0$$
时，所有 CP group 的单位 GPU 计算负载和显存负载完全相同。
当：
$$\epsilon^*>0$$
时，严格均衡解不存在，$$\epsilon^*$$ 表示当前约束和 cost 模型下能够达到的最小最大相对误差。
由于样本不可切分，严格均衡解不一定存在；但只要可行域非空，最优近似解一定存在。
7. 当前设计边界
当前模型暂不包含：
- 不同 CP size 对计算效率的影响；
- CP 通信开销；
- MagiAttention 在实际输入上的非理想组内均衡；
- 不同 GPU 型号或算力差异；
- 参数、梯度、优化器状态等固定显存；
- DP 梯度同步开销。
后续可以将单位 GPU 计算负载扩展为：
$$C_j = \frac{Q_j}{c_j\eta(c_j)} +T_{\mathrm{comm}}(c_j)$$
其中 $$\eta(c_j)$$ 表示 CP group size 对实际计算效率的影响。
