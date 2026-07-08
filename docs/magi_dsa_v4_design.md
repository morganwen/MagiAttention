# Magi_DSA V4 设计文档

- 依据 AGENTS.md 的 V4 契约与 PLAN.md 设计文档阶段清单撰写。本文冻结实现口径，代码以此为准。
- 数值权威参考：Megatron-LM dsv4 实现，本地 /home/scratch.wewen_gpu/megatron-lm 分支 dsv4-repro，commit c6449f0b2。下文引用的行号都指该 commit。

## 已消除的开放问题

- compressor 输入口径：吃 hidden，不吃投影后的 KV。参考实现 Compressor 的内容投影和门控投影都是 hidden_size 到 coff 乘 head_dim（csa.py:852、:866），主压缩流与 indexer 压缩流都从 hidden 出发。Magi_DSA 照此办理，config 携带 hidden_size。
- indexer 输入口径，对 AGENTS.md 的一处契约修正：indexer 的 Qi、Ki、Weights 都不是外部输入，而是 runtime 内部投影生成。参考实现 CSAIndexer 持有 wq_b（q_lora_rank 到 64 乘 128）、weights_proj（hidden 到 64）和自己的 rotate 压缩器（csa.py:1191 起），输入是 hidden x 和 q 的 latent qr。Magi_DSA 的 API 输入相应为 x 与 qr，indexer 参数归 runtime 所有，梯度经 autograd 落在自有参数上，对外返回 dx、dqr。此修正待用户确认后回写 AGENTS.md。
- sink 梯度路径：sink 等价于一条 value 为零向量的额外注意力条目。设 p_sink 为它的注意力概率，d_sink 对每个头 h 为对所有 query 求和的 −p_sink(q,h) 乘 dO(q,h) 点乘 O(q,h)。reference 路径由 autograd 自动产生；kernel 路径若 cudnn-fe 不产 d_sink，wrapper 用保存的 LSE 求 p_sink = exp(sink − lse_total) 按上式补算。两条路径都要过 sink 专项验收。
- KL 精确口径，逐项对齐参考实现 compute_dsa_indexer_loss（dsa.py:231）：
  - predict 是 index_scores 的 softmax。index_scores 等于对 head 求和的 weights 乘 relu(Qi 点乘 Ki)，fp32（dsa.py:_compute_index_scores）。进 KL 前 weights 额外乘 indexer softmax_scale，即 index_head_dim 的负二分之一次方（csa.py:1591）。top-k 选择用不乘 scale 的分数，因缩放不改变排序。
  - target 是主注意力分数的逐头 softmax：q 点乘压缩 KV 乘主 softmax_scale，压缩条按块级因果掩码，逐头 softmax 后对头求和再 L1 归一。
  - sparse loss 开启时，target 和 predict 都先加 top-k 的 index_mask，分布限制在选中位置上。V4 recipe 开 sparse loss，V1 只实现 sparse 口径，dense 口径留接口不留实现。
  - 全掩码行，即块级因果下可见块数为零的行，logits 置零过 softmax 再乘行掩码归零，不产生 NaN，不贡献损失。
  - KL(target 对 predict) 沿 KV 轴求和成每行标量，token-mean 归一，乘 loss_coeff。per-token-loss 模式改为求和。CP 下分母是全局 query token 数。
- 窗口折叠布局：kv_full 平铺为原始段在前压缩段在后，压缩索引加原始段长度作 offset，窗口索引与压缩 top-k 索引沿最后一维拼接，一次 sparse attention 统一算（csa.py:1629、:2415）。kernel 路径的 topk 对齐要求参照 Megatron dsa_kernels.py 的 _get_topk_alignment。

## 公共 API 与 tensor schema

- 包位置 magi_attention/experimental/dsa_v4，独立 runtime，不进 calc_attn。对外类名 MagiDSAV4，配置类 MagiDSAV4Config。
- 配置字段与 V4-Flash 默认值：compress_ratio 必填取 0、4、128 之一；hidden_size 必填；q_lora_rank 必填；num_heads 64；kv_dim 512；rope_dim 64；window_size 128；topk 512；indexer_heads 64；indexer_dim 128；softmax_scale 必填；indexer_loss_coeff 0.01；use_sparse_loss True；compress_rotary_base 40000 与 YaRN 参数；校验拒绝契约外组合。
- forward 输入，SBHD 参考布局 sq b 在前，packed THD 布局见 packed 一节：
  - x：hidden，[sq, b, hidden_size]，喂两个压缩器和 weights_proj
  - qr：q latent，[sq, b, q_lora_rank]，喂 indexer 的 wq_b
  - q：主 query，RoPE 已完成，[sq, b, 64, 512]
  - kv：每 token 一条 latent KV，RoPE 已完成，[sq, b, 512]
  - sink：[64] fp32 可学习参数，由调用方持有传入
- forward 输出：O [sq, b, 64 乘 512]，kl_loss 可微标量。compress_ratio 为 0 或 128 时 kl_loss 恒为零标量，保持签名一致。
- backward 由 autograd 驱动：dO 与 d_kl 进来，产出 dx、dqr、dq、dkv、d_sink，runtime 自有参数的梯度落在参数上。
- runtime 自有参数：主压缩器的 wkv、wgate、ape、norm；indexer 的 wq_b、weights_proj 和它的压缩器同名参数。compress_ratio 为 0 时不建任何压缩器参数，128 时不建 indexer 参数，与参考实现的条件构造一致（csa.py:1488、:1502）。

## 数学口径

- 压缩器：cutoff 等于 seqlen 对 ratio 取整乘 ratio，尾巴不压。内容投影与门控投影 hidden 到 coff 乘 head_dim，门控加块内位置嵌入 APE，块内 softmax 用 fp32，加权求和后过 RMSNorm，再按块序号位置打 RoPE，nope 与 rope 维划分为 head_dim 减 rope_dim 与 rope_dim。ratio 4 时 coff 2 做重叠：每 token 投影劈两半，前半给下一块后半留本块，块首缺口分数填负无穷；ratio 128 时 coff 1 不重叠。indexer 压缩器多一步 Hadamard 旋转。全部对齐 csa.py:795 起的 Compressor。
- 块级因果：位置 p 可见 (p+1) 整除 ratio 条压缩条，用于 top-k 掩码、KL 掩码和索引合法性校验。
- 窗口：每 query 附带同 sample 内最近 window_size 个原始 token 的索引，越界填负一。
- 统一 sparse attention：gather 平铺索引对应的 KV，fp32 打分乘 softmax_scale，无效位填负无穷，带 sink 的 softmax，加权求和。对齐 unfused_compressed_sparse_attn（csa.py:535）。
- 三形态：ratio 0 只有窗口索引；ratio 4 窗口拼 indexer top-k；ratio 128 窗口拼块级因果前缀内全部压缩条（csa.py:1622 的 else 分支语义）。

## saved-state 契约

- forward 保存：O、fp32 LSE、topk_idx、topk_length、压缩器反向所需的门控中间量。V1 的 reference 路径先用 autograd 默认保存行为跑通数值，再按本契约收紧并配 saved-tensor hooks 专项；kernel 路径从一开始就按本契约实现。
- 压缩条不保存，backward 重算或由门控中间量恢复，取舍在单卡核心阶段以显存实测定案，结论回写本文档。
- remote 与 packed tensor 不保存，CP 阶段 backward 按静态 plan 重收。

## kernel 接入计划

- 阶段一，纯 PyTorch reference：全链路 torch 算子加 fast-hadamard-transform，与 Megatron unfused 路径对拍。本机 SM90 可完整执行。
- 阶段二，cudnn-fe 接入：indexer_forward、indexer_top_k、indexer_backward、score_recompute、sparse_attention_backward，SM90 与 SM100 都官方支持，本机可验 backward 侧数值。sink 直传真实值，d_sink 视 kernel 能力走原生或 LSE 补算。
- 阶段三，FlashMLA sparse forward：nv_dev 分支，V4 形状走 head64 对齐路径。本机为 SM90，Megatron 官方容器在 SM90 上禁编该 kernel，故此阶段的构建与数值验证留待 SM100 环境，接入代码与 wrapper 先行完成并以 reference 兜底。
- kernel 与 reference 的切换走 config 的 backend 字段，默认 reference。

## CP 设计要点

- dispatch 块对齐 128，块归属最后一个 token 所在 rank，halo 行数取 window_size 与 ratio 的最大值即 128。
- 通信内容：原始 KV 与 x 的左边界 halo 点对点；压缩条、indexer 压缩 Ki 全组 allgather；backward 反向归并对应各自路径，压缩条梯度经压缩器反传，不跨 rank 通信压缩条梯度本体而是归并到原始 token 的 dx。
- metadata schema：块 id 与原始 token 区间双向映射、sample id、owner rank、packed 行号，全部显式 dataclass，细化在 CP 阶段动工前补充到本文档并评审。

## 测试计划

- 单卡：reference 对 Megatron unfused 的逐张量对拍，覆盖三形态、compressor 重叠边界、块级因果、sink 双向、KL 标量与全部梯度；容差用 magi_attention.testing.precision 流程校准后冻结。
- packed：多 sample、不连续 fragment 块对齐、行尾负一、saved-state 专项。
- CP：本机 CP=2 与 CP=1 全量对拍；CP=4、8 与性能门槛留 SM100 环境。
- 正式单测放 tests 目录，agent 自用的探索性脚本放 agents/tests。
