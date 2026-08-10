# Magi-DSA

Magi-DSA 是建立在 MagiAttention Core 之上的 DeepSeek 式稀疏注意力扩展。它是一个
独立的 Python 发行包 `magi_attn_extensions`，自己拥有计划、运行时、通信调度、
后端和 kernel。

```text
magi_attn_extensions/DSA  ---->  magi_attention (Core)
```

依赖是单向的。Core 从不反过来导入这个包。

## 怎么跑

这里的每个脚本都在固定镜像里运行，而且镜像必须带着当前 commit 作为标签，否则脚本
直接拒绝启动。所以流程永远是先提交、再构建、再运行，不会有人不小心拿一份没提交的
代码去跑出一个结果。

### 为当前 commit 构建镜像

```bash
scripts/image/build.sh --revision "$(git rev-parse HEAD)"
```

每次提交之后、想测或想 profile 之前都要做一次，几分钟，产出的标签是
`magi-dsa-v4:<12 位 commit>`。下面所有命令都读这个标签，先定义一次：

```bash
IMAGE="magi-dsa-v4:$(git rev-parse --short=12 HEAD)"
```

### 单元测试

```bash
scripts/test/run_cp1.sh --image "$IMAGE"
```

改代码的过程中你通常不想每改一行就重建一次镜像。单测只需要镜像里那套编译好的
kernel，所以把工作区挂进去盖住它，就完全不用重建：

```bash
docker run --rm --gpus '"device=0"' --ipc=host --ulimit memlock=-1 \
    --volume "$PWD:/work:ro" --workdir /tmp \
    --env PYTHONPATH=/work/extensions --env TMPDIR=/tmp \
    --entrypoint python3 "$IMAGE" \
    -m pytest /work/extensions/tests/dsa_v4 -q -p no:cacheprovider --rootdir=/tmp
```

这样跑的是你刚改的源码，配上你上次构建的镜像，两分钟出结果。日常改代码用这条，
提交前再走一遍上面那个正式脚本。

### cp=8 正确性

```bash
scripts/test/run_multigpu.sh --world-size 8 --case cp8-natural-backward --image "$IMAGE"
```

八张卡跟单卡参考实现对数。这一关是专门抓死锁的：某张卡的集合通信次数如果取决于它
自己的数据，只会在这里挂住，别处都看不出来。产物写在
`artifacts/correctness/<时间戳>/`，判决是 `SUMMARY.json` 和 `PHASE_AUDIT.json` 里
的 `"result": "PASS"`。

换成 `--case cp8-topk-diagnostic` 是比对 Top-K 选择，
`--world-size 2 --case csa-natural-backward` 是两卡的快速冒烟。

### cp=8 profile

```bash
scripts/profile/run_5step.sh --world-size 8 --cp-size 8 --case dsv4-pro-128k \
    --plans balanced --steps 5 --step-mode pro-pair \
    --layout-policy structural-balanced --local-improvement-passes 4 \
    --profiler-attach-warmup-steps 0 --skip-smoke --image "$IMAGE"
```

采五步，大约十五分钟，产物在 `artifacts/profile/<时间戳>/`。和上面两条不同，这条
跑的是镜像里烤进去的 wheel，完全不看你的工作区，所以没提交的改动根本不会被测到。

它校验的东西远多于计时：每个阶段的集合通信次数、路由发起的先后、每条路由遮挡了
哪段计算、Indexer kernel 在各卡之间是否均衡、以及显存拷贝的归因。一个跑得挺快但
违反其中任何一条契约的版本，照样是 FAIL。

跑它的时候把机器让给它。它测的是墙钟，同机跑别的负载会污染数字；而且单次之间本来
就有百分之一左右的波动，同一配置重复三次再判断差异。

想直接看结论的话，`REPORT_PRO_PAIR.md` 是人读的汇总，
`MAJOR_KERNEL_BALANCE_PRO_PAIR.json` 是各大 kernel 的耗时和跨卡均衡，
`PRO_PAIR_ROUTE_TIMINGS.json` 是每条路由的耗时和遮挡情况，
`balanced/*.sqlite` 是 nsys 原始库。

### 静态检查

```bash
PYTHONPATH=.:extensions python3 -m mypy magi_attention extensions/magi_attn_extensions/DSA
```

## 安装

Core 和扩展是两个发行包，都要装：

```bash
python3 -m pip install --no-build-isolation -e .
python3 -m pip install --no-build-isolation -e ./extensions
```

注意：本仓库和 MSA 仓库都会构建一个叫 `magi_attn_extensions` 的发行包，两者共用
同一个父级 `__init__.py`，所以它们的 wheel 不能同时装进一个环境。MSA 和 DSA 在包
布局和依赖方向上是同级的，在安装上不是。

## 边界

Magi-DSA 只把自己需要什么表述成区间，剩下的降级交给 Core。它不拥有任何集合通信、
任何逐行映射表，也没有自己的分发算法。

| Core 能力 | 使用方 |
|---|---|
| `magi_attention.common.AttnRanges` | `meta.py`、`solver.py` |
| `magi_attention.common.enum.AttnMaskType` | `solver.py` |
| `magi_attention.common.range_op` 的 range_gather 和 range_reduce | `comm.py`、`dist.py` |
| `magi_attention.comm.primitive.grpcoll` 的 group_cast 和 group_reduce | `comm.py` |
| `magi_attention.comm.work.WorkWithPostProcessFn` | `comm.py` |
| `magi_attention.meta.collection.comm_meta` 的集合通信参数 | `packing.py` |
| `magi_attention.meta.solver.dynamic_attn_solver` 的区间降级 | `packing.py` |
| `magi_attention.meta.solver.dispatch_solver` 的分发类型和算法 | `solver.py` |
| `magi_attention.utils.general._make_device_tensor` | `packing.py` |
| `magi_attention.utils.nvtx` | `nvtx.py`、`backend.py` |
| `magi_attention.meta._make_dispatch_meta` 的 dispatch meta 和 bucket | `solver.py` |

最后一条是唯一走私有路径的。`__init__.py` 会同时检查这个符号存不存在、以及它的
签名是否仍然接受 `dispatch_config`、`is_same_source`、`is_q_permutable`、
`is_k_permutable` 和 `uneven_shard`，这样一个不兼容的 Core 会在导入时就报错，而不是
等到计划构建深处才炸。

如果将来某个能力对别的扩展或者稠密注意力也有用，它应该以一个不含 DSA 概念的最小
原语的形式进 Core。

## 这个扩展不拥有什么

这些是刻意写出来的，因为每一条都曾经在这里被重复实现过一次：

- 没有自己的全体对全体数据面。一条路由就是一次 group-cast，它的伴随就是对称的
  group-reduce，两者都来自 Core。
- 没有逐行的发送、接收、消费或反向索引表。路由是区间，所以一份计划的大小取决于
  分片数量，而不是 token 数或压缩行数。
- 没有自己的行收集或 CSR 规约 kernel。重复行的收集用 `index_select`，它的反向本来
  就会累加；前缀收集用 Core 的区间算子。
- 没有可训练参数。所有权重都属于模型，以 `DsaProjections` 回调的形式到达运行时。
- 没有 object 类型的集合通信。计划是调用方元数据的纯函数，每张卡各自重算一份逐位
  相同的副本，不需要谁去收集或广播。
- 没有可选的布局策略。`structural_balanced` 是唯一的布局，就是那个在 packed-global
  chunk 上按 causal 面积做的 MinHeap。

## 公开 API

```python
from magi_attn_extensions.DSA import (
    DsaProjections,
    DsaRatio,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSALayer,
    MagiDSAPackedMeta,
    MagiDSAProExecutionBundle,
    MagiDSAProLayerStack,
    MagiDSAProModelSpec,
    MagiDSAProRuntimeMgr,
    MagiDSAProjector,
    MagiDSARuntimeMgr,
    layout_and_project_dsa_input,
    layout_source_hidden_once,
    project_local_dsa_input,
)
```

`DsaRatio` 是一个只用于类型标注的 `Literal` 别名，`MagiDSAProjector` 是一个由调用方
实现的 `Protocol`。

`MagiDSAPackedMeta` 携带 `cu_seqlens` 和完整的 `source_token_counts` 切分。调用方
本来就拥有这份切分，把它说出来之后，执行计划就成了调用方元数据的纯函数，这正是
计划广播和 owner 布局收集被去掉的原因。

### 声明的进阶子模块

它们不在顶层 `__all__` 里，但被仓库内的 benchmark 和脚本导入，所以改名要同步改
那些使用方：

```text
magi_attn_extensions.DSA.nvtx                        # dsa_nvtx_range
magi_attn_extensions.DSA.comm                        # unlayout_dsa_query_tensor
magi_attn_extensions.DSA.kernels.triton.diagnostics  # 非有限值的行和块统计
```

运行时的返回类型和诊断信息 `DsaExecutionHandle`、`DsaRuntimeCounters`，以及计划的
dataclass 和 solver，分别仍然可以从 `.runtime`、`.meta` 和 `.solver` 拿到，同样属于
进阶 API。

## 目录

```text
DSA/
├── __init__.py                  # 稳定的公开导出 + Core 兼容性检查
├── config.py, types.py          # 配置和面向张量的类型
├── projection.py                # 模型侧回调的边界
├── modeling.py                  # 参考模型模块，属于模型侧
├── runtime.py, pro_runtime.py   # 运行时管理器
├── meta.py, solver.py           # 区间形状的计划和它的求解器
├── comm.py, packing.py          # Core 集合通信路由和设备侧映射
├── dist.py, schedule.py         # 正向和反向的调度
├── backend.py, nvtx.py, phase.py, reference.py
└── kernels/triton/*.py          # 9 个融合的逐元素和索引 kernel
```

`modeling.py` 是模型侧代码。执行路径上的任何文件都不导入它，运行时只看得见
`MagiDSALayer.projections()` 交出来的那几个回调。换一个模型可以把这个文件整个替换
掉，只要它提供一份 `DsaProjections`。
