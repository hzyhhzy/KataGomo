# v112 六头模型与多目标 MCTS 实验记录

本文记录分支 `codex/multihead-vct-search` 上围绕 v112 六头模型所做的实现、
搜索改造、局面基准、正式对局和失败尝试。它的目的不是为当前所有实验参数背书，
而是保留足够完整的上下文，使后续工作不必重新踩一遍相同的坑。

记录日期：2026-07-31。

## 1. 结论先行

这轮研究没有找到接近 +100 Elo 的方案。

正式对局中，唯一稳定跨过统计误差的改动是 v24 的根节点“强制应手”验证
`fr12`：

| 候选 | 基线 | 盘数 | 胜/和/负 | Elo | 95% 区间 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `fr12` | `c0` | 1200 | 331/609/260 | +20.6 | [+6.8,+34.4] | 显著提升 |
| `fr16` | `fr12` | 400 | 96/205/99 | -2.6 | [-26.5,+21.2] | 增加预算无收益 |
| `fr12ft4` | `c0` | 400 | 94/205/101 | -6.1 | [-29.9,+17.7] | 向树内延伸后退化 |
| `tp16p1` | `fr12` | 800 | 204/400/196 | +3.5 | [-13.6,+20.5] | 不显著 |
| `tp24p15 + fr12` | `c0` | 800 | 219/380/201 | +7.8 | [-9.6,+25.3] | 不显著 |
| `tg64p2g2` | `fr12` | 800 | 174/422/204 | -13.0 | [-29.6,+3.5] | 淘汰 |
| `tp24p15` | `fr12` | 6400 | 1606/3187/1607 | -0.1 | [-6.1,+6.0] | 中性 |
| `tp24softgate_v51` | `fr12` | 5600 | 1390/2819/1391 | -0.1 | [-6.5,+6.4] | 中性 |

`tp24p15` 的前 3200 盘约为 +7.5 Elo，后 3200 盘约为 -7.6 Elo，
合并后回到 -0.1 Elo。这是为什么正式测试不能在看到顺眼的中途结果时停止。

v51 按每个 400 盘批次聚类后为 -0.06 Elo，批次级 95% t 区间约
`[-8.4,+8.3]`。原 tp24 同样为 -0.05 Elo，区间约 `[-8.1,+8.0]`。
两者不仅均值相同，批次标准差也几乎相同。

最可信的技术判断是：

1. 六头模型确实包含少量 head0 不容易发现的战术信号。
2. 这些信号非常稀疏，固定给辅助 policy 分配少量 root visit 几乎不改变实战。
3. 一旦把预算放大到足以频繁改招，误报和主搜索预算损失又会抵消收益。
4. 开发局面上“只改对了两个点”的候选极易过拟合，v49 是最清楚的反例。
5. 下一步更值得做的是训练或校准“何时值得启动独立战术验证”的触发器，
   而不是继续手调固定 visit 数。

## 2. v112 模型接入

### 2.1 输入与输出

v112 继续使用 v101 输入。`NNModelVersion::getInputsVersion(112)` 返回 101，
空间特征数和全局特征数也走 v101 路径。

ONNX 的五个输出在第一维拼接六个头：

```text
out_policy        [N, 24, P*P+1]
out_value         [N, 18]
out_miscvalue     [N, 60]
out_moremiscvalue [N, 48]
out_ownership     [N, 6, P, P]
```

原 v102 的对应头数是 1；v112 的每组尺寸是原来的 6 倍。后端按六等份解释，
head0 仍填入旧字段，额外 policy/value 头保存在 `NNOutput` 中供搜索使用。
旧模型的 value 结果复制到六个逻辑槽位，因此 `multiValueHeadUtilityMix=0`
时行为保持旧版；需要额外 policy 头的功能在非 v112 模型上会明确报错。

### 2.2 支持的后端

v112 当前只支持 ONNX 路径：

- TensorRT ONNX：`cpp/neuralnet/trtbackend.cpp`
- ONNX Runtime CPU：`cpp/neuralnet/onnxbackend_cpu.cpp`
- ONNX Runtime DirectML：`cpp/neuralnet/onnxbackend_directml.cpp`

Eigen、CUDA 原生模型、OpenCL 和文本模型解析路径检测到 v112 时直接报错。
这符合“只实现 ONNX，其他后端不要静默误读”的约束。

### 2.3 `NNOutput` 所有权

`NNOutput` 新增：

```cpp
static constexpr int NUM_VALUE_HEADS = 6;
static constexpr int NUM_POLICY_HEADS = 6;

float whiteWinProbByHead[NUM_VALUE_HEADS];
float whiteLossProbByHead[NUM_VALUE_HEADS];
float whiteNoResultProbByHead[NUM_VALUE_HEADS];
int8_t* policyProbsByHeadQuantized;
```

额外 policy 以量化数组保存。拷贝构造、赋值、对称平均和析构都显式处理这块
动态内存。开发中曾出现过 `match` 完成后段错误；没有保留可证明唯一根因的
崩溃转储，但问题暴露后重点补齐了这块新增内存的初始化、深拷贝和释放。
后续数千盘连续 `match` 均能正常结束。

## 3. 六个头在本分支中的解释

以下是搜索代码实际采用的解释，不对训练侧未提供的细节作额外推断。

| 头 | 本分支用途 |
| --- | --- |
| head0 | 原始规则下的 policy/value/no-result，兼容旧搜索 |
| head1 | 可选的第二个普通 policy；通过 `multiHeadNormalPolicyHead1Mix` 与 head0 混合 |
| head2 | “和棋算当前行动方赢”的 policy/value；用于构造对手真正获胜概率或己方不败目标 |
| head3 | “和棋算当前行动方输”的 policy/value；用于构造当前方真正获胜概率 |
| head4 | 固定进攻方 VCT 时，轮到进攻方走的 policy/value |
| head5 | 固定进攻方 VCT 时，轮到防守方走的 policy/value |

head2/head3 的选取必须跟 `nextPla` 一起解释。例如白方行棋时，真正白胜概率
取 head3 的白胜；黑方行棋时，真正白胜概率取 head2 的白胜。黑胜同理。
每个 value 头先在自己的三个 logits 内单独 softmax，再读取胜、负、无结果概率。

head0 的 no-result 概率仍用于报告。由不同规则头拼出的白胜、黑胜和 no-result
不要求和为 1，这是设计允许的。

## 4. 两套普通 utility 与节点统计

### 4.1 白方统一视角

为了减少符号错误，节点中的两个普通目标都使用白方视角：

- `whiteWinUtility`：白方获胜目标。
- `blackWinUtilityInv`：白方“不输给黑方”的目标，即黑胜概率取反后映射到
  `[-1,1]`。

设：

```text
L = head0WhiteWin - head0BlackWin
    + noResultUtilityForWhite * head0NoResult
c = multiValueHeadUtilityMix
```

先构造随 `nextPla` 选择 head2/head3 的概率：

```text
pWhite(c) = (1-c) * head0WhiteWin + c * decisiveWhiteWin
pBlack(c) = (1-c) * head0BlackWin + c * decisiveBlackWin
```

再构造两套 utility：

```text
whiteWinUtility =
  (1-c) * L + c * (2*pWhite(c)-1)

blackWinUtilityInv =
  (1-c) * L + c * (1-2*pBlack(c))

utility = 0.5 * (whiteWinUtility + blackWinUtilityInv)
```

因此：

- `c=0` 严格回到原版 utility。
- `c=1` 时白目标只看白胜，黑目标取负后只看黑胜。
- 中间值按代码进行两层插值。
- `c!=0` 时要求 `noResultUtilityReduce==0`，避免两套和棋处理叠加。

### 4.2 节点累计项

`NodeStatsAtomic` 在原有累计项之外维护：

```text
whiteWinUtilityAvg
whiteWinUtilitySqAvg
whiteWinWeightSum
whiteWinWeightSqSum

blackWinUtilityInvAvg
blackWinUtilityInvSqAvg
blackWinWeightSum
blackWinWeightSqSum
```

普通 edge 还可分别记录：

```text
whiteWinEdgeVisits
blackWinEdgeVisits
```

VCT 搜索不复用普通统计，而是维护：

```text
whiteVctStats / blackVctStats
whiteVctEdgeVisits / blackVctEdgeVisits
whiteVctVirtualLosses / blackVctVirtualLosses
```

这使“为了验证 VCT 多搜的 visit”可以留在隔离平面，而不必天然按 visit
平均进普通胜率。不过，只要一个方案最终用额外普通规则 playout 验证候选，
这些普通 playout 仍可能影响主统计；后面的 sidecar 系列就是为进一步隔离
这种污染而做的尝试。

### 4.3 selection value

白胜与黑胜反目标分别计算完整 PUCT/FPU selection value。默认
`multiHeadObjectiveSelectionSharpness=0` 时按权重平均，而不是取最大值。

固定偏置参数：

```text
p = multiValueHeadSelectionBias, p in [-1,1]

当前方胜目标权重 = 0.5 + 0.5*p
另一方胜目标权重 = 0.5 - 0.5*p
```

`multiHeadObjectiveSearchStrength>0` 时，再根据节点当前“胜”和“不败”的
未决程度动态修正权重。代码使用软权重并保留 0.10 floor，避免某个目标被
硬剪枝：

```text
winWorth     = 0.10 + nonLossProb * (1-winProb)
nonLossWorth = 0.10 + (1-nonLossProb) * (1-winProb)
```

可选的 `multiHeadObjectiveSelectionPower` 控制两者权重的尖锐程度；
`multiHeadObjectiveSelectionSharpness>0` 则把普通平均逐渐变成风险偏好的
log-sum-exp。

对外报告的白方胜率仍为：

```text
(whiteWinProb - blackWinProb) * 0.5 + 0.5
```

等价于用户约定的 `(白胜-黑胜)*50%+50%`。

## 5. 要解决的问题

目标不是简单提高平均 policy 命中率，而是缓解两类 MCTS 盲点：

1. **和棋坑**：A 点短搜 100% 和棋，B 点短搜 30% 胜、70% 负。标准 MCTS
   很快停止搜索 B，即使 B 深搜后可能变成必胜。
2. **隐蔽杀棋**：VCT 或其他强制杀成功即速胜、失败即速败。浅层 value
   容易给出很低均值，普通 PUCT 会过早排除。

同时存在相反风险：如果给可疑分支大量计算，最后证明杀不掉，这些 visit
若按普通 visit 平均回传，会污染父节点和其他目标。实验设计因此反复在三件事
之间折中：

- 给低先验但可能翻盘的分支更多机会。
- 没有硬阈值，低概率只是平滑减小预算。
- 辅助验证失败时尽量不拖低主胜率。

## 6. 测试方法

### 6.1 局面基准

研究中使用过两代局面基准：

- v1：约 200 个局面，用 5000 playout 的普通搜索作深层参考。
- v2：800 个局面，按 `vct`、`mustwin`、`drawescape`、`broad` 四类分层。

候选在固定 200 playout 下选点，然后比较该点与基线点在 5000 playout
参考中的 utility。主要指标包括：

- 平均 deep utility gain。
- 改变选点的局面数。
- 与 deep top move 完全一致的局面数。
- 大于 `+0.02` 的改善数和小于 `-0.02` 的恶化数。
- 按来源棋谱聚类 bootstrap 的置信区间。

这个基准适合快速排除明显坏方案，不适合单独证明 Elo。v28 和 v49 都曾在
开发基准上表现很好，正式对局却转负。

### 6.2 正式对局

最终采用的正式协议：

```text
model                 b24_v112.onnx
playouts per move     200 vs 200
games per replication 400
parallel games        400
search threads/game   1 per bot
nnMaxBatchSize        64
hardware              4 x RTX 4090D
```

每个批次必须完整产生 400 盘。预先决定总批次数，全部结束前不读取比分。
每盘检查 `gameHash`，最终大样本中没有重复 hash。

早期还单独测试过一次 1000 playout、400 盘的 VCT policy 混合；结果约
+3.5 Elo，未超过误差。

计分器是 `.codex_score_multi.py`：

```text
score = (wins + 0.5*draws) / games
Elo   = 400 * log10(score/(1-score))
```

正式大样本同时查看逐盘区间和“每 400 盘为一簇”的 t 区间。

## 7. 实验时间线

### 7.1 早期直接混合与引导搜索（v15 之前）

最初直接尝试把辅助 value/policy 加入普通搜索：

- head2/head3 policy 混合。
- head4/head5 VCT policy 混合。
- VCT 胜率按平滑概率控制 policy 混合强度。
- 固定进攻方的 guided playout。
- outcome interval 探索奖励。
- child value gate、visit 衰减、FPU 调整。
- 辅助 policy/value 的 envelope、optimism 和 confirmation。

`.codex_all_match_scores.txt` 保存了这一阶段的 71 条 match 记录。几个典型的
“首测很好、复测消失”：

| 方案 | 首测 | 复测/更大测试 | 判断 |
| --- | ---: | ---: | --- |
| 原始 VCT policy 混合 | +36.6 Elo / 200 | -15.6 Elo / 200；1000p 下 +3.5 / 400 | 噪声 |
| `auxenvelope_v9` | +26.1 / 200 | +5.2 / 200 | 未复现 |
| `guided_vct06_value05` | +22.6 / 200 | +1.7 / 200 | 未复现 |
| `relativeprobe_v8_A` | -33.1 / 200 | +19.1 / 200 | 方差极大 |
| 强 value 混合 `value_c1_p1` | -81.4 / 200 | 未继续 | 明显有害 |
| policy/value 全组合 | -94.3 / 200 | 未继续 | 明显有害 |

这一阶段得到的主要经验是：辅助头不能无条件混进主 policy/value；触发频率、
规则目标和回传统计必须分开。

### 7.2 四目标普通搜索（v15–v19）

#### v15：动态四目标

把目标概念化为：

1. 当前方胜。
2. 当前方不败。
3. 当前方 VCT 胜。
4. 对手 VCT 不胜。

普通两目标使用独立 utility/weight，VCT 使用独立统计平面。动态版本根据
当前胜与不败的不确定程度分配搜索价值。开发对局出现过：

- `foursoft_200p`：+16.5 Elo / 400。
- `c1soft4_200p`：+21.7 Elo / 400。
- `dynamic_200p`：+20.0 Elo / 400。

这些都是单批次开发结果，置信区间仍包含零，后续没有形成可复现收益。

#### v16：目标专属 policy

胜目标使用 head3，不败目标使用 head2，并以
`multiHeadObjectivePolicyMix` 控制混合。`objdyn50` 在 400 盘约 +5.2 Elo，
没有继续放大。

#### v17：目标专属 playout

增加 `multiHeadObjectiveSeparatePlayouts` 和独立 edge visit，希望某个目标
为了验证分支投入的 visit 不直接污染另一个目标。最初 smoke 测试曾出现
明显退化，暴露了预算和聚合实现中的问题；修正后仍未得到实战提升。

#### v18：cross weight 与近 minimax 聚合

增加：

- `multiHeadObjectiveCrossWeight`
- `multiHeadObjectiveValueWeightExponent`
- 不同 selection power

目标是让辅助目标的专用 playout 对其他目标只有可控的 cross credit，并通过
坏子节点降权接近 minimax。开发基准偶有正数，但 400 盘
`minimax_200p` 约 -0.9 Elo。

#### v19：风险偏好 softmax

使用 `multiHeadObjectiveSelectionSharpness` 把两个 selection value 从平均
逐渐变成 log-sum-exp。没有找到稳定优于普通平均的 sharpness。

### 7.3 独立 VCT 预算与根验证（v20–v23）

#### v20：隔离 VCT 平面

增加固定 attacker 的独立 VCT 搜索树面：

- attack/defense 分开预算。
- 基于 VCT 概率、UCB 不确定度和 soft ramp 的预算曲线。
- VCT policy 与普通 policy 可独立混合。
- 可选 validation signal。

在 200 局面基准上多组参数几乎给出相同选点，说明预算变化常常没有传导到
最终 root move。正式对局：

| 候选 | 盘数 | 胜/和/负 | Elo | 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| 普通双目标 `v20_p1e0` vs `c0` | 400 | 103/207/90 | +11.3 | [-12.4,+35.1] |
| 隔离 VCT `v20_decvct` vs `c0` | 400 | 106/208/86 | +17.4 | [-6.2,+41.1] |

两者都没有跨过误差。

#### v21：VCT 候选回普通规则验证

用最多一部分普通 playout 验证 isolated VCT 的最佳 root 候选。
`vrf20a` 正式 400 盘为 +4.3 Elo，区间 `[-19.4,+28.1]`。

#### v22：通用 root minimum visits 与 policy flattening

尝试：

- 给 root 子节点设置最低 visit 漏斗。
- 在 head2/head3 暴露宽和棋区间时压平 head0 policy。

通用 `minv4` 正式 400 盘为 +1.7 Elo，区间 `[-22.5,+26.0]`。

#### v23：只在和棋区间触发最低 visit

`dmin8` 正式 400 盘为 +0.9 Elo，区间 `[-23.2,+25.0]`。
比通用最低 visit 更有针对性，但仍无可测收益。

### 7.4 强制应手系列（v24–v29）

#### v24：root forced reply

核心思路：在和棋或未决局面中，如果某个候选落子后的对手 reply policy
高度集中，则这条线更像强制变化，给该 root child 一个有上限的额外 visit
目标。参数 `multiHeadDrawForcedReplyRootVisits` 控制预算。

开发基准测试 4、8、12、16、24 visits。`fr12` 被选中后做了三次独立
400 盘正式复测：

| 批次 | 胜/和/负 | Elo |
| --- | ---: | ---: |
| 1 | 115/203/82 | +28.7 |
| 2 | 111/195/94 | +14.8 |
| 3 | 105/211/84 | +18.3 |
| 合并 | 331/609/260 | +20.6 |

合并 95% 区间 `[+6.8,+34.4]`，1200 个 hash 全部唯一。这是整轮研究中
唯一稳定显著的结果。

继续把预算从 12 增加到 16，`fr16` 对 `fr12` 为 -2.6 Elo / 400，说明
12 左右已足够。

#### v25：head1 普通 policy ensemble

测试 head1 混入 25%、50%、75%、100%。在 200 局面基准中没有正向均值；
100% head1 显著更差，因此没有进入正式对局。

#### v26：reply concentration 阈值

测试 0.50–0.95。高阈值改变的局面更少，开发基准接近中性，没有比原
`fr12` 更可靠的配置。

#### v27：selection bias

测试 `multiValueHeadSelectionBias` 从 -0.25 到 1.0。偏向“当前方必胜”
没有稳定改善；负偏置尤其容易恶化。

#### v28：向非根节点继续 forced reply

`fr12ft4` 在 200 局面开发基准上平均 deep utility gain 为约 +0.00443，
是当时非常好看的结果；正式 400 盘却为 -6.1 Elo。说明把 root 启发延伸进
整棵树既消耗预算，也放大开发集过拟合。

#### v29：用 head2–head5 consensus gate

用 head3/head4 的进攻共识和 head2/head5 的防守共识筛选 forced reply。
开发基准最佳均值约 +0.0054，但它建立在未能正式复现的 v28 tree 方案上，
因此没有继续投入正式大样本。

### 7.5 sidecar、近 minimax 与候选晋升（v30–v39）

这组实验专门处理“辅助搜索失败后污染主胜率”的问题。

#### v30：普通规则 sidecar

isolated VCT 只负责提出候选，再用独立普通规则小搜索验证。不同验证比例、
共识强度和结束阶段只改变极少数局面，开发均值接近零。

#### v31：强制线近 minimax

对 forced reply 子节点使用不同 value-weight exponent，使聚合更接近
minimax。开发基准的 `fr12ft4e25` 约 +0.00495，但继承了 v28 的同类
过拟合风险，没有正式收益证据。

#### v32：`fr12` 与隔离 VCT 组合

测试 attack/defense budget、validation、normal verification 和 policy
consensus。相对 `fr12` 的开发改动通常为 0–4/200 个局面。
直接 VCT policy 的 400 盘测试约 +3.5 Elo，未超过误差。

#### v33–v35：forced sidecar 与 move-selection weight

依次尝试：

- v33：每个 forced candidate 独立验证。
- v34：把 sidecar 证据作为 root move-selection 权重。
- v35：按 sidecar mean 和 UCB 置信度晋升。

这些方案要么几乎不改招，要么增加权重后只出现恶化点。v35 的所有主要候选
在开发基准上均为负。

#### v36：辅助 policy root seed

使用 h2/h5 和 h3/h4 policy consensus 直接给 root child 少量 seed visit。
`ar4/ar8` 在开发基准上只有约 +0.0006 的微弱均值，没有正式证据。

#### v37–v39：全局、相对和 forced VCT sidecar

- v37：把 isolated VCT 的最佳候选做全局普通规则 sidecar。
- v38：只有 sidecar 相对主 root estimate 更好时才晋升。
- v39：forced reply 与 VCT sidecar 合并，并用 relative selection weight。

大部分参数不改变选点。v39 最好开发均值约 +0.0008，但同时有 3 个明显改善
和 4 个明显恶化，不值得正式扩测。

### 7.6 tactical root proposal（v40–v49）

sidecar 路线触发过少后，改为更简单的 bounded root proposal：
辅助 policy 只提出 root child 的期望最低 visit，不直接覆盖主 policy。

#### v40：四辅助头的并集

`multiHeadTacticalRootVisits` 控制总强度，机会值由辅助 value 置信度和 policy
集中度平滑决定。800 局面中最好候选只改变 6/800，平均 gain 约 +0.00035。

#### v41：contrastive novelty

使用 `multiHeadTacticalContrastiveMix` 优先选择“辅助头高、head0 低”的点，
避免把预算浪费在主 policy 本来就会搜的点。最好候选平均约 +0.00031，
仍只改变 5/800。

#### v42：objective mask

分别启用 h2、h3、h4、h5 组合。结果显示主要正信号来自 h4；h2/h3 和防守头
没有稳定贡献。`tm16h4` 在开发集只改变 2/800，1 个明显改善、0 个恶化。

#### v43：只验证高置信 VCT

对 VCT 胜率和 policy peak 设置很高软门。多数候选 800 个局面一个都不改，
说明门太保守；唯一有变化的配置也只改 1 个点。

#### v44：高置信候选的普通规则 sidecar

再次尝试用 normal-rules sidecar 验证高置信 VCT。计算更贵，触发仍然太少，
没有形成可推广的信号。

#### v45：普通 value disproof

如果普通规则已经认为辅助候选明显差，则随 visit confidence 增加而软停止
验证。大预算候选在开发基准普遍更差；disproof 没能可靠地区分隐藏杀和误报。

#### v46：低概率普通 sidecar

把低 VCT 概率当作软倾向而不是硬剪枝，并扫描概率尺度、policy power、
attack budget、UCB 和晋升权重。最好候选均值约 +0.00021，置信区间跨零；
预算过大或阈值过低明显恶化。

#### v47：policy peak power

只启用 h4，并用 `multiHeadTacticalPolicyPower` 把预算集中到辅助 policy 峰值。
开发基准：

| 候选 | 改招 | 平均 deep gain | 明显改善/恶化 |
| --- | ---: | ---: | ---: |
| `tp16p1` | 2/800 | +0.00030 | 1/0 |
| `tp24p15` | 2/800 | +0.00030 | 1/0 |
| `tp48p2` | 9/800 | +0.00041 | 2/4 |

保守的 `tp16p1` 正式 800 盘为 +3.5 Elo，区间跨零。`tp24p15` 随后成为
主要候选。

#### v48：sidecar confidence 再校准

按 isolated VCT 的 UCB、访问数和 relative weight 扫描。最好候选均值约
+0.00020，改 11/800，但有 4 个明显改善和 3 个恶化，未胜过简单 tp24。

#### v49：peak gap

新增 `multiHeadTacticalPeakGapPower`，偏向辅助 policy 第一名与第二名差距大
的局面。开发集上的 `tg64p2g2`：

```text
changed        2/800
deep exact     451/800
better/worse  2/0
mean gain      +0.00144
bootstrap CI   约 [+0.00000,+0.00420]
```

这是开发基准上最漂亮的候选之一，但正式 800 盘为 -13.0 Elo，
`174/422/204`。该参数随后从 C++ 源码移除。

### 7.7 tp24 大样本与 v50/v51

#### tp24 正式大样本

`tp24p15` 相对 `fr12` 完整跑了 16 个预声明批次，共 6400 盘：

```text
W/D/L       1606/3187/1607
score       49.99%
Elo         -0.1
95% Elo     [-6.1,+6.0]
unique hash 6400/6400
```

候选平均报告 visits 476.4，基线 476.8，计算量基本对称。结论是简单的 h4
root proposal 对整体棋力中性。

`tp24p15 + fr12` 对 `c0` 的两批共 800 盘为 +7.8 Elo，但区间
`[-9.6,+25.3]`，没有证明 tp24 在 fr12 之外提供额外收益。

#### v50：线性 must-win suppression

观察到 tp24 的一个坏例来自普通规则已经 99% 以上必胜时仍重复搜 VCT，
于是尝试：

```text
h4Opportunity *= (1-mustWinProb)
```

新 holdout 上避免了坏例，但在旧 800 局面基准中压制过强，丢掉原本的改善点，
因此淘汰。

#### v51：高阶软门

最终改为：

```cpp
objectiveOpportunity[4] =
  smoothOpportunity(sideWinProb(4)) *
  (1.0 - pow(mustWinProb,8.0));
```

它只在普通规则已经极高置信必胜时明显压低 h4，低中胜率区几乎保持 tp24。

三个战术集的结果：

| 数据集 | 局面 | 改招 | 结果 |
| --- | ---: | ---: | --- |
| 原 800 分层基准 | 800 | 2 | 1 个明显改善，0 个明显恶化 |
| 从正式第 9 批生成的新 holdout | 200 | 1 | deep utility `+0.127116` |
| 从正式第 10 批生成的 fresh holdout | 200 | 1 | deep utility `+0.113982` |

局部机制看起来更安全，但触发率只有约千分之一。最终 14 个预声明批次、
5600 盘相对 `fr12`：

```text
W/D/L          1390/2819/1391
score          49.99%
Elo            -0.1
game-level CI  [-6.5,+6.4]
batch-level CI [-8.4,+8.3]
unique hash    5600/5600
```

因此 v51 是“局部更干净，但总体仍中性”，不能宣称棋力提升。

## 8. 为什么很多方案失败

### 8.1 辅助信号过于稀疏

最稳妥的 tactical 候选通常只改变 2/800 个开发局面。即使这两个点都正确，
实战 Elo 也小到难以测量。

### 8.2 放大预算会先放大误报

从 16/24 visits 增长到 40/64 visits 时，改变的点变多，但明显恶化也迅速增加。
VCT value 高并不等于该 root move 在普通规则下是好棋。

### 8.3 规则目标与普通胜率不是同一量

head2/head3/head4/head5 在各自规则下有意义，但把它们直接当成普通规则 value
会发生校准偏差。尤其 VCT 失败即速败的目标，均值和主游戏胜率的关系很非线性。

### 8.4 visit 污染只是问题的一部分

独立 VCT 平面和 sidecar 可以避免直接污染普通均值，但还会：

- 消耗固定 200 playout 中的主搜索预算。
- 通过候选晋升或最终选点权重间接造成误报。
- 因触发太保守而完全不改招。

### 8.5 开发集选择偏差

扫描几十个参数后挑最好的 1 个，即使每个候选真实均值为零，也很容易选到
表面正向者。典型例子：

- v28 `fr12ft4`：开发基准约 +0.00443，正式 -6.1 Elo。
- v49 `tg64p2g2`：开发基准 2/0 改善，正式 -13.0 Elo。
- tp24 前半 3200 盘为正，完整 6400 盘回到零。

所以后期改用新棋谱生成 holdout，并坚持预声明完整盘数。

## 9. 当前分支保留了什么

### 9.1 默认行为

所有高风险实验功能默认关闭：

```text
multiValueHeadUtilityMix              0
multiHeadObjectiveSearchStrength      0
multiHeadNormalPolicyHead1Mix         0
multiHeadObjectivePolicyMix           0
multiHeadVctMaxAttackProp             0
multiHeadVctMaxDefenseProp            0
multiHeadVctNormalPolicyMaxMix        0
multiHeadDrawWinNormalPolicyMaxMix    0
multiHeadDrawLossNormalPolicyMaxMix   0
multiHeadTacticalRootVisits           0
multiHeadDrawForcedReplyRootVisits    0
```

因此仅加载 v112 ONNX、但不配置多头搜索参数时，搜索继续使用 head0 兼容路径。

### 9.2 值得保留的基础设施

- v112 ONNX 三后端输出解析。
- 六 value/policy 头的 `NNOutput` 存储与调试输出。
- 白胜和黑胜反目标的独立 utility/weight。
- objective-specific edge visits。
- 白/黑固定 attacker 的 isolated VCT stats。
- 普通规则与 VCT 规则状态分离。
- 完整配置解析、范围校验和 GTP 输出。
- `fr12` bounded root forced-reply 机制。
- 默认关闭的 v51 soft gate，作为后续触发器研究基线。

### 9.3 不应视为已证明有效的部分

当前 `SearchParams` 中保留了大量实验旋钮。除 `fr12` 外，它们大多只经过
开发基准或不显著正式测试。合并到长期维护分支前，应按产品目标删减或放在
明确的实验开关后，不应把“参数存在”理解成“参数已调好”。

`multiHeadTacticalPeakGapPower` 已因 v49 正式退化而从源码移除。

## 10. 代码位置

| 内容 | 主要文件 |
| --- | --- |
| v112 输入版本映射 | `cpp/neuralnet/modelversion.cpp` |
| 六头 `NNOutput` 存储与复制 | `cpp/neuralnet/nninputs.h`, `nninputs.cpp` |
| 每头 policy softmax/value 后处理 | `cpp/neuralnet/nneval.cpp`, `nneval.h` |
| TensorRT ONNX | `cpp/neuralnet/trtbackend.cpp` |
| ONNX CPU | `cpp/neuralnet/onnxbackend_cpu.cpp` |
| ONNX DirectML | `cpp/neuralnet/onnxbackend_directml.cpp` |
| 非 ONNX v112 拒绝 | `cpp/neuralnet/desc.cpp`, `eigenbackend.cpp`, `cudabackend.cpp`, `openclbackend.cpp` |
| 搜索参数定义和默认值 | `cpp/search/searchparams.h`, `searchparams.cpp` |
| config 解析和校验 | `cpp/program/setup.cpp`, `cpp/program/gtpconfig.cpp` |
| 节点普通/VCT 统计 | `cpp/search/searchnode.h`, `searchnode.cpp` |
| utility 与动态目标公式 | `cpp/search/searchhelpers.cpp` |
| PUCT、policy 混合、root proposal | `cpp/search/searchexplorehelpers.cpp` |
| playout 调度和 isolated VCT | `cpp/search/search.cpp`, `search.h`, `searchnnhelpers.cpp` |
| 多套权重聚合 | `cpp/search/searchupdatehelpers.cpp` |
| 对外报告与最终选点 | `cpp/search/searchresults.cpp`, `cpp/command/gtp.cpp` |

## 11. 复现入口

### 11.1 构建

Windows TensorRT：

```powershell
cmake --build cpp\build --config Release --target katago -j 8
cpp\build\Release\katago.exe version
```

Linux 构建包装：

```powershell
cmd /d /c .codex_build_linux.cmd
```

最终正式 Linux v51 二进制 SHA-256：

```text
390282ad2eb23fef9b944c59ae5ab005825ffd9105686d3908beea0d6b3c2979
```

### 11.2 关键正式脚本

```text
.codex_run_forced_reply_v24_match.sh
.codex_run_tactical_peak_formal_match.sh
.codex_run_tp24_final_reps.sh
.codex_run_tp24_softgate_formal.sh
.codex_run_tp24_softgate_extension.sh
.codex_finalize_tp24_softgate.sh
```

### 11.3 评分

```powershell
python .codex_score_multi.py --expect-games 400 <sgf-directory>=<candidate-name>
```

`<sgf-directory>` 必须直接包含 `*.sgfs`；计分脚本不递归扫描。不同批次解包后，
该目录可能是 `<run>/sgf`，也可能是 `<archive>/runs/<run>/sgf`。

脚本会输出：

- 胜/和/负。
- score、Elo 和 95% 区间。
- 执着棋二项检验。
- 黑白分色结果。
- game hash 重复数。
- 双方平均报告 visits。

### 11.4 关键原始数据

```text
.codex_forced_reply_v24_fr12_vs_c0_400g.tar.gz
.codex_forced_reply_v24_fr12_vs_c0_400g_rep2.tar.gz
.codex_forced_reply_v24_fr12_vs_c0_400g_rep3.tar.gz

.codex_tp24p15_formal1_vs_fr12_400g.tar.gz
...
.codex_tp24p15_formal8_vs_fr12_400g.tar.gz
.codex_tp24p15_formal9_16_vs_fr12_400g.tar.gz

.codex_tp24softgate_formal1_14_vs_fr12_400g.tar.gz
.codex_tp24softgate_formal1_14_launcher_logs.tar.gz
.codex_tp24softgate_v51_results.tar.gz

.codex_sharp_holdout_tp24_new.tar.gz
.codex_sharp_holdout_tp24_fresh2.tar.gz
```

这些大文件当前主要是本地实验资产；若要长期保存在 Git 中，应先决定是否使用
Git LFS 或外部对象存储。

## 12. 验证状态与限制

已完成：

- Windows TensorRT 构建。
- Linux TensorRT 构建。
- Windows `version` 烟测，报告 KataGo v1.12.4。
- `git diff --check`，仅有工作树换行符提示。
- 多轮 200/400 盘开发比赛。
- 1200 盘 `fr12` 正式复测。
- 6400 盘 tp24 正式复测。
- 5600 盘 v51 正式复测。
- 正式对局 hash 唯一性和黑白分色审计。

限制：

- 当前构建没有 `runtests` 子命令，也没有单独生成测试可执行文件。
- 没有为每个新 `SearchParams` 组合增加单元测试。
- DirectML/CPU 路径完成了代码和构建接入，但大规模棋力测试使用的是 TensorRT。
- 开发基准的 deep 5000 playout 仍只是更强近似，不是真实 ground truth。
- 绝大多数实验参数没有足够正式盘数支持。

## 13. 后续建议

优先级从高到低：

1. **保留并单独复核 `fr12`**：换开局集、换随机种子、换模型再做预声明复测，
   确认 +20 Elo 是否跨模型泛化。
2. **学习型触发器**：用 head0/head4 value 差、policy rank、peak gap、
   局部强制程度和浅层验证结果训练“是否值得启动 sidecar”，替代手工阈值。
3. **真正隔离的战术预算**：sidecar 的失败结果不回传主 utility，只在达到
   预注册置信标准后把候选交给固定普通规则验证。
4. **配对局面评估**：对同一 root position、同一随机种子比较候选与基线，
   先测“是否改变根决策”，再投入完整自对弈。
5. **独立开发集、验证集、正式集**：参数选择只看开发集；验证集只做一次；
   正式比赛预先固定总盘数。
6. **减少参数面**：把已经证明中性或有害的 guided、sidecar promotion、
   peak-gap 等组合从长期接口中清理，降低维护和误配置风险。
7. **补测试**：至少覆盖 v112 shape、每头 softmax、旧模型复制、NNOutput
   深拷贝、`c=0` 严格兼容、白黑符号对称和非 ONNX 明确报错。

一句话概括：六个头有信息，但目前缺的不是更多手调混合系数，而是一个可靠的
“何时相信哪个辅助头、以及如何在不污染主树的情况下验证它”的机制。
