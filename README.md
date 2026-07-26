# PEAD Baseline 数据文档

## 研究目的

**最终目标**（Design Doc）：检验 **LLM 能否从财报公告当日的信息流中提炼出超出 SUE 的增量信息，
且该信息被市场吸收的速度是否受当时的信息处理能力约束**。三个 channel 分别对应三种处理能力来源 ——
注意力稀缺（ATT）、GenAI 采用（ADOPT）、AI 停机（OUT），核心检验对象是交互项系数 β₅（M × LLM）。

**本仓库完成的是该研究的 baseline 部分**：把 PEAD 这个被检验对象本身先立住 ——

1. 从 WRDS 原始数据构造事件级面板（CAR / SUE / 10 个控制变量），**517,955 个公告事件，1996–2026**
2. 复制 Bernard & Thomas (1989) 的 PEAD 异象（组合排序 + 事件时间曲线）
3. 跑通 Hirshleifer, Lim & Teoh (2009) 的截面回归设定（SUE + 控制变量，两窗口 × 两收益口径）

只有 baseline 成立，后面加 moderator 与 LLM signal 的交互项才有意义 ——
若 PEAD 本身在我们的样本上就不存在，β₅ 无从谈起。

---

## 什么是 PEAD

**PEAD**（Post-Earnings-Announcement Drift，盈余公告后漂移）指：
盈余公告后，股价会继续朝盈余意外的方向漂移约 60 个交易日。

这是一个**异象（anomaly）**，因为它违反半强式有效市场 —— 公告信息已完全公开，
价格理应瞬间调整完毕。可交易的含义是：在公告**之后**买入高 SUE 组、卖空低 SUE 组，
仍能赚到显著的异常收益。

文献用两种方式记录它：

| 方式 | 做法 | 出处 |
|---|---|---|
| **组合排序** | 按盈余意外分十组，跟踪各组公告后**逐日**的累计异常收益 | BT (1989) fig.1–2 / FOS (1984) |
| **截面回归** | `CAR_drift` 对 SUE 回归，控制公司特征与固定效应 | HLT (2009) Sec IV.A.1 |

### 组合排序怎么分组

**单变量排序 —— 只按 SUE 分十组**，不是"先按规模再按 SUE"的双重排序：

1. 对每个公告事件算出 $SUE=(e-F)/P$
2. 按**公告日所在日历季度**把当季所有事件的 SUE 排序，切成十等份 ——
   `sue_dec = 1` 是最负的盈余意外，`sue_dec = 10` 是最正的
3. 对每一组，算公告后各交易日的**平均**累计异常收益，画成十条曲线
4. **D10 − D1** 就是多空对冲组合的收益，即这个异象的可交易规模

**规模的影响不在分组里处理，而在收益端**：CAR 的基准是该公司所属的
size × B/M **25 组合之一**（Fama-French 双重排序），个股收益减去同类组合收益，
规模与价值效应已经被扣掉。BT (1989) 用的是仅按规模调整的收益，我们这里更严格一层。

**分位断点的两种口径**（`回归准备.ipynb` §A.5 两种都跑了）：

| | 断点来源 | 被排序的数值 | 适用场景 |
|---|---|---|---|
| 当季分位（**主设定**，HLT 2009） | 当季所有事件的 SUE 分布 | 当季 SUE | SUE 作为回归控制变量，不需要可实施性 |
| 上季断点（**补充设定**，FOS 1984 / BT 1989） | **上一季度**的 SUE 分布 | **仍是当季 SUE** | 评估**可交易策略** —— 投资者在当季公告时只知道上季分布 |

关键在于：**只有切分点回退一个季度，被排序的数值始终是当季的 SUE**。
公告发生的当下，本季度的截面分布尚未形成，用它切分等于用到了同季其他公司
还没公布的盈余（一种温和的前视）；上季断点才对应一个真能执行的策略。

实测两者一致率 78.9%、相差 ≤1 档占 98.9%，D10−D1 的纯漂移 4.25% vs 4.31%（C2C）、
5.42% vs 5.52%（O2O），结论不受影响。**两种分位只用于 Part A 的组合排序，
Part B/C 的回归一律用当季分位。**

两点实现细节：
- **1996Q1 也有上季断点**：I/B/E/S 从 1995 年起有预测，1995Q3/Q4 的公告虽然因交易日历
  不完整而没有 CAR、不进图表，但它们的 SUE 可以拿来当断点。只有 385 个 1995Q3 的事件
  确实没有可用的上一季（它们本来也没有 CAR）。
- **上季断点下十分位不再等分**：9.0% 的事件 SUE 恰好为 0（分析师共识被精确命中）。
  当季分位用 `rank(method="first")` 把这些并列值摊到 D4–D6，上季断点是拿数值去
  `searchsorted`，并列值只能整块落进同一组，所以中性组偏大。D1 / D10 两端不含并列值，
  多空价差不受影响。

**为什么要同时看两个窗口**

- `CAR^ANN`（公告日 + 次日）衡量**即时反应**：市场当场消化了多少
- `CAR^DRIFT`（公告后第 2–61 个交易日）衡量**滞后反应**：还有多少留到后面

只有 DRIFT 显著为正才叫 PEAD。若市场完全有效，DRIFT 的系数应为 0。

---

## Baseline 回归设定

### 主设定

下表是本项目 baseline 的**完整设定**，后面接 moderator（ATT / ADOPT / OUT）与 LLM signal 时
一律沿用这一套。其他变体都在 §7 作为稳健性检验列出。

| 组成部分 | 取值 |
|---|---|
| **样本** | 1996-01-02 起的盈余公告，要求：有 SUE 十分位、10 个控制变量齐全、CAR 可得、同日多季报只留最新一期（`is_latest_pends_on_day`）、剔除异常报告滞后（`flag_lag_bad`） |
| **被解释变量** | $CAR[0,1]$（ANN）与 $CAR[2,61]$（DRIFT），buy-and-hold 个股收益减 size×B/M 25 组基准，**小数**计。C2C 与 O2O 各一套，共 4 条回归 |
| **核心自变量** | `sue_rank` —— 由原始 `sue` = `(e−F)/P` 按**当季**截面排出十分位 `sue_dec`（1–10），再取 $(sue\_dec-1)/9$ 映到 $[0,1]$。进回归的是 `sue_rank` |
| **控制变量** | 10 个，定义与 HLT 2009 §III.A **逐条一致**：SIZE、BM、LNANALYST、LAG、LAG²、LAG³、IO、EVOL、EPERSIST、TURN |
| **固定效应** | 年 + 月 + 星期几 + FF10 行业 |
| **标准误** | 按**公告日**聚类（CRV1） |
| **变量处理** | 原始值直接进回归 |
| **两个设定** | **(0a)** 控制变量不与 SUE 交互；**(0b)** 10 个控制变量全部与 SUE 交互（交互前先中心化） |

**固定效应、标准误、变量处理都与 HLT 2009 Table III 一致。**
$\beta_1$ 回答的是"好消息公司 vs 坏消息公司"的横截面差异 —— PEAD 作为可交易异象，
交易者正是跨公司买卖，这个变异就是收益本身；公司层面的差异由 10 个控制变量
（含 SIZE、BM、IO、分析师覆盖）直接控住。

### 与 HLT 2009 的全部差异

设定层面（固定效应、标准误、变量处理、10 个控制变量的定义、SUE 的构造与当季分位、
CAR 的两个窗口与 size×B/M 25 组基准）已经完全一致。剩下的差异分三类：

**① 会让数字对不上的**

| | 本项目 | HLT 2009 | 影响 |
|---|---|---|---|
| **样本期** | 1996-01-02 – 2026 | 1995–2004 | **最主要**。PEAD 在 2000 年后大幅衰减，这是漂移端系数只有原文约五分之一的主因（§D 的 "Post-2000 only" 一行可以看到同向的衰减） |
| **基准组合** | 自建 25 组：NYSE 断点、**等权**、自己的 universe | Kenneth French 网站的 25 组，**市值加权** | 等权基准里小盘股权重高，而小盘股漂移更强，扣掉后剩余的异常收益偏小 —— 与样本期效应同向叠加 |

自建基准是硬约束：O2O 口径 French 没有提供，两个口径必须同源才可比；
且 French 的数据发布有滞后，覆盖不到 2026。断点用 NYSE 股票计算这一条与 French 相同。

**② 我们额外加的，不影响与论文的可比性**

| | 说明 |
|---|---|
| **O2O 口径** | 论文只有 close-to-close，我们同时算 open-to-open，两套独立不混用 |
| **设定 (0a)** | 论文主表（Table III 第 2、4 列）只有控制变量与 FE 全交互的版本，对应我们的 **(0b)**；(0a) 是我们额外报的简化版 |
| **样本筛选** | 额外剔除 `flag_lag_bad`（报告滞后 < 0 或 > 180 天）、同一公司同一天多个财季只留最新一期（`is_latest_pends_on_day`）—— 论文未描述这两种情况的处理 |

**③ 纯刻度与单位，不改变任何结论**

| | 本项目 | HLT 2009 |
|---|---|---|
| SUE | `sue_rank` = (十分位−1)/9 ∈ [0,1] | FE = 十分位整数 1–10 |
| CAR | 小数 | 百分点 |
| EVOL | 美元/股 | Table I 报 2.15%，看起来按股价标准化 |
| IO | 小数（0.47） | 百分数（46.97%） |

SUE 与 CAR 是单调线性变换，t 值完全相同，$\beta_1$ 差 $100/9$ 倍；
`回归准备.ipynb` §B.6 有一张完全照论文刻度的对照表。
EVOL 与 IO 的单位只影响这两个控制变量自己的系数刻度，不影响 $\beta_1$。

### 方程

$w \in \{ANN,\ DRIFT\}$，收益口径 $\in \{C2C,\ O2O\}$ —— 每个设定 4 条回归。

**(0a) 不带交互项**

$$CAR^w_{i,d}=\alpha+\beta_1 SUE^{rank}_{i,d}+\sum_{k=1}^{10}\gamma_k X_{k,i,d}+\eta_t+\psi_j+\varepsilon_{i,d}$$

**(0b) 加入控制变量与 SUE 的交互项**

$$CAR^w_{i,d}=\alpha+\beta_1 SUE^{rank}_{i,d}+\sum_k\gamma_k X_{k,i,d}+\sum_k\delta_k\left(X_{k,i,d}\times SUE^{rank}_{i,d}\right)+\eta_t+\psi_j+\varepsilon_{i,d}$$

| 记号 | 含义 |
|---|---|
| $X_{k}$ | 10 个控制变量：SIZE、BM、LNANALYST、LAG、LAG²、LAG³、IO、EVOL、EPERSIST、TURN（定义见 §1.1） |
| $\eta_t$ | 年 + 月 + 星期几固定效应 |
| $\psi_j$ | 行业固定效应，Fama-French 10 分类（由 SIC 映射） |
| $SUE^{rank}$ | 进回归的核心自变量 = $(sue\_dec-1)/9 \in [0,1]$，由当季十分位映射而来 |
| $\beta_1$ | **核心系数**。因为 $SUE^{rank}$ 从 0 走到 1 正好是 D1 到 D10，$\beta_1$ 直接读作「D10 相对 D1 的 CAR 差异」 |

**为什么回归里用十分位 rank 而不是原始 SUE 值** —— 这是 HLT 2009 §IV.A.1 的明确选择，原文：

> "Because the relation between announcement-day abnormal returns and earnings surprise is
> **highly nonlinear** (e.g., Kothari (2001)), with small negative surprises having big effects,
> we use the **decile rank of forecast error** as opposed to the forecast error itself following past
> literature. This **reduces the influence of outliers**, and the relation between CAR[0,1] and the
> earnings surprise deciles is **almost linear**."

三个理由：① 原始 $(e-F)/P$ 分布极厚尾，少数极端值主导系数；② CAR 对原始 SUE 高度非线性、
对十分位却几乎线性（论文 Figure 1）；③ 断点按**日历季度**重算，自动吸收盈余意外分布的时间漂移。

**刻度**：论文的 FE 是 **1–10 的整数**（Table III 表注 "FE = 1: lowest, 10: highest"），
系数读作"每升一档多少个百分点"；我们除以 9 缩放到 [0,1]，是单调线性变换 —— t 值完全相同、
$\beta_1$ 差 9 倍，好处是直接读成 D10−D1 的价差。原始连续 SUE 只在稳健性一节单独跑一行。
`回归准备.ipynb` §B.6 有一张完全照论文刻度的对照表。

标准误按**公告日**聚类，与 HLT 2009 相同。
交互项中的控制变量已中心化，故 (0b) 的 $\beta_1$ 是控制变量取均值处的效应，与 (0a) 可比。

**控制变量的主效应 $\gamma_k$ 在两个设定里都估计、也都报告**：Table 1 全部列出，
Table 2 分上下两栏 —— 上栏是 SUE 与 10 个交互项 $\delta_k$，下栏是同样 10 个控制变量的主效应。
$SUE$ 在回归里始终是 **`sue_rank` = (十分位 − 1)/9 ∈ [0,1]**，不是原始连续值；
原始连续 SUE 只在 §D 的稳健性里单独跑一行（系数量纲不同，只比符号与显著性）。

---

## 主要结果

**组合排序**（343,609 个事件，D10 − D1）：

| | 公告窗口 [0,1] | 纯漂移 [2,61] |
|---|---|---|
| C2C | +8.09% (t = 82.6) | **+4.25% (t = 14.6)** |
| O2O | +7.30% (t = 86.5) | **+5.42% (t = 18.3)** |

对照 BT (1989) 报告的 60 日对冲收益 6.31%（1974–1986 样本），方向一致、量级同级。

**截面回归**（设定 0a，$\beta_1$，N = 302,952）：

| 窗口 | C2C | O2O |
|---|---|---|
| ANN | 0.0796 (t = 110.7) | 0.0731 (t = 119.6) |
| **DRIFT** | **0.0291 (t = 15.8)** | **0.0388 (t = 20.3)** |

估计用 `pyfixest`（Python 版 `reghdfe`），标准误按公告日聚类。

加入行业/日历固定效应与 10 个控制变量后，DRIFT 的系数依然显著为正 ——
**PEAD 在 1996–2026 的样本上成功复制**。

交互项（设定 0b）显示 `IO × SUE`、`SIZE × SUE`、`TURN × SUE` 在 DRIFT 窗口全部显著为负：
机构持股越高、公司越大、流动性越好，漂移越弱 —— 支持"套利受限"而非"风险补偿"的解释。

---

**最终产出**：`build/pead_panel.parquet` — 事件级大表，517,955 行 × 75 列
**目录**：`/project/dachxiu/zan/PEAD/`

---

## 0. 总览

### 0.1 三个阶段

```
阶段一  WRDS 下载        new_WRDS_Data_Download.ipynb  （本地跑）→ data/   102 个 parquet
阶段二  收益面板导出      export_returns.py             （midway 跑）→ export/ 1 个 parquet
阶段三  数据准备 8 步     数据准备.ipynb                 （midway 跑）→ build/  16 个文件
阶段四  组合排序 + 回归    回归准备.ipynb                 （midway 跑）→ build/  结果表与图
```

**所有数据处理都在 notebook 里**，只有阶段二是独立 py 脚本 —— 因为它要单遍流式扫描
yifei 的 16 GB O2O 面板，跑一次就够，没必要每次执行 notebook 都重跑。

### 0.2 文件清单

| 文件 | 性质 | 说明 |
|---|---|---|
| `new_WRDS_Data_Download.ipynb` | 生产 | WRDS 拉数，9 节，逐表带字段说明 |
| `export_returns.py` | 生产 | 合成 C2C + O2O 日收益面板（跑一次） |
| `数据准备.ipynb` | 生产 | 8 步流水线，从原始数据到大表 |
| `回归准备.ipynb` | 生产 | 组合排序 + 事件时间图 + baseline 回归（论文格式表） |
| `回归结果补充.ipynb` | 生产 | 按 channel 组织的回归：ADOPT（GenAI 采用）、ATT（注意力）；§4 记录 LLM signal 的来源、处理思路与列字典 |
| `build_llm_signal.py` | 生产 | LLM signal：新闻级预测 → firm-day → 事件窗口，两层产出（约 6 分钟） |
| `build_car_path.py` | 生产 | 事件时间 CAR 路径（343,609 事件 × 122 天 × 2 口径 × 2 种分位，约 20 分钟） |
| `verify_readme.py` | 工具 | 逐条核对本文档引用的数字与实际产物是否一致（84 项，含回归结果）
| `audit_panel.py` | 工具 | 审计大表的内部逻辑：恒等式、前视偏差、取值域、重复（35 项） |
| `数据view.ipynb` | 草稿 | 临时查看数据用 |
| `export_run.log` | 日志 | `export_returns.py` 的运行记录 |

### 0.3 运行环境与依赖

**解释器**：`/home/zan1/envs/nlp3/bin/python`（Python 3.11），notebook 内核 `nlp3`。

| 包 | 版本 | 用途 |
|---|---|---|
| `pandas` | 2.2.3 | 全流程的数据处理；`merge_asof` 做时点对齐、`rolling` 做滚动窗口统计 |
| `numpy` | 1.26.4 | CAR 的累积对数收益、分位断点、手写 OLS 的矩阵运算 |
| `pyarrow` | 23.0.0 | parquet 读写；`iter_batches` 流式扫描 16 GB 的 O2O 面板，避免整表进内存 |
| **`pyfixest`** | **0.60.0** | **回归估计**。Python 版的 Stata `reghdfe` / R `fixest`：多维固定效应交替投影吸收 + 聚类稳健标准误。见 §4 与 `回归准备.ipynb` §B.1 |
| `matplotlib` | 3.10.8 | PEAD 事件时间图（两张） |
| `scipy` | 1.17.0 | `pyfixest` 的依赖；本项目代码不直接调用（t 检验用 numpy 手算） |

**为什么回归要用 `pyfixest` 而不是 `statsmodels`**：四组固定效应（年 / 月 / 星期 / FF10）
加上 (0b) 的 20 个交互项，`statsmodels` 要显式构造哑变量矩阵；`pyfixest` 用交替投影
把固定效应吸收掉，且原生支持聚类稳健方差，与 Stata 的 `reghdfe` 同算法。
自由度与聚类口径与 `reghdfe` 一致，`回归准备.ipynb` §B.5 用一份独立的手写实现
（numpy 直接做 FWL 去均值 + 聚类方差）交叉验证过，最大系数偏差 6e-07。

安装：`pip install pyfixest`（其余包环境里已有）。

### 0.4 样本期

**样本期：1996-01-02 ~ 2026-05-14**

| 层次 | 起 | 止 | 决定因素 |
|---|---|---|---|
| **事件（有 CAR）** | **1996-01-02** | **2026-05-14** | 起点 = CRSP 日频起点；止点 = I/B/E/S `anndats` 最大值 |
| 回归样本 | 1996-01-02 | 2026-05-14 | 上述 ∩ 控制变量齐备 |
| **满窗** CAR · C2C | 1996-01-02 | **2026-04-01** | 见下 |
| **满窗** CAR · O2O | 1996-01-02 | **2025-12-29** | 见下 |

**什么是"满窗"（full window）**

`CAR^DRIFT` 要用公告后**第 2 到第 61 个交易日**共 60 天的收益。
若公告日太靠近数据末端，后面的交易日还没发生或还没入库，窗口就被截断 ——
CAR 是在不足 60 天上算出来的，与其他事件不可比。
`n_days_drift_*` 记录实际可用天数，`flag_short_window` 标记不足的（占 3.3%）。

两个口径的收益数据末端不同，所以满窗的最后一个公告日也不同：

| 口径 | 收益数据止于 | 倒推 61 个交易日 → 最后一个满窗公告日 |
|---|---|---|
| C2C | 2026-06-30（CRSP 日频） | **2026-04-01** |
| O2O | 2026-03-30（yifei v5） | **2025-12-29** |

**即：要求 DRIFT 满 60 天时，O2O 的样本最多到 2025 年底，C2C 可到 2026 年 4 月初。**
不要求满窗则两者都能用到 2026-05-14。主设定保留未满窗事件（其 CAR 仍无偏，只是方差更大），
`回归准备.ipynb` §D 有一档剔除它们的稳健性检验，结论不变。

**起点为什么是 1996**：CRSP 日频从 1996 下载（CAR 窗口只往公告后看，无需更早）。
这也与新闻语料的起点一致，后续接 LLM signal 时样本期天然对齐。
更早的数据（funda 1993 / msf 1994 / fundq 1992 / IBES 1995）**只用于构造**
1996 年事件所需的回溯量（BE、ME、16 季 EPS、60 天预测窗口），本身不产生事件。

⚠️ **1995 年的 5,628 个公告已剔除**：它们早于交易日历起点，`td0` 会被强行落到日历首日
（1996-01-02），CAR 窗口与公告相隔数月，纯属噪音。见 §6.10。

### 0.5 数据流

```
WRDS 11 张表 ──┐
               ├─→ Step 1  股票池 + 3 张连接表 ─────────────────┐
CRSP 日/月频 ──┤                                                │
               ├─→ Step 2  BE/ME/BM → 25 组 → 基准组合日收益 ──┤
Compustat ─────┘                                                │
                                                                ├─→ Step 7 合并
I/B/E/S actuals ─→ Step 3  事件表 (517,955) ────────────────────┤   → pead_panel
                              │                                 │
                              ├─→ Step 4  CAR（4 个变量）───────┤
I/B/E/S detail ───────────────┼─→ Step 5  SUE ──────────────────┤
                              │                                 │
13F + Compustat ──────────────┴─→ Step 6  10 个控制变量 ────────┘
                                                                 → Step 8 数据字典
```

### 0.6 `build/` 产物清单

19 个 parquet + 2 个 csv + 4 张图。除最终大表外，其余都是各步的中间产物 ——
独立存盘是为了**改某一个变量的定义时只需重跑对应步骤**，不必全量重建。

| 文件 | 由哪步产生 | 行数 | 粒度（主键） | 内容 |
|---|---|---|---|---|
| `universe.parquet` | Step 1 | 193,968 | permno × 信息区间 | 股票池：交易所、SIC、CUSIP，以及 `is_common`/`is_us`/`in_universe`/`is_nyse` 等筛选布尔列 |
| `link_ccm.parquet` | Step 1 | 33,324 | gvkey × 链接区间 | PERMNO ↔ GVKEY，含 `linkdt`/`linkenddt` 有效期 |
| `link_ibes.parquet` | Step 1 | 30,080 | ticker × 链接区间 | I/B/E/S ticker ↔ PERMNO，含 `sdate`/`edate` |
| `port25_membership.parquet` | Step 2 | 135,230 | permno × formation 年 | 25 组成员表：`me_june`/`me_dec`/`be`/`bm_raw`/`size_q`/`bm_q`/`port25` |
| `port25_bench_returns.parquet` | Step 2 | 191,825 | 交易日 × port25 | **基准组合日收益**：`bench_ret_c2c`/`bench_ret_o2o` 及各自成员数（7,673 天 × 25 组） |
| `events.parquet` | Step 3 | 517,955 | permno × 公告日 × 财季 | 事件表：`anndats`/`td0`/`td0_idx`/`pends`/`actual_eps`/`lag` 及各类标记列 |
| `car.parquet` | Step 4 | 517,955 | eid | 四个 CAR 及其分项：`stk_*`（个股）、`bench_*`（基准）、`n_days_*`（窗口天数）、`port25` |
| `sue.parquet` | Step 5 | 517,955 | eid | `sue`/`sue_dec`/`consensus_f`/`price_adj`/`prc_unadj`/`n_analyst_sue`/`flag_sue_dropped` |
| `ctrl_size.parquet` | Step 6 | 260,320 | permno × formation 年 | `me_jun`（6 月末市值）、`size_dec` |
| `ctrl_bm.parquet` | Step 6 | 135,230 | permno × formation 年 | `be`、`bm_raw`、`bm_dec` |
| `ctrl_turn.parquet` | Step 6 | 2,871,577 | permno × 月 | `turn`（过去 12 个月月均换手率） |
| `ctrl_io.parquet` | Step 6 | 889,973 | permno × 13F 报告期 | `io`（机构持股比例） |
| `ctrl_evol_epersist.parquet` | Step 6 | 698,171 | permno × 财季 | `evol`、`epersist` |
| `ctrl_lnanalyst.parquet` | Step 6 | 443,648 | eid | `n_analyst_cover`（公告前 365 天覆盖分析师数） |
| `ctrl_att.parquet` | Step 6.2 | 517,955 | eid | `n_ann_day`（同日全市场公告家数）、`nrank`（季度内十分位）、`att` = 11 − NRANK |
| `llm_daily.parquet` | `build_llm_signal.py` | 2,296,215 | permno × 交易日 | `td_idx`、`n_news`、`pred_mean`（当天所有新闻 `pred` 的平均 = 这一天的 LLM signal） |
| `llm_signal.parquet` | `build_llm_signal.py` | 517,955 | eid | 各窗口的 `llm_sum` / `llm_mean` / `llm_ndays` / `llm_n` / `llm_rank`。窗口后缀编码偏移量：`0_1` = [d, d+1]，`m1_1` = [d−1, d+1] |
| **`pead_panel.parquet`** | **Step 7** | **517,955** | **eid** | **最终大表，75 列**，上述全部合并 + LAG 高阶项 + 日历键。列字典见 §5 |
| `car_path.parquet` | `build_car_path.py` | 4,880 | 分位口径 × 收益口径 × 十分位 × 事件日 | 事件时间路径：`grouping`（`current` 当季断点 / `prior` 上季断点）/`conv`/`sue_dec`/`event_day`（−60…+61）/`car`/`n`，三张 PEAD 图的数据源 |

**非 parquet 产物**

| 文件 | 由哪步产生 | 内容 |
|---|---|---|
| `data_dictionary.csv` | Step 8 | 大表 75 列的列名 / dtype / 覆盖率 / 中位数 / 说明（机读版） |
| `baseline_results.csv` | `回归准备.ipynb` §E | 8 条 baseline 回归的 β₁ / 标准误 / 两种聚类的 t 值 / N / within R² |
| `pead_paths.png` | `回归准备.ipynb` §A.3 | 公告后 [0,61] 的十分位累计 CAR 曲线（C2C + O2O） |
| `pead_paths_full.png` | `回归准备.ipynb` §A.3 | 仿 BT (1989) fig.2：公告前后分两段各自从 0 起算 |
| `pead_paths_breakpoints.png` | `回归准备.ipynb` §A.5 | 当季断点 vs 上季断点（FOS）的路径对照，C2C / O2O 各一行 |
| `car_path_events.npz` | `build_car_path.py` | 逐事件路径矩阵（343,609 × 122 × 2 口径，280 MB）。换一种分组时可直接重算均值，不必重扫收益。已在 `.gitignore` 中 |

**`eid`** 是事件唯一编号（= `events.parquet` 的行号），`car` / `sue` / `ctrl_lnanalyst` 都用它对齐，
可以据此回溯任何一个事件在各中间表里的原始值。

**注意**：`build/*.parquet` 未纳入版本库（`.gitignore`），因为可由代码完全重建；
`build/*.csv` 与 `build/*.png` 体积小且是结果，已提交。

---

## 1. 术语与缩写

### 1.1 回归变量

| 缩写 | English | 中文 | 定义 |
|---|---|---|---|
| **CAR** | Cumulative Abnormal Return | 累计异常收益 | 个股 buy-and-hold 收益 − 同 size×B/M 组合 buy-and-hold 收益 |
| **ANN** | Announcement window | 公告窗口 | 交易日 `[td0, td0+1]`，公告日 + 次日 |
| **DRIFT** | Post-announcement drift | 漂移窗口 | 交易日 `[td0+2, td0+61]`，公告后第 2–61 个交易日 |
| **SUE** | Standardized Unexpected Earnings | 标准化盈余意外 | `(e − F) / P`。HLT 2009 原文记作 **FE** (Forecast Error) |
| **e** | Actual EPS | 实际每股收益 | I/B/E/S 公告披露值 |
| **F** | Consensus Forecast | 共识预测 | 公告前 60 天内各分析师最新预测的**中位数** |
| **P** | Price | 财季末股价 | 已折算到与 EPS 相同的拆股基准 |
| **SIZE** | Firm Size Decile | 规模十分位 | 年初市值，NYSE breakpoint |
| **BM** | Book-to-Market | 账面市值比 | `BE / ME` |
| **LNANALYST** | Log(1 + #Analysts) | 分析师覆盖对数 | 公告前 365 天内出过预测的不同分析师数 |
| **LAG** | Reporting Lag | 报告滞后 | `anndats − pends`（天），另有 `LAG²` `LAG³` |
| **IO** | Institutional Ownership | 机构持股比例 | 13F 持股合计 / 流通股本 |
| **EVOL** | **Earnings Volatility** | 盈余波动率 | 过去 **16 个财季（4 年）** 内、同比变化 Δ₄EPS 的**样本标准差**（美元/股） |
| **EPERSIST** | **Earnings Persistence** | 盈余持续性 | 16 季窗口内**季度 EPS 水平值**的一阶自相关系数（−1 ~ 1），照 HLT 2009 §III.A |
| **TURN** | Share Turnover | 换手率 | 过去 12 个月的月均换手率 |

### 1.2 收益口径

| 缩写 | English | 定义 | 来源 |
|---|---|---|---|
| **C2C** | Close-to-Close | `close_{t−1} → close_t` | CRSP `dlyret`，**原生字段**，含分红、含退市 |
| **O2O** | Open-to-Open | `open_{t−1} → open_t` | yifei v5 的 `O2O_RET`，**原样取用，不做任何调整** |

> O2O 不是 CRSP 原生字段，但本项目**不自行构造**——直接用 yifei v5 面板已算好的值。
> 逆向核对过它的口径等价于 `(prc_{t−1}/open_{t−1})·(1+ret_t)·open_t/prc_t − 1`
> （即已做拆股/分红调整，而非原始开盘价比值 `open_t/open_{t−1}`），
> 该公式仅用于**验证**，不参与任何计算。

### 1.3 BE（账面权益）的构造分量

$$BE = SEQ + TXDITC - PS$$

| 分量 | Compustat 字段 | English | 中文 | 取值优先级 |
|---|---|---|---|---|
| **SEQ** | `seq` | Stockholders' Equity — Total | 股东权益合计 | ① `seq` |
| | `ceq` + `pstk` | Common Equity + Preferred Stock | 普通股权益 + 优先股 | ② `seq` 缺失时 |
| | `at` − `lt` | Total Assets − Total Liabilities | 总资产 − 总负债 | ③ 前两者都缺时 |
| **TXDITC** | `txditc` | Deferred Taxes and Investment Tax Credit | 递延所得税及投资税收抵免 | 缺失记 **0** |
| **PS** | `pstkrv` | Preferred Stock — Redemption Value | 优先股赎回价值 | ① 首选 |
| | `pstkl` | Preferred Stock — Liquidating Value | 优先股清算价值 | ② |
| | `pstk` | Preferred Stock — Total (Carrying Value) | 优先股账面价值 | ③ |

**ME（Market Equity，市值）** = CRSP 的 `mthcap`，单位**千美元**；
Compustat 的 BE 单位是**百万美元**，所以 `BM = BE × 1000 / ME`。

### 1.4 标识符

| 缩写 | 全称 | 说明 |
|---|---|---|
| **PERMNO** | CRSP Permanent Number | CRSP 股票永久标识，**本项目主键** |
| **PERMCO** | CRSP Permanent Company Number | 公司层面标识（一家公司可有多个 PERMNO） |
| **GVKEY** | Global Company Key | Compustat 公司标识 |
| **CUSIP** | Committee on Uniform Securities Identification Procedures | 证券识别码，8 位。13F 用它报告持仓 |
| **IBES ticker** | I/B/E/S Ticker | I/B/E/S 内部永久标识（≤6 位字母，如 `ABCR`）。**不是**交易所代码 —— 那是 `oftic` |
| **SICCD** | Standard Industrial Classification Code | 行业代码，用于行业固定效应 |
| **MGRNO** | Manager Number | 13F 申报机构编号 |

### 1.5 数据字段

**I/B/E/S**

| 字段 | English | 含义 |
|---|---|---|
| `anndats` | Announcement Date | 盈余公告日 —— **事件锚点** |
| `anntims` | Announcement Time | 公告时间（可判断盘前/盘后） |
| `pends` | Period End Date | 财季结束日 |
| `fpedats` | Forecast Period End Date | 预测目标财季结束日（与 `pends` 匹配才是同一季度的预测） |
| `actdats` | Activation Date | 预测**发布**日（issued） |
| `revdats` | Review Date | 预测**复核**日（reviewed） |
| `analys` | Analyst Code | 分析师代码（去重、计数用） |
| `estimator` | Estimator Code | 所属机构代码 |
| `fpi` | Forecast Period Indicator | 预测期：`1`=本财年, `2`=下财年, **`6`=下 1 季度, `7`=下 2 季度** |
| `value` | Estimate / Actual Value | 预测值（detail）或实际值（actuals） |
| `usfirm` | US Firm Flag | =1 表示来自**美国数据库**，不代表公司注册在美国 |

**CRSP**（CIZ 格式，字段名与旧版 SIZ 不同）

| 字段 | English | 含义 | 旧版对应 |
|---|---|---|---|
| `dlycaldt` / `mthcaldt` | Calendar Date | 交易日 / 月末日 | `date` |
| `dlyret` / `mthret` | Return | 日/月收益（含分红，含退市） | `ret` |
| `dlyprc` / `mthprc` | Price | 收盘价（负值 = 买卖报价中点） | `prc` |
| `dlycap` / `mthcap` | Market Cap | 市值（千美元），CIZ 直接提供 | 需 `prc×shrout` |
| `dlyvol` / `mthvol` | Volume | 成交量（**股**） | `vol` |
| `shrout` | Shares Outstanding | 流通股本（**千股**） | 同 |
| `mthcumfacpr` | Cumulative Factor to Adjust Price | 累积价格调整因子 | `cfacpr` |
| `mthcumfacshr` | Cumulative Factor to Adjust Shares | 累积股数调整因子（本项目未使用，见 §6.3） | `cfacshr` |
| `primaryexch` | Primary Exchange | `N`=NYSE, `A`=AMEX, `Q`=NASDAQ | `exchcd` |
| `sharetype` / `securitytype` / `securitysubtype` | — | 股票类型三件套 | `shrcd` |
| `usincflg` | US Incorporation Flag | 是否美国注册 | — |
| `issuertype` | Issuer Type | `CORP`/`ACOR`/`REIT` | — |
| `secinfostartdt` / `secinfoenddt` | Security Info Start/End Date | 该条信息的有效区间 | — |

**Compustat**

| 字段 | English | 含义 |
|---|---|---|
| `datadate` | Data Date | 财年/财季结束日 |
| `fyearq` / `fqtr` | Fiscal Year / Quarter | 财政年度 / 季度序号 |
| `rdq` | Report Date of Quarterly Earnings | Compustat 记录的公告日（本项目仅备用） |
| `epspxq` | EPS Basic — Excluding Extraordinary Items | 季度基本 EPS（不含非经常项） |
| `ajexq` | Adjustment Factor — Cumulative by Ex-Date | 累积拆股调整因子，`epspxq / ajexq` 才能跨季比较 |
| `cshprq` | Common Shares Used to Calculate EPS | 计算 EPS 所用股数（备用） |
| `indfmt` / `datafmt` / `popsrc` / `consol` | — | Compustat 标准筛选四件套（见 §2.1） |

**连接表**

| 字段 | 含义 |
|---|---|
| `linkdt` / `linkenddt` | GVKEY↔PERMNO 链接有效期（`ccmxpf_lnkhist`） |
| `linktype` | 链接类型，`LU`/`LC` = 可靠链接 |
| `linkprim` | 主链接标记，`P`=主要, `C`=次要但可靠 |
| `sdate` / `edate` | IBES ticker↔PERMNO 链接有效期（`ibcrsphist`） |
| `score` | 匹配质量分，越小越好（本项目筛 ≤2） |

---

## 2. 阶段一：WRDS 下载

`new_WRDS_Data_Download.ipynb` → `data/`（102 个 parquet）

### 2.1 逐表清单

| 文件 | WRDS 库表 | 筛选条件 | 行数 | 时间范围 |
|---|---|---|---|---|
| `link_crsp_compustat` | `crsp.ccmxpf_lnkhist` | `linktype∈{LU,LC}`, `linkprim∈{P,C}` | 33,324 | — |
| `link_ibes_crsp` | `wrdsapps.ibcrsphist` | `score ≤ 2` | 30,080 | 止于 2025-12-31 |
| `crsp_security_info` | `crsp_m_stock.stksecurityinfohist` | 全量 | 193,968 | 1925 ~ 2026-06-30 |
| `crsp_daily_{1996..2026}` | `crsp_m_stock.dsf_v2` | 按年分批 | 60,255,050 | 1996-01-02 ~ 2026-06-30 |
| `crsp_monthly` | `crsp_m_stock.msf_v2` | `mthcaldt ≥ 1994-01-01` | 3,091,767 | 1994-01-31 ~ 2026-06-30 |
| `compustat_annual` | `comp.funda` | `indfmt='INDL'`, `datafmt='STD'`, `popsrc='D'`, `consol='C'`, `datadate ≥ 1993-01-01` | 388,641 | 1993-01-31 ~ 2026-06-30 |
| `compustat_quarterly` | `comp.fundq` | 同上，`datadate ≥ 1992-01-01` | 1,607,253 | 1992-01-31 ~ 2026-06-30 |
| `ibes_actuals` | `ibes.act_epsus` | `measure='EPS'`, `curr_act='USD'`, `pdicity='QTR'`, `usfirm=1`, `pends ≥ 1995-07-01` | 742,513 | pends 1995-07-31 起 |
| `ibes_detail_{1995..2026}` | `ibes.det_epsus` | `measure='EPS'`, `fpi∈{1,2,6,7}`, `usfirm=1`，按 `fpedats` 年份分批 | 16,888,927 | 1995 ~ 2026 |
| `ibes_adjustment` | `ibes.adj` | 全量 | 202,959 | — |
| `tr13f_{1996..2026}` | `tfn.s34` | 按 `fdate` 年份分批 | 113,948,360 | 1996 ~ 2025（2026 为空） |

`indfmt='INDL'` = 工业格式（非金融专用格式），`datafmt='STD'` = 标准化数据，
`popsrc='D'` = 国内数据源，`consol='C'` = 合并报表 —— Compustat 的标准四件套，
不加会取到同一公司的多份重复记录。

### 2.2 为什么每张表的起点不同

样本从 1996 年的公告开始，但各变量的**回溯窗口**长度不一样：

| 表 | 起点 | 原因 |
|---|---|---|
| `comp.funda` | 1993 | formation 1995 的 BE 来自 FY1994；财年跨年需留余量 |
| `crsp msf_v2` | 1994 | formation 1995 需要 1994 年 12 月末的 ME |
| `comp.fundq` | 1992 | EVOL/EPERSIST 要回溯 **16 个季度** |
| `ibes.act_epsus` | pends 1995-07 | 1996 年 1–2 月公告的是 1995Q4 财季 |
| `ibes.det_epsus` | 1995 | 同上，预测按 `fpedats` 年份存放 |
| `crsp dsf_v2` | 1996 | CAR 窗口只往公告日之后看，无需回溯 |

⚠️ **踩过的坑**：最初所有表都从 1996-01-01 下，导致基准组合从 1997-07 才开始，
35,318 个事件（4.8%）算不出 CAR。修正起点后覆盖到 1996-01。

### 2.3 未使用的下载

- `ibes_adjustment`：见 §6.2，I/B/E/S 的 EPS 已统一到最新拆股基准，无需折算。保留备用
- `tr13f` 的 `shrout1`/`shrout2`：13F 自带的股本口径，本项目统一用 CRSP 的 `shrout` 保证一致
- `compustat_quarterly` 的 `saleq`/`ibq`/`atq`/`ltq`/`cshoq`：备用字段

---

## 3. 阶段二：收益面板导出

`export_returns.py` → `export/crsp_daily_ret_c2c_o2o.parquet`（65,834,310 行，1.31 GB）

| 列 | 来源 | 覆盖 |
|---|---|---|
| `RET` | CRSP `dsf_v2.dlyret` | C2C，1993-01-04 ~ 2026-04-30 |
| `O2O_RET` | yifei `O2O_RET_by_permno_with_future_ret_v5.pq` | O2O，1996-01-02 ~ 2026-03-30 |
| `MarketCap` | 同上（= \|PRC\| × SHROUT，千美元） | |
| `PRC` `OPENPRC` `SHROUT` | CRSP dsf | 价格已取绝对值 |

**O2O 原样取自 v5，未做任何处理**。已逐行验证：2015 年 3 月切片 126,451 行，
与 v5 原文件**最大差异 0.000e+00**，缺失位置也完全一致（各 28 行）。

**为什么不自己算**：O2O 不是 CRSP 原生字段，需要从开盘价构造 —— 拆股要调、
当天无成交的开盘价缺失要处理、停牌前后要断链。曾按公式重建过一版，与 v5 比对
**4.0% 的行对不上**（自建缺失率 3.8% vs v5 的 0.87%），该版本已废弃。
v5 是整条 O2O residual / factor 线的基础，用它可保证与项目其他部分口径一致。

**yifei 的文件全程只读，未做任何修改。**

---

## 4. 阶段三：数据准备（8 步）

`数据准备.ipynb`，每步产出存 `build/`，通过 `FORCE` 开关和文件存在性复用缓存。

### Step 1 — 股票池 + 连接表

**输出**：`universe.parquet` (193,968) · `link_ccm.parquet` (33,324) · `link_ibes.parquet` (30,080)

**普通股口径**：CIZ 格式没有旧版的 `shrcd`，需多字段联合判断：

| 条件 | permno 数 |
|---|---|
| 全部 | 41,520 |
| `sharetype='NS'` & `securitytype='EQTY'` & `securitysubtype='COM'` | 30,517 |
| + `usincflg='Y'`（美国注册） | 27,472 |
| + `issuertype ∈ {CORP, ACOR}`（排除 REIT） | **27,016** ← `in_universe` |

最后一档对应旧版 `shrcd ∈ {10,11}`，也是 HLT 2009 的样本口径。

**设计**：不做硬删除，全部存成布尔列 —— `is_common` / `is_us` / `is_operating` /
`in_universe` / `is_nyse`（breakpoint 用）/ `on_major_exch`。想放宽口径（比如把 REIT 纳回来）
改筛选条件即可，无需重建。

**连接表的两处处理**：
1. `linkenddt` / `edate` 为空表示至今有效 → 统一填 `2099-12-31`。留 `NaT` 会让按区间筛选时整行被丢弃
2. 745 个 permno 历史上对应过多个 gvkey，631 个 ticker 对应过多个 permno（并购、重组、代码回收）
   → 所有 merge **必须按日期区间**，不能简单 `merge(on='permno')`

### Step 2 — BE / ME / BM → 25 组 → 基准组合日收益

**输出**：`port25_membership.parquet` (135,230，formation 年 1995–2026) ·
`port25_bench_returns.parquet` (191,825 = 7,673 交易日 × 25 组)

**Formation 规则**（HLT 2009 Sec III / Fama-French 标准）：

| 用途 | 时点 | 数据来源 |
|---|---|---|
| Size 排序的 ME | formation 当年 **6 月末**市值 | `msf_v2.mthcap` |
| B/M 分子 BE | formation **前一年**财年末 | `funda`（见 §1.3） |
| B/M 分母 ME | formation **前一年 12 月末**市值 | `msf_v2.mthcap` |
| Breakpoint universe | 只用 **NYSE** 股票算 20/40/60/80 分位 | `primaryexch='N'` |
| 生效期 | 当年 **7 月 至 次年 6 月**，成员不变 | — |

组合编号 `port25 = (size_q − 1) × 5 + bm_q`，取值 1–25。
每日组合收益取**等权平均**，C2C 与 O2O **各算一套**，绝不混用。

> **与 HLT 2009 的差异**：论文用的是 Kenneth French 网站现成的 25 个 size-B/M 组合日收益
> （原文 §II："The daily returns of the 25 size-B/M portfolios are from Kenneth French's web site"），
> 并未说明加权方式；French 的标准 25 组是**市值加权**。本项目自建组合有两个必须的理由：
> ① O2O 口径 French 没有提供，必须自己算，两个口径要同源才可比；
> ② 样本期到 2026 年，French 的数据发布有滞后。自建用 NYSE 断点 + 等权。

**验证**（2000–2019 年化收益，C2C）：

```
bm_q      1      2      3      4      5
size_q
1     0.125  0.162  0.170  0.192  0.243   ← 小市值组，value 效应清晰
5     0.094  0.111  0.115  0.109  0.120   ← 大市值组，整体更低
```

size 效应（自上而下递减）与 value 效应（自左向右递增）都出现，量级正常。

C2C 与 O2O 组合的 **21 日累计相关 0.958**；同日相关仅 0.23 是数学必然
（`C2C_t` = 隔夜_t + 日内_t，`O2O_t` = 日内_{t−1} + 隔夜_t，只共享隔夜段）。

### Step 3 — 事件表

**输出**：`events.parquet` — **517,955 条**，1995-04-01 ~ 2026-05-14，14,214 个 permno

| 漏斗 | 剩余 |
|---|---|
| I/B/E/S actuals 原始 | 742,513 |
| 贴上 PERMNO（按日期区间） | 613,272（82.6%） |
| 限定最终宇宙 | 517,966 |
| 去重后 | **517,955** |

**流失去向**（已逐项验证，不是连接失败）：

| 类别 | 事件数 |
|---|---|
| 非普通股（ADR / 优先股 / 基金） | 50,087 |
| 普通股但非美国注册 | 39,979 |
| 普通股 + 美国注册但是 REIT | 17,706 |
| ticker 不在连接表（**外国公司，本就不在 CRSP**） | 其余 |

用 CUSIP（比 ticker 可靠，ticker 会被回收复用）回捞验证：122,527 个未匹配事件里
只有 6,166 个能定位到宇宙内 PERMNO，其中 5,971 个在 2026 年 ——
**1995–2025 三十年只多捞回约 195 个**。所以不采用 ticker 映射（只会引入错配）。

**三类重复的处理**：

| 类型 | 规模 | 处理 |
|---|---|---|
| A. 一个事件匹配多个 PERMNO | — | 宇宙筛选后**自动清零**（多出的都是 ADR/优先股） |
| B. 同 permno-财季多次公告（重述） | 22 行 | 保留**最早**一次 —— PEAD 关心信息首次到达 |
| C. 同 permno 同日多个财季（补报） | 13,118 行 | **全部保留**，见 §6.7 |

**交易日映射**：`td0` = 公告日当天或之后的第一个交易日（1.6% 的公告落在周末/假日需顺延），
`td0_idx` 是它在交易日历中的下标，Step 4 直接用下标取窗口。

**LAG 质量**：中位数 33 天，5%–95% 分位 16–73 天；异常（<0 或 >180 天）仅 1.15%，
由 `flag_lag_bad` 标记。

### Step 4 — CAR

**输出**：`car.parquet` — 517,955 行

$$CAR^{ANN}=\prod_{k=d}^{d+1}(1+R_{i,k})-\prod_{k=d}^{d+1}(1+R_{p,k}),\qquad
CAR^{DRIFT}=\prod_{k=d+2}^{d+61}(1+R_{i,k})-\prod_{k=d+2}^{d+61}(1+R_{p,k})$$

注意是 **buy-and-hold（连乘后相减）**，不是逐日 AR 加总。

**实现思路**：不逐事件循环。把日收益转成每只股票的**累积对数收益**，
窗口收益 = 两端相减取指数：

```
CAR_ANN   = exp(cum[td0+1]  − cum[td0−1]) − 1
CAR_DRIFT = exp(cum[td0+61] − cum[td0+1])  − 1
```

按交易日**流式推进**，内存 O(股票数) 与数据量无关（直接读 6000 万行面板会 OOM）。全程约 50 秒。

**四个实现细节**：

1. `dlyret = −1`（退市全损）→ `log1p` 得 `−inf`，会污染该股票之后整条累积序列 → 下限截到 `−0.999999`。
   实测只对 C2C 的 **3 行**生效；O2O 最小值 −0.9957，从未触发（开盘价不会归零），对它是空操作
2. 窗口内一天交易记录都没有 → CAR 记 `NaN`，**不是 0**
3. 窗口跨过样本末端（2026 年的事件）→ 用截至末端的累积值，`n_days_*` 会小于满窗，由 `flag_short_window` 标记
4. 个股与基准**必须同口径**：C2C 配 C2C 基准，O2O 配 O2O 基准

| | car_ann_c2c | car_drift_c2c | car_ann_o2o | car_drift_o2o |
|---|---|---|---|---|
| 有效样本 | 454,218 | 454,207 | 451,493 | 448,864 |
| 均值 | −0.0003 | −0.0160 | 0.0020 | −0.0235 |
| 标准差 | 0.099 | 0.305 | 0.089 | 0.305 |

C2C 与 O2O 的相关：ANN **0.73**，DRIFT **0.96**（DRIFT 窗口长，采样时点差异被摊薄）。
窗口完整性：ANN 满窗 98.9%，DRIFT 满窗 96.7%。
CAR 缺失 11.3%，来源是 `port25` 贴不上（覆盖率 88.8%）—— 这些公司当年没有 BE 或 6 月末市值。

### Step 5 — SUE

**输出**：`sue.parquet` — 517,955 行，其中 **344,026 条有 SUE**（66.4%）

$$SUE_{i,d}=\frac{e_{i,d}-F_{i,d}}{P_{i,d}}$$

**F 的构造**（HLT 2009 Sec II 原文规则）：
1. 窗口终点锚定公告日 `anndats`，起点往前推 **60 个日历天**
2. 只纳入 1–2 季度前瞻预测（`fpi ∈ {6,7}`），且 `fpedats == pends`（同一财季）
3. 窗口内 issued（`actdats`）**或** reviewed（`revdats`）的预测都算
4. 同一分析师在窗口内多次预测 → 只取**最新**一条
5. 对所有符合条件的分析师取**中位数**

**P 的构造**：`|mthprc| / mthcumfacpr` —— 财季末股价折算到与 EPS 相同的拆股基准。
**为什么必须折算**：I/B/E/S 的 EPS 已统一到最新基准（AAPL 2012Q2 记为 0.4393 而非披露时的 \$12.30，
差 28 倍 = 2014 年 7:1 × 2020 年 4:1），而 CRSP 的 `mthprc` 是当年历史价格。
直接相除会让拆过股的公司 SUE 被系统性缩小，且每家缩小倍数不同，污染截面排序。

**清洗规则**（HLT 2009 Sec II 末尾）：

| 规则 | 剔除数 |
|---|---|
| 拆股调整**前**股价 < \$1 | 18,454 |
| \|e\| 或 \|F\| 大于股价 | 7,895 |
| 合计（去重叠） | **23,801** → `flag_sue_dropped` |

**分十分位**：按公告日所在**日历季度**独立分十份（`sue_dec` 1–10）。
分位点随时间自适应，不会因为某年整体盈余意外偏大而失衡。

**覆盖**：68.3% 的事件有共识预测（中位数 4 位分析师，最多 44 位），98.5% 能取到财季末股价。

### Step 6 — 10 个控制变量

| 输出文件 | 行数 | 粒度 | 构造 |
|---|---|---|---|
| `ctrl_size.parquet` | 260,320 | firm×ffyear | formation 年 6 月末市值 → NYSE 十分位 |
| `ctrl_bm.parquet` | 135,230 | firm×formation年 | 与 25 组同一套 BE/ME → NYSE 十分位（独立于 size） |
| `ctrl_turn.parquet` | 2,871,577 | firm×month | 过去 12 个月 `mthvol/(shrout×1000)` 的均值（min 6 个月） |
| `ctrl_io.parquet` | 889,973 | firm×quarter | `Σshares/(shrout×1000)`，同 (mgrno,cusip,rdate) 多次申报取最新 `fdate` |
| `ctrl_evol_epersist.parquet` | 698,171 | firm×quarter | 见下 |
| `ctrl_lnanalyst.parquet` | 443,648 | firm×event | 公告前 365 天内不同 `analys` 计数 |
| `ctrl_att.parquet` | 517,955 | firm×event | 同日公告数 → 季度内十分位 NRANK → ATT = 11 − NRANK |

**EVOL（Earnings Volatility，盈余波动率）与 EPERSIST（Earnings Persistence，盈余持续性）的构造**

两者都取**过去 16 个财季（4 年）**的窗口、都要求窗口内至少 4 个非缺失观测。
序列各用各的，照 HLT 2009 §III.A 原文：

> "Earnings Persistence is the first-order autocorrelation coefficient of **quarterly earnings per
> share** during the past 4 years (split-adjusted; minimum four observations required) ...
> Earnings Volatility is the standard deviation during the preceding 4 years of the **deviations of
> quarterly earnings from 1-year-ago earnings** (split-adjusted; minimum four observations required)."

**① 拆股调整**（两者共用）：`eps_adj = epspxq / ajexq`，要求 `ajexq ≥ 0.01`（见 §6.3）。
`epspxq` 是当期口径的每股金额，拆股后数字凭空变小，除掉累积调整因子才能跨季比较。

**② EVOL —— 季节差分序列的标准差**

$$\Delta_4 EPS_t = eps\_adj_t - eps\_adj_{t-4},\qquad
EVOL_t = \mathrm{sd}\big(\Delta_4 EPS_{t-15..t}\big)\ (\text{ddof}=1)$$

减的是**去年同一财季**，不是上一季 —— 很多公司盈余有强季节性（零售 Q4 最高），
用相邻季度差分测到的主要是季节规律。单位：美元/股，取值 ≥ 0。

**③ EPERSIST —— EPS 水平值的一阶自相关**

$$EPERSIST_t = \mathrm{corr}\big(eps\_adj_s,\ eps\_adj_{s-1}\big),\quad s \in [t-15,\ t]$$

配对的是 **t 与 t−1（相邻财季）的 EPS 水平值**。取值 −1 ~ 1，无量纲。

两处都用 `fyearq×4 + fqtr` 构造的绝对季度序号自连接来对齐（`+4` 配 EVOL、`+1` 配 EPERSIST）：
Compustat 会缺季度、也会变更财年，按绝对序号配对才能保证取到的确实是目标财季。
自相关用滚动和向量化实现（`rolling.apply` 慢 10 倍），
公式 $\rho=\frac{\overline{xy}-\bar x\bar y}{\sqrt{(\overline{x^2}-\bar x^2)(\overline{y^2}-\bar y^2)}}$，六个滚动和拼出。

**实测分布**（`build/ctrl_evol_epersist.parquet`，692,836 条有 EPERSIST）：

| 分位 | P10 | P25 | 中位数 | P75 | P90 | 均值 |
|---|---|---|---|---|---|---|
| EPERSIST | −0.195 | −0.038 | **0.184** | 0.482 | 0.721 | 0.221 |

回归样本内均值 **0.254**、中位数 0.220。论文 Table I Panel B 报告的各组均值是 **0.342–0.431**，
比我们高。两个可解释的来源：① 论文样本 1995–2004 且只含 I/B/E/S 覆盖 + 控制变量齐全的公司，
偏大盘、盈余更平滑；② 我们的样本期长一倍多、覆盖到更多小盘股。方向与量级一致。

**两个变量的分工**：EVOL 用 Δ₄ 消掉季节性，量的是"盈余波动多大"；
EPERSIST 用水平值，量的是"盈余水平有多黏"。季度 EPS 本身有强季节性
（苹果 Q1 假日季远高于 Q3），所以水平值的相邻季相关度普遍偏低 ——
实测苹果 2015–2019 的 EPERSIST 中位数 0.127、微软 0.017。

**基准是公司自己**：EVOL 衡量该公司的 Δ₄ 围绕**它自己这 16 季的均值**波动多大，
截面比较在回归里通过系数完成。窗口每季度前滚一格。

**单位陷阱**（都已实测验证）：
- `mthvol` 是**股数**，`shrout` 是**千股** → TURN = `mthvol/(shrout×1000)`。AAPL 2015-06 = 0.154 ✓
- 13F `shares` 是**股数** → IO = `shares/(shrout×1000)`。AAPL 2015Q2 = 0.586 ✓（与实际机构持股吻合）
- TURN **不需要**拆股调整：分子分母同月口径，比值本身自洽

### Step 7 — 合并成大表

**输出**：`pead_panel.parquet` — **517,955 行 × 75 列**

`IO` / `EVOL` / `EPERSIST` 是季度数据，用 `merge_asof(direction='backward')` 取公告前最近一期，
并记录陈旧程度：`io_stale_days > 200 天`、`eps_stale_days > 400 天` 则置为缺失 ——
否则会把两年前的机构持股当成当期值。

**控制变量覆盖率**：

| 变量 | 覆盖 | 中位数 |
|---|---|---|
| `size_dec` | 96.6% | 2 |
| `bm_dec` | 88.8% | 5 |
| `lnanalyst` | 100% | 1.79 |
| `lag` | 100% | 33 天 |
| `io` | 96.9% | 0.525 |
| `evol` | 93.9% | 0.282 |
| `epersist` | 93.9% | 0.203 |
| `turn` | 96.9% | 0.115 |

**关键变量全齐（可进 baseline 回归）：303,466 条（58.6%）**
≈ 有 SUE (66.4%) × 有 CAR (88.7%) × 有 BM (88.8%)。

### Step 8 — 数据字典

**输出**：`build/data_dictionary.csv` —— 75 列的列名 / dtype / 覆盖率 / 中位数 / 说明。
带 `⚠️ 未登记` 检查：新增列若忘记写说明会被标出。

---

## 5. 大表列字典（75 列）

> 机读版见 `build/data_dictionary.csv`。★ = 回归主变量，◆ = 控制变量。

### 键与事件信息

| 列 | 说明 |
|---|---|
| `eid` | 事件唯一编号（= `events.parquet` 行号，可回溯各中间表） |
| `permno` `ticker` `cusip` `cname` | 标识符（`ticker` 是 IBES ticker） |
| `anndats` `anntims` | 公告日、公告时间 |
| `td0` `td0_idx` | 事件日（顺延后的交易日）及其日历下标 |
| `pends` | 财季结束日 |
| `actual_eps` | 实际 EPS |
| `siccd` `primaryexch` | SIC 行业码、主交易所 |
| `year` `month` `dow` `qtr` | 日历固定效应键（年/月/星期几/季度） |
| `ffyear` `ym_ann` | formation 年、公告年月（内部匹配键） |
| `rdate` `datadate` | IO 所用的 13F 报告期、EVOL 所用的财季末 |

### 被解释变量

| 列 | 说明 |
|---|---|
| ★ `car_ann_c2c` `car_drift_c2c` | **C2C 口径的 CAR**（建议主口径） |
| ★ `car_ann_o2o` `car_drift_o2o` | **O2O 口径的 CAR** |
| `stk_ann_*` `stk_drift_*` | 个股 buy-and-hold 收益（未减基准），4 列 |
| `bench_ann_*` `bench_drift_*` | 基准组合 buy-and-hold 收益，4 列 |
| `port25` | 所属 size×B/M 组合编号 1–25 |
| `n_days_ann_*` `n_days_drift_*` | 窗口内实际有收益记录的天数（满窗 = 2 / 60），4 列 |

### 核心自变量

| 列 | 说明 |
|---|---|
| ★ `sue` | 连续值 `(e−F)/P_adj` |
| ★ `sue_dec` | 十分位（按日历季度分组，1 = 最负意外，10 = 最正） |
| `consensus_f` | 共识预测 F |
| `price_adj` `prc_unadj` | 折算后 / 未折算的财季末股价（后者用于 <\$1 清洗规则） |
| `n_analyst_sue` | 算这次共识用到的分析师数（**诊断用，非控制变量**） |
| `fcst_last_dt` | 共识中最新一条预测的日期 |

### 控制变量

| 列 | 说明 |
|---|---|
| ◆ `size_dec` / `me_jun` | 规模十分位 / formation 年 6 月末市值原值（千美元） |
| ◆ `bm_dec` / `bm_raw` / `be` | B/M 十分位 / 连续值 / 账面权益（百万美元） |
| ◆ `lnanalyst` / `n_analyst_cover` | 覆盖分析师数的对数 / 原始计数 |
| ◆ `lag` `lag2` `lag3` | 报告滞后及高阶项 |
| ◆ `io` | 机构持股比例 |
| ◆ `evol` `epersist` | 盈余波动率 / 持续性 |
| ◆ `turn` | 换手率 |

### 标记与诊断

| 列 | 含义 | 规模 |
|---|---|---|
| `flag_lag_bad` | LAG < 0 或 > 180 天 | 1.15% |
| `flag_same_day_multi` | 同 permno 同日多个财季（补报） | 13,118 |
| `is_latest_pends_on_day` | 是否该日的当期财季 | False 8,332 |
| `flag_link_extended` | IBES 链接在表末期仍有效（**无风险**） | 168,515 |
| `flag_beyond_link_end` | 公告日晚于 2025-12-31，只因延长链接才存在 | 5,846 |
| `flag_short_window` | ANN < 2 天 或 DRIFT < 60 天 | 3.3% |
| `flag_sue_dropped` | 被 SUE 清洗规则剔除 | 23,801 |
| `ann_on_nontrading` | 公告落在非交易日（已顺延） | 1.6% |
| `flag_td0_gap` / `flag_pre_calendar` | 公告日早于交易日历起点，CAR 已置空 | 5,628 |
| `td0_gap_days` | `td0 − anndats` 的天数 | — |
| `io_stale_days` `eps_stale_days` | 控制变量数据的陈旧天数 | — |

**设计原则**：原始数据全留，筛选交给回归阶段。

---

## 6. 已知问题与处理决策

### 6.1 yifei v5 面板的 `ret` 列不是 C2C

`O2O_RET_by_permno_with_future_ret_v5.pq` 里名为 `ret` 的列，实测是**前移一天的 O2O**
（`open_t → open_{t+1}`），不是 close-to-close。2019 年 1 月 12.1 万行比对：
与老文件的 `O2O_RET+1` 97.8% 完全相同，与真 C2C **98.7% 对不上**。
**C2C 必须从 CRSP `dlyret` 取。**

### 6.2 I/B/E/S 的 EPS 已是最新拆股基准

AAPL 2012Q2 在 `ibes_actuals` 里是 `0.4393`，当年披露的是 `$12.30`，差 **28 倍**。
所以 `(e − F)` 内部基准一致，**不需要** `ibes.adj` 折算；
但分母 P 必须用 `mthcumfacpr` 折到同一基准（见 §4 Step 5）。
`ibes_adjustment.parquet` 保留备用（若将来改用未调整版 `detu_epsus`/`actu_epsus`）。

### 6.3 三个数值陷阱

| 问题 | 影响 | 处理 |
|---|---|---|
| `dlyret = −1`（退市全损，C2C 全样本仅 3 行） | `log1p` 得 `−inf`，污染该股票之后整条累积序列 | 下限截到 `−0.999999`；O2O 从未触发 |
| `ajexq < 0.01`（4.11% 的行，甚至为 0） | `epspxq/ajexq` 炸出 −34 亿 | 加下限，这些行剔除 |
| TURN 的拆股调整 | 无需调整 | `mthvol` 与 `shrout` 同月口径，比值自洽 |

### 6.4 `sue_dec` 的 nullable dtype 陷阱（已修复）

`sue` 继承自 I/B/E/S 的 nullable `Float64`，而 `pd.qcut` 对 `pd.NA` 的处理与 `np.nan` 不同：
**会把缺失值也打进极端分位**。修复前有 9,029 个 `sue` 缺失的事件被分到 D1/D10，
十分位样本量畸形（31,094 vs 35,018）。
修复：先 `pd.to_numeric(...).astype("float64")`，分位算完再把缺失位置显式置回 `NaN`，并加 `assert`。
修复后各分位均衡（约 34,400），`car_drift_c2c` 的 D10−D1 从 +4.22% 变为 **+4.23%**（t 从 13.0 升到 14.6）。
`size_dec` / `bm_dec` 已检查，无此问题。

### 6.5 等权组合再平衡偏差

`car_drift_c2c` 均值 −1.64% 而非 0。原因：基准是"每日等权平均后连乘"，
相当于每天把仓位拉回等权，这个再平衡收益（尤其在买卖价差跳动大的小盘股上）系统性高于个股买入持有。
**对 PEAD 检验无影响**（检验的是截面差异，常数偏移被截距吸收），
但报告"某组绝对 CAR 显著为负"时要考虑这个偏移。
O2O 版在最小市值五分位偏差更大（组合年化差 +2.7 ~ +5.8pp）。

### 6.6 IBES 连接表止于 2025-12-31

`wrdsapps.ibcrsphist` 定期更新，4,386 条链接的 `edate` 停在 2025-12-31。
`EXTEND_IBES_LINK=True` 把它们延长到 2099-12-31，恢复了 2026 年的 5,846 个事件。
保守做法：回归时加 `~flag_beyond_link_end`（等于样本截到 2025-12-31）。

### 6.7 补报（同日多财季）

13,118 行、4,786 个 permno-日，最多的一家一次补 13 个季度。
**全部保留** —— 公司确实披露了，滞后由 `lag`/`lag2`/`lag3` 吸收。
但这些行**共享同一个 CAR**（同 permno 同日 → 同一收益窗口），回归时三选一：

1. `is_latest_pends_on_day == True`（丢 8,332 行，1.6%）— 推荐
2. 全留 + 按 `permno × anndats` 聚类标准误
3. 剔除 `flag_same_day_multi`（丢 13,118 行，2.5%）

补报行的 `actual_eps` 缺失率 50.2%（当期仅 10.2%），实际影响比行数看着小。

### 6.8 审计发现的两个问题（已修复）

`audit_panel.py` 做独立审计（恒等式 / 前视偏差 / 取值域 / 重复，共 35 项）时发现：

**① 共识预测"取最新"的排序日期用错**：原先按 `max(actdats, revdats)` 给同一分析师的多条
预测排序，而 `revdats`（复核日）常远在公告之后 —— **23.1% 的事件**里这个日期落在公告日之后，
中位数晚 167 天。虽然预测**值**本身是公告前的（IBES 改值会新建记录），但用公告后的复核日
决定"谁最新"既有轻微前视、也不是 HLT 说的最新一条。
改为取 `actdats` / `revdats` 中**落在公告前**的较晚者。
影响：1.20% 的事件共识预测发生变化，SUE 中位变化为 0，`car_drift_c2c` 的 D10−D1 从 +4.26% → +4.23%。

**② `epersist` 出现 `inf`**：窗口内 16 个季节差分完全相同时方差为 0，相关系数除出无穷（1 行）。
改为方差 ≤ 0 时记 `NaN`，并强制 \|corr\| ≤ 1。

### 6.9 公告日早于交易日历起点（已修复）

`td0` 定义为"公告日当天或之后的第一个交易日"，用 `searchsorted` 实现。
但 I/B/E/S 的事件从 1995-04 开始，而交易日历（CRSP 日频）从 1996-01-02 开始 ——
**1995 年的 5,628 个事件的 `td0` 全被落到 1996-01-02**，CAR 窗口与公告相隔最多 8 个月，
测的完全是无关时段的收益。修复前其中 5,051 个有 CAR 值、2,899 个进了回归样本。

修复：`flag_pre_calendar = anndats < 交易日历起点`，这些事件的 CAR / stk / bench 列全部置空。
样本起点因此明确为 **1996-01-02**，与新闻语料起点一致。

### 6.11 内存

login node 内存吃紧，三处必须流式处理：CAR 的日收益累积、13F 的 1.14 亿行聚合、
交易日历构建（逐文件先 `drop_duplicates` 再合并）。
notebook 各步之间有 `del` + `gc.collect()`，删掉会导致 kernel 被 OOM 杀掉。

---

## 7. 验证：PEAD 复制结果

各 SUE 十分位的平均 CAR（%），样本 343,609（有 SUE 十分位、同日多季报只留最新一季）：

| sue_dec | car_ann_c2c | car_drift_c2c | car_ann_o2o | car_drift_o2o | n |
|---|---|---|---|---|---|
| 1 | −4.167 | −3.535 | −3.595 | −5.032 | 34,364 |
| 5 | −0.161 | −0.891 | −0.115 | −1.176 | 34,369 |
| 10 | 3.925 | 0.715 | 3.705 | 0.390 | 34,394 |

| D10 − D1 | 价差 | t | 十分位单调 |
|---|---|---|---|
| `car_ann_c2c` | **+8.09%** | 82.6 | ✓ |
| `car_drift_c2c` | **+4.25%** | 14.6 | ✓ |
| `car_ann_o2o` | +7.30% | 86.5 | ✓ |
| `car_drift_o2o` | +5.42% | 18.3 | ✓ |

四个 CAR 变量随 SUE 十分位递增。**DRIFT 窗口的正向价差就是 PEAD。**

---

## 8. 复现与核对

**重跑流水线**：

```bash
cd /project/dachxiu/zan/PEAD
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=nlp3 \
    --ExecutePreprocessor.timeout=10800 数据准备.ipynb
```

耗时约 15 分钟（缓存命中时更快）。`FORCE = True` 忽略缓存全量重算
（`port25_bench_returns` 约 5 分钟，`ctrl_io` 约 5 分钟）。
各步产物独立存盘，改某个变量的定义只需重跑对应 cell + Step 7。

**核对文档**：

```bash
python verify_readme.py
```

逐条比对本文档引用的 **84 个数字**与实际产物（原始数据行数、各步产出、CAR/SUE 分布、控制变量覆盖率、
PEAD 十分位与 t 值、baseline 回归的 β₁/t/N、数据字典完整性），输出不一致项。
数据或回归重跑后跑一次即可确认文档没过期。

---

## 9. 进度

### 已完成：PEAD baseline（`回归准备.ipynb`）

| 部分 | 内容 | 产出 |
|---|---|---|
| A | 组合排序（BT 1989 方法）：十分位表、事件时间路径（公告后 / 公告前后分段两张图）、对冲组合、上季断点（FOS）补充面板 + 对照图 | `build/car_path.parquet`, `build/pead_paths.png`, `build/pead_paths_full.png`, `build/pead_paths_breakpoints.png` |
| B | 截面回归 (0a) `SUE + controls`，两窗口 × 两口径 | `build/baseline_results.csv` |
| C | 截面回归 (0b) 加 `X × SUE` 交互项 | 同上 |
| B.5 | 交叉验证：`pyfixest` 与独立手写实现（numpy 做 FWL 去均值 + 聚类方差）逐系数比对 | — |
| B.6 | 论文刻度复制：FE 用 1–10 整数、CAR 用百分点，与原文 1.028 / 0.979 直接对照 | — |
| D | 稳健性：连续 SUE / 2000 年后 / 剔除微盘股 / 只保留满窗事件 | — |

**核心结果**：对冲组合（D10−D1，事件时间路径口径）在公告前 60 天已累计 +6.96%（C2C），
公告窗口 [0,1] 再得 +8.03%，公告后的**纯漂移 [2,61] 为 +5.17%（C2C）/ +6.20%（O2O）**，
对照 BT 1989 的 6.31%。十分位表（窗口端点口径）给出的纯漂移是 +4.25% / +5.42%，
两者的差异来自样本：路径要求 [−60,+61] 整段可得，比十分位表少约 12% 的事件。
回归中 DRIFT 的 β₁ 在控制行业/日历固定效应与 10 个控制变量后仍显著为正。

### 已完成：Channel 回归（`回归结果补充.ipynb`）

两个 moderator 各自先跑不含 LLM signal 的基本回归，设定与 baseline 完全相同，
并按 HLT 2009 eq. (4) 加入控制变量 × SUE 的交互。

| 节 | Channel | Moderator | 结果 |
|---|---|---|---|
| §2 | **ADOPT** | `ADOPT` = 1{公告日 ≥ 2022-11-30}（ChatGPT 发布） | `SUE × ADOPT` 在 ANN 显著为正（+0.0096\*\*\* C2C / +0.0092\*\*\* O2O），DRIFT 为负但不显著 |
| §3 | **ATT** | `ATT` = 11 − NRANK，NRANK 为同日公告数的季度内十分位 | `SUE × ATT` 在 ANN 两口径均 1% 显著为正，DRIFT 为负、O2O 边际显著 |

**ADOPT 的分段斜率**（Table A3，直接估两段而非只看交互项）：公告窗口从 0.0803 升到 0.0899
（+12%，C2C），漂移窗口 0.0237 → 0.0232，**post 期仍显著为正**。
所以结论是"没有证据显示漂移变小"，而非"漂移消失"。ADOPT 是纯日历断点，
同期发生的其他变化都与它共线，单独用无法把功劳归给 GenAI。

**ATT 与原文的对照**（`回归结果补充.ipynb` §3.3，与原文同刻度同设定）：
`FE × NRANK` 在 CAR[0,1] 是 −0.0205\*\*\*（原文 −0.015），公告窗口完全复制；
CAR[2,61] 是 +0.0089（原文 +0.049），方向一致、量级约五分之一 ——
PEAD 本身在 2000 年后大幅衰减，作用在漂移上的调节效应随之缩小。

**尚未完成**
- **LLM signal 已建好**（`build_llm_signal.py`，处理思路与列字典见 `回归结果补充.ipynb` §4）：
  新闻级预测 → firm-day 取平均 → 事件窗口求和。已产出 $[d,\ d+1]$ 与 $[d-1,\ d+1]$ 两个窗口，
  加窗口只需改脚本顶部一行、重跑第二层（秒级）
- **Moderator `OUT`**（AI 停机）：尚未构造
- **核心检验** $\beta_5$（$M \times LLM$）：待 LLM signal 接入后
