"""LLM signal：新闻级预测 → firm-day → 事件窗口。

数据源（**只读**，不写入）：
  /project/dachxiu/yifei/news/experiment/US/ARTICLE/RidgeProximal/
  QUESTION_CHOICE_pred_1d_is4cv3_cossim_0.8_O2O_RET_future_1d_O2O_RET_not_rank_normed
  _rolling_move_trading_days_expectation_only/
  逐年一个 pred_YYYY.pkl，2004-01 ~ 2026-03。新闻级：索引 (timestamp, PERMNO)，
  约三成 firm-day 有多条新闻。`pred` 是该条新闻对**其所在交易日 open → 次日 open**
  收益的预测（`labels` = `future_1d_O2O_RET` 是实现值，firm-day 级）。
  `DATE` 已是可交易日（盘后新闻推到次日），实测 100% 落在 CRSP 交易日历内。

两层产出：
  build/llm_daily.parquet   公司 × 交易日：n_news、pred_mean
  build/llm_signal.parquet  事件级：各窗口的 cum / mean / ndays / n / rank

窗口后缀直接编码偏移量，一看即知覆盖哪几天：
  [d,   d+1] → 0_1        [d-1, d+1] → m1_1
  [d-5, d-1] → m5_m1      [d+2, d+61] → 2_61
窗口运算用交易日历下标做 `cum[t+b] - cum[t+a-1]`，任意长度都是 O(1)：
改 WINDOWS 后只需重跑第二层（秒级），第一层不用重建。
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

SRC = ("/project/dachxiu/yifei/news/experiment/US/ARTICLE/RidgeProximal/"
       "QUESTION_CHOICE_pred_1d_is4cv3_cossim_0.8_O2O_RET_future_1d_O2O_RET"
       "_not_rank_normed_rolling_move_trading_days_expectation_only")
DATA, BUILD = "data", "build"
DAILY = f"{BUILD}/llm_daily.parquet"
OUT = f"{BUILD}/llm_signal.parquet"

WINDOWS = [(0, 1),        # [d, d+1]，与 CAR[0,1] 同窗口
           (-1, 1)]       # [d-1, d+1]，多含公告前一天
# 加窗口就加一个 tuple，例如 (-5, -1) 公告前一周、(2, 61) 与 CAR[2,61] 同窗口

MIN_Q_RANK = 100          # 一个季度内至少这么多事件才排分位


def wname(a, b):
    """(0,1) -> '0_1'，(-1,1) -> 'm1_1'，(-5,-1) -> 'm5_m1'"""
    f = lambda v: f"m{abs(v)}" if v < 0 else str(v)
    return f"{f(a)}_{f(b)}"


# ---------- 交易日历（与大表 td0_idx、build_car_path.py 同一套）----------
cal = pd.concat([pd.read_parquet(f, columns=["dlycaldt"])["dlycaldt"].drop_duplicates()
                 for f in sorted(glob.glob(f"{DATA}/crsp_daily_*.parquet"))])
cal = pd.to_datetime(cal).drop_duplicates().sort_values().reset_index(drop=True)
cal_map = pd.Series(np.arange(len(cal), dtype="int32"), index=cal.values)
NDAY = len(cal)
print(f"交易日历 {NDAY:,} 天: {cal.iloc[0].date()} ~ {cal.iloc[-1].date()}", flush=True)

# ---------- 第一层：新闻级 → firm-day ----------
if os.path.exists(DAILY):
    daily = pd.read_parquet(DAILY)
    print(f"第一层已存在，直接读入: {len(daily):,} 个 firm-day", flush=True)
else:
    parts = []
    for f in sorted(glob.glob(f"{SRC}/pred_*.pkl")):
        if f.endswith("_r2s.pkl"):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        d = d.reset_index()[["PERMNO", "DATE", "pred"]]
        d["permno"] = pd.to_numeric(d["PERMNO"], errors="coerce").astype("int32")
        d["date"] = pd.to_datetime(d["DATE"])
        # 同一天的多条新闻预测的是**同一个**单日收益，取平均
        g = (d.groupby(["permno", "date"], sort=False)
              .agg(n_news=("pred", "size"), pred_mean=("pred", "mean")).reset_index())
        parts.append(g)
        print(f"  {os.path.basename(f)}: {len(d):,} 条 → {len(g):,} 个 firm-day", flush=True)
        del d, g

    daily = pd.concat(parts, ignore_index=True)
    del parts
    # 同一 firm-day 若被两个年度文件同时覆盖，按条数加权合并
    dup = daily.duplicated(["permno", "date"], keep=False)
    if dup.any():
        print(f"  跨年度文件重复的 firm-day: {int(dup.sum()):,}，按条数加权合并", flush=True)
        daily["_w"] = daily["pred_mean"] * daily["n_news"]
        daily = (daily.groupby(["permno", "date"], as_index=False)
                      .agg(n_news=("n_news", "sum"), _w=("_w", "sum")))
        daily["pred_mean"] = daily["_w"] / daily["n_news"]
        daily = daily.drop(columns="_w")

    daily["td_idx"] = daily["date"].map(cal_map)
    miss = daily["td_idx"].isna()
    if miss.any():
        print(f"  DATE 不在交易日历内、被剔除: {int(miss.sum()):,}", flush=True)
        daily = daily[~miss]
    daily["td_idx"] = daily["td_idx"].astype("int32")
    daily["n_news"] = daily["n_news"].astype("int16")
    daily = daily.sort_values(["permno", "td_idx"]).reset_index(drop=True)
    daily[["permno", "date", "td_idx", "n_news", "pred_mean"]].to_parquet(DAILY, index=False)
    print(f"\n→ {DAILY}  {len(daily):,} 个 firm-day | {daily['permno'].nunique():,} 家公司 | "
          f"{daily['date'].min().date()} ~ {daily['date'].max().date()}", flush=True)

# ---------- 第二层：firm-day → 事件窗口 ----------
ev = pd.read_parquet(f"{BUILD}/pead_panel.parquet",
                     columns=["eid", "permno", "anndats", "td0_idx"]).dropna(subset=["td0_idx"])
ev["permno"] = ev["permno"].astype("int32")
ev["td0_idx"] = ev["td0_idx"].astype("int32")
print(f"\n事件 {len(ev):,}", flush=True)

# 每家公司在交易日轴上的累积量：cum[j] = 前 j 天的合计（j 从 0 到 NDAY）
codes, uniq = pd.factorize(daily["permno"])
NP = len(uniq)
pos = codes.astype("int64") * (NDAY + 1) + daily["td_idx"].to_numpy("int64") + 1
cum_log = np.zeros(NP * (NDAY + 1), dtype="float64")   # log(1+pred) 的累积 → 连乘
cum_raw = np.zeros(NP * (NDAY + 1), dtype="float64")   # pred 的累积 → 算算术平均
cum_n = np.zeros(NP * (NDAY + 1), dtype="float64")
cum_d = np.zeros(NP * (NDAY + 1), dtype="float64")
_pm = daily["pred_mean"].to_numpy("float64")
np.add.at(cum_log, pos, np.log1p(_pm))
np.add.at(cum_raw, pos, _pm)
np.add.at(cum_n, pos, daily["n_news"].to_numpy("float64"))
np.add.at(cum_d, pos, 1.0)
for arr in (cum_log, cum_raw, cum_n, cum_d):
    v = arr.reshape(NP, NDAY + 1)
    v.cumsum(axis=1, out=v)

pmap = pd.Series(np.arange(NP), index=uniq)
_row = ev["permno"].map(pmap)
has = _row.notna().to_numpy()
row = _row.fillna(0).to_numpy().astype("int64")
t0 = ev["td0_idx"].to_numpy("int64")

out = ev[["eid", "permno", "td0_idx"]].copy()
for a, b in WINDOWS:
    tag = wname(a, b)
    lo = np.clip(t0 + a, 0, NDAY)            # [lo, hi] 的合计 = cum[hi+1] - cum[lo]
    hi = np.clip(t0 + b, -1, NDAY - 1)
    i_hi = row * (NDAY + 1) + hi + 1
    i_lo = row * (NDAY + 1) + lo
    valid = has & (hi >= lo)
    lg = np.where(valid, cum_log[i_hi] - cum_log[i_lo], np.nan)
    rw = np.where(valid, cum_raw[i_hi] - cum_raw[i_lo], np.nan)
    nn = np.where(valid, cum_n[i_hi] - cum_n[i_lo], 0.0)
    nd = np.where(valid, cum_d[i_hi] - cum_d[i_lo], 0.0)
    out[f"llm_n_{tag}"] = nn.astype("int32")
    out[f"llm_ndays_{tag}"] = nd.astype("int8")
    # 与 CAR 同法：窗口内各日 (1+pred) 连乘再减 1
    out[f"llm_cum_{tag}"] = np.where(nd > 0, np.expm1(lg), np.nan)
    out[f"llm_mean_{tag}"] = np.where(nd > 0, rw / np.where(nd > 0, nd, 1), np.nan)
    print(f"  窗口 [{a:+d},{b:+d}] → 后缀 {tag}: 有新闻的事件 {int((nd > 0).sum()):,} "
          f"({(nd > 0).mean():.1%}) | 平均 {nn[nd > 0].mean():.2f} 条 / {nd[nd > 0].mean():.2f} 天",
          flush=True)

# 季度内十分位 rank，与 sue_rank 同构
qtr = ev["anndats"].dt.to_period("Q").to_numpy()
for a, b in WINDOWS:
    tag = wname(a, b)
    x = pd.Series(out[f"llm_cum_{tag}"].to_numpy(), index=out.index)
    dec = (x.groupby(qtr)
            .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1
                       if s.notna().sum() >= MIN_Q_RANK else np.nan))
    dec[x.isna()] = np.nan
    out[f"llm_rank_{tag}"] = (dec - 1) / 9.0

out.to_parquet(OUT, index=False)
print(f"\n→ {OUT}  {len(out):,} 行 × {out.shape[1]} 列")
print(out.drop(columns=["eid", "permno", "td0_idx"]).describe().T[
    ["count", "mean", "50%", "min", "max"]].to_string())