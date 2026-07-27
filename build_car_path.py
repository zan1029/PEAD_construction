"""PEAD 事件时间路径：每个事件公告后 0..61 个交易日的累计异常收益，按 SUE 十分位平均。

这是 BT1989 记录 PEAD 异象的经典方式（组合排序 + 事件时间累计曲线），
与 HLT2009 的截面回归互为补充。

窗口 [-60, +61]，基线为第 -61 个交易日，因此曲线从 0 出发。

两套分组同时产出（列 `grouping`）：
  current —— 当季截面分位（主设定，HLT2009 把 SUE 分位当控制变量的做法）
  prior   —— 上一季度截面分布定断点、当季 SUE 值对号入座（FOS1984/BT1989 的可实施版本）

输出 build/car_path.parquet: grouping × 口径 × sue_dec × event_day 的平均 CAR，
     build/car_path_events.npz: 逐事件路径矩阵（便于以后换分组重算，无需重扫收益）。
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA, BUILD, EXPORT = "data", "build", "export"
KMIN = -60          # 公告前 60 个交易日（BT1989 / FOS 图 1 覆盖公告前后共 120 天）
KMAX = 61
RET_FLOOR = -0.999999

# 交易日历
_d = [pd.read_parquet(p, columns=["dlycaldt"])["dlycaldt"].drop_duplicates()
      for p in sorted(glob.glob(f"{DATA}/crsp_daily_*.parquet"))]
cal = pd.to_datetime(pd.concat(_d)).drop_duplicates().sort_values().reset_index(drop=True)
del _d
cal_map = pd.Series(np.arange(len(cal), dtype="int32"), index=cal.values)
NDAY = len(cal)

# 事件（有 SUE 十分位的）
p = pd.read_parquet(f"{BUILD}/pead_panel.parquet",
                    columns=["eid", "permno", "anndats", "td0_idx", "port25", "sue", "sue_dec",
                             "is_latest_pends_on_day", "flag_lag_bad"])
# 与回归准备.ipynb 的 sort_sample 同口径：有 SUE 十分位 + 同日多季报只留最新一季
ev = p[p["sue_dec"].notna() & p["is_latest_pends_on_day"]].copy()
ev["permno"] = ev["permno"].astype("int64")
ev["year"] = ev["anndats"].dt.year

# FOS 分组：断点取自上一日历季度的 SUE 分布，当季的 SUE 值对号入座。
# 1996 年的事件用得上 1995Q3/Q4 的 SUE —— I/B/E/S 从 1995 年起有数据，那些事件本身
# 因为日历不完整没有 CAR、不进图，但可以拿来当断点，所以 1996Q1 也有上季分位。
ev["qtr"] = ev["anndats"].dt.to_period("Q")
qs = sorted(ev["qtr"].unique())
MIN_Q = 100                                      # 断点至少要有这么多观测才可信
bp = {q: (np.nanpercentile(v, np.arange(10, 100, 10)) if len(v) >= MIN_Q else None)
      for q in qs for v in [ev.loc[ev["qtr"] == q, "sue"].dropna().to_numpy()]}
ev["sue_dec_prior"] = np.nan
for i, q in enumerate(qs):
    cuts = bp[qs[i - 1]] if i > 0 else None
    if cuts is None:                             # 上一季不存在或样本太少
        continue
    m = (ev["qtr"] == q).to_numpy()
    ev.loc[m, "sue_dec_prior"] = np.searchsorted(cuts, ev.loc[m, "sue"].to_numpy(),
                                                 side="right") + 1
ev.loc[ev["sue"].isna(), "sue_dec_prior"] = np.nan
print(f"事件 {len(ev):,} | 当季分位 {ev['sue_dec'].min():.0f}-{ev['sue_dec'].max():.0f}"
      f" | 上季分位可得 {ev['sue_dec_prior'].notna().sum():,}", flush=True)

# 基准组合的累积对数收益矩阵（26 × NDAY）
bench = pd.read_parquet(f"{BUILD}/port25_bench_returns.parquet")
bench["td_idx"] = pd.to_datetime(bench["date"]).map(cal_map)
bench = bench.dropna(subset=["td_idx"])
bench["td_idx"] = bench["td_idx"].astype(int)
CUMB = {}
for tag, col in [("c2c", "bench_ret_c2c"), ("o2o", "bench_ret_o2o")]:
    piv = bench.pivot_table(index="port25", columns="td_idx", values=col).reindex(columns=range(NDAY))
    mat = np.zeros((26, NDAY))
    vals = np.nan_to_num(piv.to_numpy(dtype="float64"), nan=0.0)
    mat[piv.index.to_numpy().astype(int), :] = np.cumsum(np.log1p(np.maximum(vals, RET_FLOOR)), axis=1)
    CUMB[tag] = mat


def year_returns(tag, y0, y1):
    """产出 (permno, td_idx, ret) 三列，覆盖 [y0, y1] 两个日历年。"""
    if tag == "c2c":
        parts = []
        for y in range(y0, y1 + 1):
            f = f"{DATA}/crsp_daily_{y}.parquet"
            try:
                d = pd.read_parquet(f, columns=["permno", "dlycaldt", "dlyret"]).dropna(subset=["dlyret"])
            except FileNotFoundError:
                continue
            d["td_idx"] = pd.to_datetime(d["dlycaldt"]).map(cal_map)
            parts.append(d.dropna(subset=["td_idx"])[["permno", "td_idx", "dlyret"]]
                         .rename(columns={"dlyret": "ret"}))
        return pd.concat(parts, ignore_index=True) if parts else None
    lo = pd.Timestamp(f"{y0}-01-01").date()
    hi = pd.Timestamp(f"{y1}-12-31").date()
    pf = pq.ParquetFile(f"{EXPORT}/crsp_daily_ret_c2c_o2o.parquet")
    parts = []
    for b in pf.iter_batches(batch_size=3_000_000, columns=["PERMNO", "date", "O2O_RET"]):
        d = b.to_pandas().dropna(subset=["O2O_RET"])
        d = d[(d["date"] >= lo) & (d["date"] <= hi)]
        if d.empty:
            continue
        d["td_idx"] = pd.to_datetime(d["date"]).map(cal_map)
        parts.append(d.dropna(subset=["td_idx"])[["PERMNO", "td_idx", "O2O_RET"]]
                     .rename(columns={"PERMNO": "permno", "O2O_RET": "ret"}))
    return pd.concat(parts, ignore_index=True) if parts else None


out, RAW = [], {}
for tag in ["c2c", "o2o"]:
    paths = np.full((len(ev), KMAX - KMIN + 1), np.nan, dtype="float32")
    pos_of_eid = {e: i for i, e in enumerate(ev["eid"].to_numpy())}
    for y in sorted(ev["year"].unique()):
        sub = ev[ev["year"] == y]
        if sub.empty:
            continue
        # 前扩一年：公告前 60 天的基线（td0-61）对 1、2 月的公告会落到上一年，
        # 只载入 [y, y+1] 会把这些事件整条丢弃（1-2 月是 Q4 财报公告高峰，损失约 24%）
        r = year_returns(tag, y - 1, y + 1)
        if r is None or r.empty:
            continue
        r["permno"] = r["permno"].astype("int64")
        r["td_idx"] = r["td_idx"].astype("int32")
        d0, d1 = int(r["td_idx"].min()), int(r["td_idx"].max())
        pcodes, puniq = pd.factorize(r["permno"])
        # 稠密矩阵：缺失日贡献 0（等价于持有不动），cumsum 后即"最近可得的累积值"
        M = np.zeros((len(puniq), d1 - d0 + 1), dtype="float64")
        np.add.at(M, (pcodes, r["td_idx"].to_numpy() - d0),
                  np.log1p(np.maximum(r["ret"].to_numpy(dtype="float64"), RET_FLOOR)))
        np.cumsum(M, axis=1, out=M)

        pmap = pd.Series(np.arange(len(puniq)), index=puniq)
        rows = sub["permno"].map(pmap)
        ok = rows.notna().to_numpy()
        rows = rows.to_numpy()[ok].astype(int)
        base = (sub["td0_idx"].to_numpy()[ok] + KMIN - 1 - d0)   # 基线 = td0 + KMIN - 1
        p25 = sub["port25"].fillna(0).to_numpy()[ok].astype(int)
        has_b = sub["port25"].notna().to_numpy()[ok]
        eids = sub["eid"].to_numpy()[ok]
        idxs = np.array([pos_of_eid[e] for e in eids])
        B = CUMB[tag]
        base_ok = (base >= 0) & (base < M.shape[1])
        for j, k in enumerate(range(KMIN, KMAX + 1)):
            tgt = base + 1 + (k - KMIN)
            valid = base_ok & (tgt < M.shape[1]) & has_b
            if not valid.any():
                continue
            stk = np.expm1(M[rows[valid], tgt[valid]] - M[rows[valid], base[valid]])
            b0 = sub["td0_idx"].to_numpy()[ok][valid] + KMIN - 1
            b1 = b0 + 1 + (k - KMIN)
            b1 = np.minimum(b1, NDAY - 1)
            bch = np.expm1(B[p25[valid], b1] - B[p25[valid], np.maximum(b0, 0)])
            paths[idxs[valid], j] = (stk - bch).astype("float32")
        print(f"  {tag} {y}: {valid.sum():,}", flush=True)

    RAW[tag] = paths
    df = pd.DataFrame(paths, columns=[f"k{k}" for k in range(KMIN, KMAX + 1)])
    for grouping, dec_col in [("current", "sue_dec"), ("prior", "sue_dec_prior")]:
        df["_dec"] = ev[dec_col].to_numpy()
        g = df.groupby("_dec")[[f"k{k}" for k in range(KMIN, KMAX + 1)]].mean().stack().reset_index()
        g.columns = ["sue_dec", "k", "car"]
        g["event_day"] = g["k"].str[1:].astype(int)
        g["conv"] = tag.upper()
        g["grouping"] = grouping
        n = df.groupby("_dec")["k0"].count().rename("n").reset_index().rename(columns={"_dec": "sue_dec"})
        out.append(g.merge(n, on="sue_dec")[["grouping", "conv", "sue_dec", "event_day", "car", "n"]])
    df.drop(columns="_dec", inplace=True)

res = pd.concat(out, ignore_index=True)
res.to_parquet(f"{BUILD}/car_path.parquet", index=False)
np.savez_compressed(f"{BUILD}/car_path_events.npz", eid=ev["eid"].to_numpy(),
                    sue_dec=ev["sue_dec"].to_numpy(), sue_dec_prior=ev["sue_dec_prior"].to_numpy(),
                    kmin=KMIN, kmax=KMAX, **{t: RAW[t] for t in RAW})
print(f"\\n→ {BUILD}/car_path.parquet  {len(res):,} 行")
res = res[res["grouping"] == "current"]
piv = res[res["conv"] == "C2C"].pivot(index="event_day", columns="sue_dec", values="car")
print("\\nC2C 各十分位的累计异常收益 (%)：")
print((piv.loc[[0, 5, 20, 40, 61]] * 100).round(2).to_string())
print("\\n对冲组合 (D10 − D1) 路径 (%)：")
for conv in ["C2C", "O2O"]:
    q = res[res["conv"] == conv].pivot(index="event_day", columns="sue_dec", values="car")
    h = (q[10.0] - q[1.0]) * 100
    print(f"  {conv}: 第-1天 {h.loc[-1]:.2f} | 第1天 {h.loc[1]:.2f} | 第10天 {h.loc[10]:.2f} | "
          f"第30天 {h.loc[30]:.2f} | 第61天 {h.loc[61]:.2f}")
