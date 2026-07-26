"""PEAD 事件时间路径：每个事件公告后 0..61 个交易日的累计异常收益，按 SUE 十分位平均。

这是 BT1989 记录 PEAD 异象的经典方式（组合排序 + 事件时间累计曲线），
与 HLT2009 的截面回归互为补充。

窗口 [-60, +61]，基线为第 -61 个交易日，因此曲线从 0 出发。
输出 build/car_path.parquet: sue_dec × event_day × 口径 的平均 CAR。
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
                    columns=["eid", "permno", "anndats", "td0_idx", "port25", "sue_dec",
                             "is_latest_pends_on_day", "flag_lag_bad"])
ev = p[p["sue_dec"].notna()].copy()
ev["permno"] = ev["permno"].astype("int64")
ev["year"] = ev["anndats"].dt.year
print(f"事件 {len(ev):,} | 十分位 {ev['sue_dec'].min():.0f}-{ev['sue_dec'].max():.0f}", flush=True)

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


out = []
for tag in ["c2c", "o2o"]:
    paths = np.full((len(ev), KMAX - KMIN + 1), np.nan, dtype="float32")
    pos_of_eid = {e: i for i, e in enumerate(ev["eid"].to_numpy())}
    for y in sorted(ev["year"].unique()):
        sub = ev[ev["year"] == y]
        if sub.empty:
            continue
        r = year_returns(tag, y, y + 1)
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

    df = pd.DataFrame(paths, columns=[f"k{k}" for k in range(KMIN, KMAX + 1)])
    df["sue_dec"] = ev["sue_dec"].to_numpy()
    g = df.groupby("sue_dec").mean().stack().reset_index()
    g.columns = ["sue_dec", "k", "car"]
    g["event_day"] = g["k"].str[1:].astype(int)
    g["conv"] = tag.upper()
    n = df.groupby("sue_dec")["k0"].count().rename("n").reset_index()
    out.append(g.merge(n, on="sue_dec")[["conv", "sue_dec", "event_day", "car", "n"]])

res = pd.concat(out, ignore_index=True)
res.to_parquet(f"{BUILD}/car_path.parquet", index=False)
print(f"\\n→ {BUILD}/car_path.parquet  {len(res):,} 行")
piv = res[res["conv"] == "C2C"].pivot(index="event_day", columns="sue_dec", values="car")
print("\\nC2C 各十分位的累计异常收益 (%)：")
print((piv.loc[[0, 5, 20, 40, 61]] * 100).round(2).to_string())
print("\\n对冲组合 (D10 − D1) 路径 (%)：")
for conv in ["C2C", "O2O"]:
    q = res[res["conv"] == conv].pivot(index="event_day", columns="sue_dec", values="car")
    h = (q[10.0] - q[1.0]) * 100
    print(f"  {conv}: 第-1天 {h.loc[-1]:.2f} | 第1天 {h.loc[1]:.2f} | 第10天 {h.loc[10]:.2f} | "
          f"第30天 {h.loc[30]:.2f} | 第61天 {h.loc[61]:.2f}")
