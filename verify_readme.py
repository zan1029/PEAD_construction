"""逐条核对 README.md 里引用的数字与实际产物是否一致。"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OK, BAD = [], []


def chk(label, actual, claimed, tol=0):
    good = (abs(actual - claimed) <= tol) if isinstance(claimed, (int, float)) else (actual == claimed)
    (OK if good else BAD).append((label, actual, claimed))


# --- 目录 ---
chk("data/ parquet 数", len(glob.glob("data/*.parquet")), 102)
chk("build/ parquet 数", len(glob.glob("build/*.parquet")), 16)
chk("export/ parquet 数", len(glob.glob("export/*.parquet")), 1)

# --- 原始数据行数 ---
for f, n in [("data/link_crsp_compustat.parquet", 33324), ("data/link_ibes_crsp.parquet", 30080),
             ("data/crsp_security_info.parquet", 193968), ("data/crsp_monthly.parquet", 3091767),
             ("data/compustat_annual.parquet", 388641), ("data/compustat_quarterly.parquet", 1607253),
             ("data/ibes_actuals.parquet", 742513), ("data/ibes_adjustment.parquet", 202959)]:
    chk(os.path.basename(f), pq.ParquetFile(f).metadata.num_rows, n)
for pre, n in [("crsp_daily", 60255050), ("ibes_detail", 16888927), ("tr13f", 113948360)]:
    chk(f"{pre}_* 合计", sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"data/{pre}_*.parquet")), n)
chk("export 面板行数", pq.ParquetFile("export/crsp_daily_ret_c2c_o2o.parquet").metadata.num_rows, 65834310)

# --- Step 1 宇宙 ---
u = pd.read_parquet("build/universe.parquet")
chk("universe 行数", len(u), 193968)
chk("permno 总数", u["permno"].nunique(), 41520)
chk("普通股 permno", u.loc[u["is_common"], "permno"].nunique(), 30517)
chk("+美国注册", u.loc[u["is_common"] & u["is_us"], "permno"].nunique(), 27472)
chk("in_universe permno", u.loc[u["in_universe"], "permno"].nunique(), 27016)
ccm = pd.read_parquet("build/link_ccm.parquet")
ibl = pd.read_parquet("build/link_ibes.parquet")
chk("permno 对多 gvkey", int((ccm.groupby("permno")["gvkey"].nunique() > 1).sum()), 745)
chk("ticker 对多 permno", int((ibl.groupby("ticker")["permno"].nunique() > 1).sum()), 631)

# --- Step 2 ---
mem = pd.read_parquet("build/port25_membership.parquet")
bench = pd.read_parquet("build/port25_bench_returns.parquet")
chk("membership 行数", len(mem), 135230)
chk("formation 起", int(mem["ffyear"].min()), 1995)
chk("formation 止", int(mem["ffyear"].max()), 2026)
chk("bench 行数", len(bench), 191825)
chk("bench 交易日数", bench["date"].nunique(), 7673)
chk("每日恒为 25 组", int(bench.groupby("date")["port25"].nunique().eq(25).all()), 1)

# --- Step 3 事件 ---
ev = pd.read_parquet("build/events.parquet")
chk("events 行数", len(ev), 517955)
chk("events permno 数", ev["permno"].nunique(), 14214)
chk("非交易日公告占比%", round(100 * ev["ann_on_nontrading"].mean(), 1), 1.6, tol=0.05)
chk("flag_lag_bad %", round(100 * ev["flag_lag_bad"].mean(), 2), 1.15, tol=0.01)
chk("LAG 中位数", float(ev.loc[~ev["flag_lag_bad"], "lag"].median()), 33)
chk("补报总行数", int(ev["flag_same_day_multi"].sum()), 13118)
chk("补报中旧财季", int((~ev["is_latest_pends_on_day"]).sum()), 8332)
chk("flag_beyond_link_end", int(ev["flag_beyond_link_end"].sum()), 5846)

# --- Step 4 CAR ---
car = pd.read_parquet("build/car.parquet")
for c, n in [("car_ann_c2c", 454218), ("car_drift_c2c", 454207),
             ("car_ann_o2o", 451493), ("car_drift_o2o", 448864)]:
    chk(f"{c} 有效样本", int(car[c].notna().sum()), n)
chk("car_ann_c2c 均值", round(float(car["car_ann_c2c"].mean()), 4), -0.0003, tol=2e-4)
chk("car_drift_c2c 均值", round(float(car["car_drift_c2c"].mean()), 4), -0.0160, tol=2e-4)
chk("ANN 满窗 %", round(100 * (car["n_days_ann_c2c"] == 2).mean(), 1), 98.9, tol=0.05)
chk("DRIFT 满窗 %", round(100 * (car["n_days_drift_c2c"] == 60).mean(), 1), 96.7, tol=0.05)
chk("corr ANN c2c-o2o", round(float(car["car_ann_c2c"].corr(car["car_ann_o2o"])), 2), 0.73, tol=0.005)
chk("corr DRIFT c2c-o2o", round(float(car["car_drift_c2c"].corr(car["car_drift_o2o"])), 2), 0.96, tol=0.005)

# --- Step 5 SUE ---
sue = pd.read_parquet("build/sue.parquet")
chk("有 SUE 的事件", int(sue["sue"].notna().sum()), 344026)
chk("flag_sue_dropped", int(sue["flag_sue_dropped"].sum()), 23801)
chk("sue_dec 错配", int((sue["sue_dec"].notna() & sue["sue"].isna()).sum()), 0)

# --- Step 7 大表 ---
panel = pd.read_parquet("build/pead_panel.parquet")
chk("panel 行数", len(panel), 517955)
chk("panel 列数", panel.shape[1], 72)
for c, v in [("size_dec", 96.6), ("bm_dec", 88.8), ("lnanalyst", 100.0), ("lag", 100.0),
             ("io", 96.9), ("evol", 93.9), ("epersist", 93.9), ("turn", 96.9)]:
    chk(f"{c} 覆盖率%", round(100 * panel[c].notna().mean(), 1), v, tol=0.05)
need = ["car_ann_c2c", "car_drift_c2c", "sue", "size_dec", "bm_dec", "lnanalyst", "lag", "io", "evol", "epersist", "turn"]
chk("可回归样本", int(panel[need].notna().all(axis=1).sum()), 303466)

# --- PEAD 验证（与回归准备.ipynb 的 sort_sample 同口径：有 SUE 十分位 + 同日多季报只留最新） ---
chk("有 SUE 的事件（全部）", int(panel["sue_dec"].notna().sum()), 344026)
chk_p = panel[panel["sue_dec"].notna() & panel["is_latest_pends_on_day"]]
chk("PEAD 组合排序样本", len(chk_p), 343609)
tab = chk_p.groupby("sue_dec")[["car_ann_c2c", "car_drift_c2c", "car_ann_o2o", "car_drift_o2o"]].mean() * 100
for c, d1, d10 in [("car_ann_c2c", -4.167, 3.925), ("car_drift_c2c", -3.535, 0.715),
                   ("car_ann_o2o", -3.595, 3.705), ("car_drift_o2o", -5.032, 0.390)]:
    chk(f"{c} D1", round(float(tab.loc[1, c]), 3), d1, tol=0.002)
    chk(f"{c} D10", round(float(tab.loc[10, c]), 3), d10, tol=0.002)


def welch(x, y):
    x, y = x.dropna().to_numpy(), y.dropna().to_numpy()
    return (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))


for c, t in [("car_ann_c2c", 82.6), ("car_drift_c2c", 14.6), ("car_ann_o2o", 86.5), ("car_drift_o2o", 18.3)]:
    chk(f"{c} t值", round(float(welch(chk_p.loc[chk_p["sue_dec"] == 10, c], chk_p.loc[chk_p["sue_dec"] == 1, c])), 1),
        t, tol=0.05)

# --- 事件时间路径与对冲组合（README §主要结果 / §9） ---
pt = pd.read_parquet("build/car_path.parquet")
chk("car_path 行数", len(pt), 4880)
chk("car_path 两套分位", sorted(pt["grouping"].unique()), ["current", "prior"])
for g, n in [("current", 307242), ("prior", 307242)]:
    q = pt[(pt["grouping"] == g) & (pt["conv"] == "C2C")].drop_duplicates("sue_dec")
    chk(f"路径事件数 {g}", int(q["n"].sum()), n)
for conv, pre, drift in [("C2C", 6.96, 5.17), ("O2O", 6.88, 6.20)]:
    q = pt[(pt["grouping"] == "current") & (pt["conv"] == conv)].pivot(
        index="event_day", columns="sue_dec", values="car")
    h = (q[10.0] - q[1.0]) * 100
    chk(f"对冲 公告前[-60,-1] {conv}", round(float(h.loc[-1]), 2), pre, tol=0.005)
    chk(f"对冲 纯漂移[2,61] {conv}", round(float(h.loc[61] - h.loc[1]), 2), drift, tol=0.005)
for conv, sp in [("C2C", 4.31), ("O2O", 5.52)]:
    q = pt[(pt["grouping"] == "prior") & (pt["conv"] == conv)]
    chk(f"上季断点存在 {conv}", int(len(q) > 0), 1)
chk("PEAD 图数量", len(glob.glob("build/pead_paths*.png")), 3)


# --- Baseline 回归结果（README「主要结果」一节） ---
reg = pd.read_csv("build/baseline_results.csv")
reg = reg[reg["spec"] == "(0a) no interactions"].set_index(["window", "returns"])
for (w, c), b, t in [(("ANN", "C2C"), 0.0795, 100.5), (("DRIFT", "C2C"), 0.0196, 13.8),
                     (("ANN", "O2O"), 0.0724, 98.3), (("DRIFT", "O2O"), 0.0301, 20.5)]:
    chk(f"回归 β₁ {w} {c}", round(float(reg.loc[(w, c), "beta_SUE"]), 4), b, tol=5e-5)
    chk(f"回归 t  {w} {c}", round(float(reg.loc[(w, c), "t_firm"]), 1), t, tol=0.05)
chk("回归样本量 N", int(reg.loc[("ANN", "C2C"), "N"]), 302274)

# --- 数据字典 ---
dd = pd.read_csv("build/data_dictionary.csv")
chk("字典行数", len(dd), 72)
chk("字典未登记列", int(dd["说明"].astype(str).str.startswith("⚠️").sum()), 0)

print(f"通过 {len(OK)} 项")
if BAD:
    print(f"\n❌ 不一致 {len(BAD)} 项:")
    for lab, a, c in BAD:
        print(f"  {lab:26s} 实际={a}  README 写的={c}")
else:
    print("✅ README 中所有可核对的数字与实际产物完全一致")
