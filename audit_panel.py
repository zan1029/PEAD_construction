"""对 pead_panel 做独立审计：恒等式、前视偏差、取值域、重复。

与 verify_readme.py 的区别：那个核对文档数字，这个检查数据内部逻辑是否自洽。
"""
import numpy as np
import pandas as pd

OK, BAD = [], []


def chk(label, passed, detail=""):
    (OK if passed else BAD).append((label, detail))


p = pd.read_parquet("build/pead_panel.parquet")
print(f"审计 {len(p):,} 行 × {p.shape[1]} 列\n")

# ---------- 1. 恒等式 ----------
for tag in ["c2c", "o2o"]:
    for w in ["ann", "drift"]:
        m = p[f"car_{w}_{tag}"].notna()
        d = (p.loc[m, f"car_{w}_{tag}"] - (p.loc[m, f"stk_{w}_{tag}"] - p.loc[m, f"bench_{w}_{tag}"])).abs()
        chk(f"CAR 恒等式 {w}_{tag}", d.max() < 1e-12, f"最大偏差 {d.max():.2e}")

m = p["sue"].notna()
d = (p.loc[m, "sue"] - (p.loc[m, "actual_eps"] - p.loc[m, "consensus_f"]) / p.loc[m, "price_adj"]).abs()
chk("SUE 恒等式", d.max() < 1e-9, f"最大偏差 {d.max():.2e}")

d = (p["lag2"] - p["lag"] ** 2).abs().max()
chk("LAG² 恒等式", d == 0)
d = (p["lag3"] - p["lag"] ** 3).abs().max()
chk("LAG³ 恒等式", d == 0)
d = (p["lnanalyst"] - np.log1p(p["n_analyst_cover"].fillna(0))).abs().max()
chk("LNANALYST 恒等式", d < 1e-12)
chk("port25 = (size_q−1)×5+bm_q 值域", p["port25"].dropna().between(1, 25).all())

# ---------- 2. 前视偏差 ----------
chk("共识预测早于公告日", (p.loc[p["fcst_last_dt"].notna(), "fcst_last_dt"] <
                          p.loc[p["fcst_last_dt"].notna(), "anndats"]).all())
m = p["io"].notna()
chk("13F 报告期不晚于公告日", (p.loc[m, "rdate"] <= p.loc[m, "anndats"]).all(),
    f"最大滞后 {p.loc[m, 'io_stale_days'].max():.0f} 天")
m = p["evol"].notna()
chk("Compustat 财季末不晚于公告日", (p.loc[m, "datadate"] <= p.loc[m, "anndats"]).all(),
    f"最大滞后 {p.loc[m, 'eps_stale_days'].max():.0f} 天")
chk("td0 不早于公告日", (p["td0"] >= p["anndats"]).all())
chk("pends 早于公告日（除异常样本）", (p.loc[~p["flag_lag_bad"], "pends"] <=
                                       p.loc[~p["flag_lag_bad"], "anndats"]).all())
# formation 年：7月起用当年，1-6月用上年
exp_ff = np.where(p["anndats"].dt.month >= 7, p["anndats"].dt.year, p["anndats"].dt.year - 1)
chk("ffyear 的 7 月切换规则", (p["ffyear"] == exp_ff).all())

# ---------- 3. 取值域 ----------
chk("n_days_ann ≤ 2", p["n_days_ann_c2c"].max() <= 2)
chk("n_days_drift ≤ 60", p["n_days_drift_c2c"].max() <= 60)
chk("sue_dec ∈ 1..10", p["sue_dec"].dropna().between(1, 10).all())
chk("size_dec ∈ 1..10", p["size_dec"].dropna().between(1, 10).all())
chk("bm_dec ∈ 1..10", p["bm_dec"].dropna().between(1, 10).all())
chk("io ∈ (0,2)", p["io"].dropna().between(0, 2).all())
chk("epersist ∈ [−1,1]", p["epersist"].dropna().between(-1, 1).all())
chk("evol ≥ 0", (p["evol"].dropna() >= 0).all())
chk("turn ≥ 0", (p["turn"].dropna() >= 0).all())
chk("CAR ≥ −1 附近", p["car_ann_c2c"].dropna().min() > -1.1,
    f"min {p['car_ann_c2c'].min():.3f}")

# ---------- 4. 缺失一致性 ----------
chk("sue_dec 不出现在 sue 缺失处", not (p["sue_dec"].notna() & p["sue"].isna()).any())
chk("size_dec 不出现在 me_jan 缺失处", not (p["size_dec"].notna() & p["me_jan"].isna()).any())
chk("bm_dec 不出现在 bm_raw 缺失处", not (p["bm_dec"].notna() & p["bm_raw"].isna()).any())
chk("CAR 缺失 ⟺ port25 缺失或个股无数据",
    (p.loc[p["port25"].isna(), "car_ann_c2c"].isna()).all())
chk("被清洗的 SUE 已置空", p.loc[p["flag_sue_dropped"], "sue"].isna().all())

# ---------- 5. 重复 ----------
chk("eid 唯一", p["eid"].is_unique)
chk("(permno, anndats, pends) 唯一", not p.duplicated(["permno", "anndats", "pends"]).any())
dup_car = p[p["flag_same_day_multi"]].groupby(["permno", "anndats"])["car_ann_c2c"].nunique(dropna=True)
chk("补报行共享同一 CAR（应如此）", (dup_car <= 1).all(),
    f"{(dup_car > 1).sum()} 个 permno-日 的 CAR 不一致")

# ---------- 6. 十分位均衡性 ----------
cnt = p["sue_dec"].value_counts()
chk("SUE 各十分位样本量均衡", cnt.max() / cnt.min() < 1.05, f"最大/最小 = {cnt.max()/cnt.min():.3f}")
q = p.dropna(subset=["sue", "sue_dec"]).groupby("sue_dec")["sue"].median()
chk("SUE 十分位与 SUE 单调对应", (q.diff().dropna() > 0).all())

print(f"通过 {len(OK)} 项")
for lab, det in OK:
    if det:
        print(f"   {lab}: {det}")
if BAD:
    print(f"\n❌ 未通过 {len(BAD)} 项:")
    for lab, det in BAD:
        print(f"   {lab}  {det}")
else:
    print("\n✅ 全部通过")
