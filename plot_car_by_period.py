"""PEAD 事件时间 CAR：2004-2026 整段 + 按四年切成 6 段。

数据直接来自 build/car_path_events.npz 的逐事件路径矩阵，年份由 eid 回 build/pead_panel.parquet
取 anndats。不需要重扫收益，几秒钟出图。

产出：
  build/pead_paths_2004_2026.png        整段，C2C | O2O
  build/pead_paths_subperiods_C2C.png   6 段 × C2C
  build/pead_paths_subperiods_O2O.png   6 段 × O2O
  build/pead_drift_by_period.png        D10-D1 的第 61 日漂移随时段变化
  build/car_path_by_period.parquet      period × conv × sue_dec × event_day 的平均 CAR
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BUILD = "build"
Y0, Y1 = 2004, 2026
PERIODS = [(2004, 2007), (2008, 2011), (2012, 2015),
           (2016, 2019), (2020, 2023), (2024, 2026)]
GROUPING = "current"      # current = 当季断点；改成 "prior" 用上一季断点（FOS）
REQUIRE_FULL = True       # 只留 [-1, +61] 全程有值的事件，各面板内组成不随 k 变化

# ---------------------------------------------------------------- 数据
z = np.load(f"{BUILD}/car_path_events.npz")
KMIN, KMAX = int(z["kmin"]), int(z["kmax"])
ks = np.arange(KMIN, KMAX + 1)
dec = z["sue_dec"] if GROUPING == "current" else z["sue_dec_prior"]

_p = pd.read_parquet(f"{BUILD}/pead_panel.parquet",
                     columns=["eid", "anndats", "sue_dec", "is_latest_pends_on_day"])
_ev = _p[_p["sue_dec"].notna() & _p["is_latest_pends_on_day"]]
assert np.array_equal(_ev["eid"].to_numpy(), z["eid"]), "npz 与 panel 事件顺序不一致"
year = _ev["anndats"].dt.year.to_numpy()

MATS = {"C2C": z["c2c"], "O2O": z["o2o"]}
i_m1, i_61 = np.where(ks == -1)[0][0], np.where(ks == 61)[0][0]


def curves(conv, y0, y1):
    """返回 event_day × decile 的平均 CAR（%），已以公告前一日为基线归零；附各组事件数。"""
    M = MATS[conv]
    m = (year >= y0) & (year <= y1) & ~np.isnan(dec)
    if REQUIRE_FULL:
        m &= ~np.isnan(M[:, i_m1]) & ~np.isnan(M[:, i_61])
    d = pd.DataFrame(M[m], columns=ks)
    d["_dec"] = dec[m]
    g = d.groupby("_dec").mean().T * 100
    g.columns = [int(c) for c in g.columns]
    return g.loc[0:] - g.loc[-1], d.groupby("_dec").size()


# 落盘一份长表，方便另做表格/回归
rows = []
for label, (y0, y1) in [(f"{Y0}-{Y1}", (Y0, Y1))] + [(f"{a}-{b}", (a, b)) for a, b in PERIODS]:
    for conv in ("C2C", "O2O"):
        g, n = curves(conv, y0, y1)
        t = g.stack().reset_index()
        t.columns = ["event_day", "sue_dec", "car_pct"]
        t["period"], t["conv"] = label, conv
        t["n"] = t["sue_dec"].map(n)
        rows.append(t)
by_period = pd.concat(rows, ignore_index=True)[
    ["period", "conv", "sue_dec", "event_day", "car_pct", "n"]]
by_period.to_parquet(f"{BUILD}/car_path_by_period.parquet", index=False)

# ---------------------------------------------------------------- 样式（与 pead_paths.png 一致）
RAMP = LinearSegmentedColormap.from_list("sue", ["#e34948", "#8a8a86", "#2a78d6"])
COLORS = {d: RAMP((d - 1) / 9) for d in range(1, 11)}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
handles = [plt.Line2D([], [], color=COLORS[d], lw=2.4) for d in range(1, 11)]
LEG_TITLE = "SUE decile  (D1 = most negative earnings surprise, D10 = most positive)"


def dress(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


# ---------------------------------------------------------------- 图 1：整段 2004-2026
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, facecolor=SURFACE)
n_tot = 0
for ax, conv in zip(axes, ["C2C", "O2O"]):
    d, n = curves(conv, Y0, Y1)
    n_tot = max(n_tot, int(n.sum()))
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvspan(0, 1, color="#000000", alpha=0.05, lw=0, zorder=0)
    for dd in range(1, 11):
        ax.plot(d.index, d[dd], color=COLORS[dd], lw=2.0, solid_capstyle="round", zorder=3)
    for dd in (10, 1):
        ax.annotate(f"D{dd}  {d[dd].iloc[-1]:+.1f}%", xy=(61, d[dd].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", color=COLORS[dd],
                    fontsize=10, fontweight="bold", va="center")
    ax.set_xlim(0, 74)
    ax.set_xlabel("Trading days after announcement", color=INK2, fontsize=10)
    dress(ax, f"{conv} returns")
axes[0].set_ylabel("Cumulative abnormal return (%)", color=INK2, fontsize=10)
fig.legend(handles, [f"D{d}" for d in range(1, 11)], loc="lower center", ncol=10,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04),
           title=LEG_TITLE, title_fontsize=9)
fig.suptitle("Post-Earnings-Announcement Drift, 2004-2026", fontsize=13.5,
             fontweight="bold", color=INK, x=0.5, y=1.0)
fig.text(0.5, 0.935, f"{n_tot:,} earnings announcements;  "
                     "benchmark = matched size x B/M portfolio (25 groups)",
         ha="center", fontsize=9.5, color=INK2)
fig.tight_layout(rect=[0, 0.02, 1, 0.93])
fig.savefig(f"{BUILD}/pead_paths_2004_2026.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)

# ---------------------------------------------------------------- 图 2：6 个四年段（每个口径一张）
for conv in ("C2C", "O2O"):
    segs = [(f"{a}-{b}",) + curves(conv, a, b) for a, b in PERIODS]
    lo = min(s[1].values.min() for s in segs)
    hi = max(s[1].values.max() for s in segs)
    pad = (hi - lo) * 0.08
    fg, axs = plt.subplots(2, 3, figsize=(15, 8.6), sharey=True, facecolor=SURFACE,
                           gridspec_kw={"wspace": 0.08, "hspace": 0.3})
    for ax, (lab, d, n) in zip(axs.ravel(), segs):
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.axvspan(0, 1, color="#000000", alpha=0.05, lw=0, zorder=0)
        for dd in range(1, 11):
            ax.plot(d.index, d[dd], color=COLORS[dd], lw=1.7, solid_capstyle="round", zorder=3)
        for dd in (10, 1):
            ax.annotate(f"{dd}", xy=(61, d[dd].iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", color=COLORS[dd], fontsize=9,
                        fontweight="bold", va="center")
        spread = d[10].iloc[-1] - d[1].iloc[-1]
        ax.annotate(f"D10-D1 @ day 61:  {spread:+.2f}%", xy=(0.03, 0.04),
                    xycoords="axes fraction", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xlim(0, 70)
        ax.set_ylim(lo - pad, hi + pad)
        dress(ax, f"{lab}   (n = {int(n.sum()):,})")
    for ax in axs[1]:
        ax.set_xlabel("Trading days after announcement", color=INK2, fontsize=9.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("Cumulative abnormal return (%)", color=INK2, fontsize=9.5)
    fg.legend(handles, [f"D{d}" for d in range(1, 11)], loc="lower center", ncol=10,
              frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02),
              title=LEG_TITLE, title_fontsize=9)
    fg.suptitle(f"PEAD by sub-period, {conv} returns", fontsize=13.5, fontweight="bold",
                color=INK, x=0.5, y=1.0)
    fg.text(0.5, 0.955, "common y-axis across panels;  CAR rebased at day -1;  "
                        "benchmark = matched size x B/M portfolio",
            ha="center", fontsize=9.5, color=INK2)
    fg.tight_layout(rect=[0, 0.015, 1, 0.945])
    fg.savefig(f"{BUILD}/pead_paths_subperiods_{conv}.png", dpi=150,
               bbox_inches="tight", facecolor=SURFACE)

# ---------------------------------------------------------------- 图 3：漂移强度随时段
fig3, ax3 = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
labs = [f"{a}-{b}" for a, b in PERIODS]
x = np.arange(len(labs))
for conv, c, mk in [("C2C", "#2a78d6", "o"), ("O2O", "#e34948", "s")]:
    v = [curves(conv, a, b)[0].loc[61].pipe(lambda s: s[10] - s[1]) for a, b in PERIODS]
    ax3.plot(x, v, color=c, marker=mk, lw=2.2, ms=7, label=conv)
    for xi, vi in zip(x, v):
        ax3.annotate(f"{vi:+.1f}", xy=(xi, vi), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8.5, color=c, fontweight="bold")
ax3.axhline(0, color=INK2, lw=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(labs)
ax3.set_ylabel("D10 - D1 CAR at day +61 (%)", color=INK2, fontsize=10)
ax3.legend(frameon=False, fontsize=10)
dress(ax3, "Drift is shrinking: hedge-portfolio 61-day CAR by sub-period")
fig3.tight_layout()
fig3.savefig(f"{BUILD}/pead_drift_by_period.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)

# ---------------------------------------------------------------- 摘要表
print("D10 - D1 累计异常收益 (%)，按时段：")
tab = []
for lab, (a, b) in [(f"{Y0}-{Y1}", (Y0, Y1))] + list(zip(labs, PERIODS)):
    r = {"period": lab}
    for conv in ("C2C", "O2O"):
        g, n = curves(conv, a, b)
        h = g[10] - g[1]
        r[f"{conv}_d1"] = h.loc[1]
        r[f"{conv}_d10"] = h.loc[10]
        r[f"{conv}_d61"] = h.loc[61]
        r["n"] = int(n.sum())
    tab.append(r)
print(pd.DataFrame(tab).set_index("period").round(2).to_string())
print(f"\n→ {BUILD}/car_path_by_period.parquet 及 4 张 png")
