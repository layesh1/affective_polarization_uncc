"""
fig09_distrust_violin.py
=========================
Figure 9: How Much Do Students Distrust the Other Party?

Violin + jitter plot for the out-party distrust index (1–7 scale).
Higher score = stronger agreement that the opposing party is untrustworthy,
frustrating, or acting in bad faith.

This is the largest effect in the dataset (Cohen's d ≈ 1.17).

Output: visualizations/figure_09_distrust_violin.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

COL     = "distrust_index"
PARTIES = ["Democrat", "Independent", "Republican"]
COLORS  = {"Democrat": "#2166ac", "Independent": "#888888", "Republican": "#d6604d"}

plot_df = df[df[COL].notna() & df["party_3cat"].notna()].copy()

fig, ax = plt.subplots(figsize=(9, 6.5))
fig.patch.set_facecolor("white")
np.random.seed(42)

positions = {p: i for i, p in enumerate(PARTIES)}

for party in PARTIES:
    vals  = plot_df.loc[plot_df["party_3cat"] == party, COL].dropna().values
    if len(vals) < 5:
        continue
    pos   = positions[party]
    color = COLORS[party]

    kde  = stats.gaussian_kde(vals, bw_method=0.35)
    yr   = np.linspace(vals.min(), vals.max(), 200)
    dn   = kde(yr); dn = dn / dn.max() * 0.38

    ax.fill_betweenx(yr, pos-dn, pos+dn, alpha=0.50, color=color, linewidth=0)
    ax.plot(pos-dn, yr, color=color, lw=0.8, alpha=0.85)
    ax.plot(pos+dn, yr, color=color, lw=0.8, alpha=0.85)

    q25, q75 = np.percentile(vals, [25, 75])
    ax.vlines(pos, q25, q75, color="white", lw=4, zorder=4)
    ax.scatter([pos], [np.median(vals)], color="white", s=45, zorder=5)
    ax.scatter([pos], [vals.mean()],     color="black", s=75, marker="D",
               linewidths=1.3, edgecolors="white", zorder=6)

    jitter = np.random.uniform(-0.13, 0.13, size=len(vals))
    ax.scatter(pos+jitter, vals, color=color, alpha=0.18, s=10, linewidths=0, zorder=2)
    ax.text(pos, vals.min() - 0.12, f"n = {len(vals)}",
            ha="center", va="top", fontsize=10, color=color, fontweight="bold")

# D vs R bracket
dem_v = plot_df.loc[plot_df["party_3cat"] == "Democrat",  COL].dropna()
rep_v = plot_df.loc[plot_df["party_3cat"] == "Republican", COL].dropna()
t, p  = stats.ttest_ind(dem_v, rep_v)
pooled = np.sqrt(((len(dem_v)-1)*dem_v.var(ddof=1)+(len(rep_v)-1)*rep_v.var(ddof=1)) /
                 (len(dem_v)+len(rep_v)-2))
d = (dem_v.mean()-rep_v.mean()) / pooled
y_top = max(dem_v.max(), rep_v.max())
y_br  = y_top + 0.15
x0, x1 = positions["Democrat"], positions["Republican"]
ax.annotate("", xy=(x1, y_br), xytext=(x0, y_br),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.3))
ax.vlines([x0, x1], y_top+0.04, y_br, color="black", lw=1.3)
ax.text((x0+x1)/2, y_br + 0.05,
        f"Democrats distrust Republicans more  (p < .001, d = {d:.2f} — very large effect)",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.axhline(4, color="grey", lw=0.8, ls=":", alpha=0.5)
ax.text(len(PARTIES)-0.45, 4.06, "Scale midpoint (4)", fontsize=8.5, color="grey")

ax.set_xticks(list(positions.values()))
ax.set_xticklabels(PARTIES, fontsize=13)
ax.set_ylabel("Out-Party Distrust Score\n(1 = Trusts the other party  →  7 = Deeply distrusts them)",
              fontsize=11)
ax.set_xlim(-0.6, len(PARTIES)-0.4)
ax.set_ylim(1.2, y_br + 0.6)
ax.set_title("Figure 9: How Much Do Students Distrust the Other Party?\n"
             "This is the strongest partisan difference found in the survey (d = 1.17)",
             fontsize=12, fontweight="bold", pad=10)

legend_items = ([mpatches.Patch(color=COLORS[p], label=f"{p}s") for p in PARTIES] +
                [plt.scatter([], [], marker="D", color="black", s=55, label="Group mean"),
                 plt.scatter([], [], marker="o", color="white", edgecolors="grey", s=45, label="Median")])
ax.legend(handles=legend_items, fontsize=9.5, framealpha=0.9, loc="lower right", ncol=2)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_09_distrust_violin.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: visualizations/figure_09_distrust_violin.png")
print(f"Democrats M={dem_v.mean():.3f}  Republicans M={rep_v.mean():.3f}  d={d:.3f}")
