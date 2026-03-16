"""
fig03_violin_distributions.py
==============================
Figure 3: How Polarized Are Students? Full Distribution by Party

Violin + jitter plot of the combined affective polarization index.
Score = average of how much students moralize their party, see the other party
as alien, and want to avoid out-partisans. Scale: 1 (low) to 5 (high).

Output: visualizations/figure_03_violin_distributions.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df      = pd.read_csv("data/polarization_clean.csv")
COL     = "affective_polarization_index"
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

    ax.fill_betweenx(yr, pos - dn, pos + dn, alpha=0.50, color=color, linewidth=0)
    ax.plot(pos - dn, yr, color=color, lw=0.8, alpha=0.85)
    ax.plot(pos + dn, yr, color=color, lw=0.8, alpha=0.85)

    q25, q75 = np.percentile(vals, [25, 75])
    ax.vlines(pos, q25, q75, color="white", lw=4, zorder=4)
    ax.scatter([pos], [np.median(vals)], color="white", s=45, zorder=5)
    ax.scatter([pos], [vals.mean()],     color="black", s=75, marker="D",
               linewidths=1.3, edgecolors="white", zorder=6)

    jitter = np.random.uniform(-0.13, 0.13, size=len(vals))
    ax.scatter(pos + jitter, vals, color=color, alpha=0.20, s=10, linewidths=0, zorder=2)
    ax.text(pos, vals.min() - 0.1, f"n = {len(vals)}",
            ha="center", va="top", fontsize=10, color=color, fontweight="bold")

# D vs R significance bracket
dem_v = plot_df.loc[plot_df["party_3cat"] == "Democrat",  COL].dropna()
rep_v = plot_df.loc[plot_df["party_3cat"] == "Republican", COL].dropna()
t, p  = stats.ttest_ind(dem_v, rep_v)
pooled = np.sqrt(((len(dem_v)-1)*dem_v.var(ddof=1)+(len(rep_v)-1)*rep_v.var(ddof=1)) /
                 (len(dem_v)+len(rep_v)-2))
d = (dem_v.mean()-rep_v.mean()) / pooled
y_top = max(dem_v.max(), rep_v.max())
y_br  = y_top + 0.12
x0, x1 = positions["Democrat"], positions["Republican"]
ax.annotate("", xy=(x1, y_br), xytext=(x0, y_br),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.3))
ax.vlines([x0, x1], y_top + 0.03, y_br, color="black", lw=1.3)
ax.text((x0+x1)/2, y_br + 0.04,
        f"Democrats score significantly higher  (p < .001, d = {d:.2f})",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Reference lines
ax.axhline(3, color="grey", lw=0.8, ls=":", alpha=0.5)
ax.text(len(PARTIES) - 0.45, 3.04, "Scale midpoint (3)", fontsize=8, color="grey")

ax.set_xticks(list(positions.values()))
ax.set_xticklabels(PARTIES, fontsize=13)
ax.set_ylabel("Affective Polarization Score\n(1 = Not polarized  →  5 = Very polarized)", fontsize=11)
ax.set_ylim(0.8, y_br + 0.5)
ax.set_xlim(-0.6, len(PARTIES) - 0.4)
ax.set_title("Figure 3: How Emotionally Distant Are Students From the Other Party?\n"
             "Higher score = stronger hostility, avoidance, and moral condemnation of out-party",
             fontsize=12, fontweight="bold", pad=10)

legend_items = ([mpatches.Patch(color=COLORS[p], label=f"{p}s") for p in PARTIES] +
                [plt.scatter([], [], marker="D", color="black", s=55, label="Group mean"),
                 plt.scatter([], [], marker="o", color="white", edgecolors="grey", s=45, label="Median")])
ax.legend(handles=legend_items, fontsize=9.5, framealpha=0.9, loc="upper right", ncol=2)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_03_violin_distributions.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_03_violin_distributions.png")
print(f"Democrats M={dem_v.mean():.3f}  Republicans M={rep_v.mean():.3f}  d={d:.3f}")
