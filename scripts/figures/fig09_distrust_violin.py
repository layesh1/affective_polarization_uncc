"""
fig09_distrust_violin.py
=========================
Figure 9: Out-Party Distrust Distribution by Party

Violin + jitter plot for the distrust index (1–7 scale, higher = more distrust
of the opposing party).  This is one of the strongest effects in the dataset
(Cohen's d ≈ 1.17) and deserves its own figure.

Annotates the Democrat–Republican comparison with t-statistic and d.

Output:
    visualizations/figure_09_distrust_violin.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

PARTY_ORDER  = ["Democrat", "Independent", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}

INDEX_COL = "distrust_index"
plot_df = df[df[INDEX_COL].notna() & df["party_3cat"].notna()].copy()

fig, ax = plt.subplots(figsize=(8, 6))

np.random.seed(42)
positions = {p: i for i, p in enumerate(PARTY_ORDER)}

for party in PARTY_ORDER:
    vals  = plot_df.loc[plot_df["party_3cat"] == party, INDEX_COL].dropna().values
    if len(vals) < 5:
        continue
    pos   = positions[party]
    color = PARTY_COLORS[party]

    kde   = stats.gaussian_kde(vals, bw_method=0.35)
    yr    = np.linspace(vals.min(), vals.max(), 200)
    dens  = kde(yr)
    dn    = dens / dens.max() * 0.38

    ax.fill_betweenx(yr, pos - dn, pos + dn, alpha=0.55, color=color, linewidth=0)
    ax.plot(pos - dn, yr, color=color, linewidth=0.7, alpha=0.9)
    ax.plot(pos + dn, yr, color=color, linewidth=0.7, alpha=0.9)

    q25, q75 = np.percentile(vals, [25, 75])
    median   = np.median(vals)
    ax.vlines(pos, q25, q75, color="white", linewidth=3.5, zorder=4)
    ax.scatter([pos], [median], color="white", s=40, zorder=5)
    ax.scatter([pos], [vals.mean()], color="black", s=70, marker="D",
               linewidths=1.2, edgecolors="white", zorder=6)

    jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(pos + jitter, vals, color=color, alpha=0.20, s=10,
               linewidths=0, zorder=2)
    ax.text(pos, vals.min() - 0.1, f"n={len(vals)}",
            ha="center", va="top", fontsize=9, color=color, fontweight="bold")

# Annotation
dem_v = plot_df.loc[plot_df["party_3cat"] == "Democrat",  INDEX_COL].dropna()
rep_v = plot_df.loc[plot_df["party_3cat"] == "Republican", INDEX_COL].dropna()
t, p  = stats.ttest_ind(dem_v, rep_v)
pooled_sd = np.sqrt(((len(dem_v)-1)*dem_v.var(ddof=1) + (len(rep_v)-1)*rep_v.var(ddof=1)) /
                    (len(dem_v) + len(rep_v) - 2))
d = (dem_v.mean() - rep_v.mean()) / pooled_sd
y_br = max(dem_v.max(), rep_v.max()) + 0.15
x0, x1 = positions["Democrat"], positions["Republican"]
ax.annotate("", xy=(x1, y_br), xytext=(x0, y_br),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
ax.vlines([x0, x1], max(dem_v.max(), rep_v.max()) + 0.05, y_br, color="black", lw=1.2)
ax.text((x0+x1)/2, y_br + 0.05,
        f"t = {t:.2f}, p < .001, d = {d:.2f}  (very large effect)",
        ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(list(positions.values()))
ax.set_xticklabels(PARTY_ORDER, fontsize=12)
ax.set_ylabel("Out-Party Distrust Index (1–7)", fontsize=11)
ax.set_title("Figure 9: Out-Party Distrust by Party\n"
             "◆ = mean; ● = median; dots = individual respondents",
             fontsize=12, fontweight="bold")
legend_patches = [mpatches.Patch(color=PARTY_COLORS[p], label=p) for p in PARTY_ORDER]
ax.legend(handles=legend_patches, fontsize=9, framealpha=0.85, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.6, len(PARTY_ORDER) - 0.4)

plt.tight_layout()
plt.savefig("visualizations/figure_09_distrust_violin.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_09_distrust_violin.png")
print(f"\nDistrust  Dems: M={dem_v.mean():.3f}  Reps: M={rep_v.mean():.3f}  d={d:.3f}")
