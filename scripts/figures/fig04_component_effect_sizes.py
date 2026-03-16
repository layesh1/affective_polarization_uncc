"""
fig04_component_effect_sizes.py
================================
Figure 4: Where Do Democrats and Republicans Differ Most?

Grouped bar chart showing average scores for Democrats vs Republicans
on the three components of affective polarization. Error bars = 95% CI.
Cohen's d = standardized measure of how far apart the groups are.

Output: visualizations/figure_04_component_effect_sizes.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

DEM_COLOR = "#2166ac"
REP_COLOR = "#d6604d"

COMPONENTS = {
    "Moral Identity\n(sees party as\nmoral cause)":  "ap_moral",
    "Othering\n(sees out-party\nas alien)":           "ap_othering",
    "Social Aversion\n(avoids out-party\nmembers)":   "ap_aversion",
}

dems = df[df["party_3cat"] == "Democrat"]
reps = df[df["party_3cat"] == "Republican"]


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var(ddof=1)+(n2-1)*g2.var(ddof=1))/(n1+n2-2))
    return (g1.mean()-g2.mean()) / pooled

def ci95(v):
    return stats.t.ppf(0.975, df=len(v)-1) * v.std(ddof=1) / np.sqrt(len(v))


labels  = list(COMPONENTS.keys())
cols    = list(COMPONENTS.values())
x       = np.arange(len(labels))
bar_w   = 0.33

fig, ax = plt.subplots(figsize=(9.5, 6))
fig.patch.set_facecolor("white")

dem_means = [dems[c].dropna().mean() for c in cols]
rep_means = [reps[c].dropna().mean() for c in cols]
dem_cis   = [ci95(dems[c].dropna())  for c in cols]
rep_cis   = [ci95(reps[c].dropna())  for c in cols]
d_vals    = [cohens_d(dems[c].dropna(), reps[c].dropna()) for c in cols]

ax.bar(x - bar_w/2, dem_means, bar_w, color=DEM_COLOR, alpha=0.88,
       label="Democrats", edgecolor="white")
ax.bar(x + bar_w/2, rep_means, bar_w, color=REP_COLOR, alpha=0.88,
       label="Republicans", edgecolor="white")

ax.errorbar(x - bar_w/2, dem_means, yerr=dem_cis,
            fmt="none", color="black", capsize=5, lw=1.4, capthick=1.4)
ax.errorbar(x + bar_w/2, rep_means, yerr=rep_cis,
            fmt="none", color="black", capsize=5, lw=1.4, capthick=1.4)

# Annotate d above each pair
for i, d in enumerate(d_vals):
    dv = dems[cols[i]].dropna()
    rv = reps[cols[i]].dropna()
    _, p = stats.ttest_ind(dv, rv)
    sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
    y_top = max(dem_means[i]+dem_cis[i], rep_means[i]+rep_cis[i]) + 0.07
    ax.plot([x[i]-bar_w/2, x[i]+bar_w/2], [y_top, y_top], color="black", lw=1.2)
    size_word = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    ax.text(x[i], y_top + 0.03,
            f"d = {d:.2f}  ({size_word} difference)  {sig}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

# Labels
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylabel("Average Score  (1 = Low  →  5 = High)", fontsize=11)
ax.set_ylim(1, ax.get_ylim()[1] + 0.35)
ax.axhline(3, color="grey", lw=0.8, ls=":", alpha=0.5)
ax.text(len(x)-0.45, 3.04, "Scale midpoint", fontsize=8.5, color="grey")

ax.set_title("Figure 4: Where Do Democrats and Republicans Differ Most?\n"
             "Average scores on each component of political hostility  (error bars = 95% confidence interval)",
             fontsize=12, fontweight="bold", pad=10)

legend_patches = [mpatches.Patch(color=DEM_COLOR, alpha=0.88, label="Democrats"),
                  mpatches.Patch(color=REP_COLOR, alpha=0.88, label="Republicans")]
ax.legend(handles=legend_patches, fontsize=11, framealpha=0.9, loc="upper right")

# Key finding box
note = (
    "d = Cohen's d = standardized gap between groups\n"
    "Small: d < 0.5  |  Medium: d = 0.5–0.8  |  Large: d > 0.8\n"
    "*** = statistically significant at p < .001"
)
ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f0f0f0", alpha=0.9, ec="grey"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_04_component_effect_sizes.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_04_component_effect_sizes.png")
for l, d, dm, rm in zip(labels, d_vals, dem_means, rep_means):
    print(f"  {l.split(chr(10))[0]:<20}  Dem={dm:.3f}  Rep={rm:.3f}  d={d:.3f}")
