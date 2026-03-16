"""
fig08_ideology_distribution.py
================================
Figure 8: Do Democrats Think Liberal and Republicans Think Conservative?
          (Partisan Sorting)

Shows how ideology (political beliefs on a spectrum from very liberal to very
conservative) is distributed within each party.  Strong partisan sorting means
little overlap — Democrats cluster liberal, Republicans cluster conservative.

Output: visualizations/figure_08_ideology_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")
df = df[df["ideology_num"].notna()].copy()

PARTIES = ["Democrat", "Independent", "Republican"]
COLORS  = {"Democrat": "#2166ac", "Independent": "#888888", "Republican": "#d6604d"}
IDEO_LABELS = ["Very\nLiberal", "Liberal", "Somewhat\nLiberal", "Moderate",
               "Somewhat\nConservative", "Conservative", "Very\nConservative"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                gridspec_kw={"height_ratios": [2, 1]})
fig.patch.set_facecolor("white")

# ── Top: density curves ────────────────────────────────────────────────────────
for party in PARTIES:
    vals = df.loc[df["party_3cat"] == party, "ideology_num"].dropna()
    if len(vals) < 5:
        continue
    kde = stats.gaussian_kde(vals, bw_method=0.5)
    xs  = np.linspace(1, 7, 300)
    ax1.fill_between(xs, kde(xs), alpha=0.18, color=COLORS[party])
    ax1.plot(xs, kde(xs), color=COLORS[party], lw=2.5,
             label=f"{party}s  (n={len(vals)},  avg={vals.mean():.1f})")
    ax1.axvline(vals.mean(), color=COLORS[party], lw=1.5, ls="--", alpha=0.7)

ax1.set_xlim(1, 7)
ax1.set_xticks(range(1, 8))
ax1.set_xticklabels(IDEO_LABELS, fontsize=9.5)
ax1.set_ylabel("Density (how many students\nhave this ideology)", fontsize=10)
ax1.set_title("Figure 8: Do Students' Beliefs Align With Their Party?\n"
              "Dashed lines = group averages  |  Less overlap = stronger partisan sorting",
              fontsize=12, fontweight="bold", pad=10)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.spines[["top", "right"]].set_visible(False)

# Spearman r annotation
valid = df[["ideology_num", "party_num"]].dropna()
rho, p = stats.spearmanr(valid["party_num"], valid["ideology_num"])
ax1.text(0.98, 0.95,
         f"Partisan sorting:  r = {rho:.2f}  (very strong)",
         transform=ax1.transAxes, fontsize=9.5, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="grey"))

# ── Bottom: stacked proportion bars ───────────────────────────────────────────
props = {}
for party in PARTIES:
    vals   = df.loc[df["party_3cat"] == party, "ideology_num"].dropna()
    counts = vals.value_counts().reindex(range(1, 8), fill_value=0)
    props[party] = counts / counts.sum()

bottom = np.zeros(7)
for party in PARTIES:
    p_arr = np.array([props[party].get(i, 0) for i in range(1, 8)])
    ax2.bar(range(1, 8), p_arr, bottom=bottom, color=COLORS[party],
            alpha=0.82, edgecolor="white", lw=0.5)
    bottom += p_arr

ax2.set_xlim(0.5, 7.5)
ax2.set_xticks(range(1, 8))
ax2.set_xticklabels(IDEO_LABELS, fontsize=9.5)
ax2.set_ylabel("Share of\neach party", fontsize=10)
ax2.set_xlabel("Political Ideology  (1 = Very Liberal  →  7 = Very Conservative)", fontsize=10.5)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_08_ideology_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: visualizations/figure_08_ideology_distribution.png")
print(f"Partisan sorting: Spearman r = {rho:.3f}")
