"""
fig08_ideology_distribution.py
================================
Figure 8: Ideology Distribution by Party (Partisan Sorting)

Overlapping histograms / density curves showing the ideology spread (1=Very
Liberal → 7=Very Conservative) for Democrats, Independents, and Republicans.

Demonstrates the degree of partisan sorting: how tightly ideology clusters
within each party, and where the distributions overlap.

Output:
    visualizations/figure_08_ideology_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")
df = df[df["ideology_num"].notna()].copy()

PARTY_ORDER  = ["Democrat", "Independent", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}
IDEOLOGY_LABELS = ["Very\nLiberal", "Liberal", "Somewhat\nLiberal", "Moderate",
                   "Somewhat\nConservative", "Conservative", "Very\nConservative"]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 1]})

# ── Top: KDE density curves ──
ax = axes[0]
for party in PARTY_ORDER:
    vals = df.loc[df["party_3cat"] == party, "ideology_num"].dropna()
    if len(vals) < 5:
        continue
    kde = stats.gaussian_kde(vals, bw_method=0.5)
    xs  = np.linspace(1, 7, 300)
    ax.plot(xs, kde(xs), color=PARTY_COLORS[party], linewidth=2.5, label=f"{party} (n={len(vals)})")
    ax.fill_between(xs, kde(xs), alpha=0.12, color=PARTY_COLORS[party])
    # Mean line
    ax.axvline(vals.mean(), color=PARTY_COLORS[party], linewidth=1.2, linestyle="--", alpha=0.7)

ax.set_xlim(1, 7)
ax.set_xticks(range(1, 8))
ax.set_xticklabels(IDEOLOGY_LABELS, fontsize=8.5)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Figure 8: Ideology Distribution by Party (Partisan Sorting)\n"
             "Dashed lines = group means", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.85)
ax.spines[["top", "right"]].set_visible(False)

# ── Bottom: stacked bar showing proportions at each ideology level ──
ax2 = axes[1]
props = {}
for party in PARTY_ORDER:
    vals = df.loc[df["party_3cat"] == party, "ideology_num"].dropna()
    counts = vals.value_counts().reindex(range(1, 8), fill_value=0)
    props[party] = counts / counts.sum()

bottom = np.zeros(7)
x = np.arange(1, 8)
for party in PARTY_ORDER:
    p = np.array([props[party].get(i, 0) for i in range(1, 8)])
    ax2.bar(x, p, bottom=bottom, color=PARTY_COLORS[party],
            alpha=0.82, label=party, edgecolor="white", linewidth=0.5)
    bottom += p

ax2.set_xlim(0.5, 7.5)
ax2.set_xticks(range(1, 8))
ax2.set_xticklabels(IDEOLOGY_LABELS, fontsize=8.5)
ax2.set_ylabel("Proportion", fontsize=11)
ax2.set_xlabel("Ideology (1=Very Liberal → 7=Very Conservative)", fontsize=10)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_08_ideology_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_08_ideology_distribution.png")

# Spearman r
valid = df[["ideology_num", "party_num"]].dropna()
rho, p = stats.spearmanr(valid["party_num"], valid["ideology_num"])
print(f"Partisan sorting — Spearman r = {rho:.3f}, p = {p:.4f}")
print("\nIdeology means by party:")
for party in PARTY_ORDER:
    g = df.loc[df["party_3cat"] == party, "ideology_num"].dropna()
    print(f"  {party:<14}  M={g.mean():.2f}  SD={g.std():.2f}  N={len(g)}")
