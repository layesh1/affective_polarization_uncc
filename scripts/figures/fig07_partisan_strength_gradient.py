"""
fig07_partisan_strength_gradient.py
=====================================
Figure 7: Partisan Strength Gradient — AP Components by Party ID Strength

X-axis: 7-category party ID (Strong Dem → Strong Rep) in order
Y-axis: Mean score on each AP component (1–5 scale)
Lines:  Moral Identity, Othering, Social Aversion

Each respondent's score comes from THEIR OWN PARTY'S version of each item
(e.g., a Strong Democrat's othering score = other1D–other3D average).

Expected pattern: monotonic decline from Strong Dem to Strong Rep, with
the D > R gap preserved across all strength levels. Strong partisans of
both parties should be highest within their wing.

Output:
    visualizations/figure_07_partisan_strength_gradient.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

# 7-category ordering (numeric → label, in order Strong Dem → Strong Rep)
STRENGTH_ORDER = [1, 1.5, 2, 3, 4.5, 4, 5]
STRENGTH_LABELS = [
    "Strong\nDem", "Lean\nDem", "Somewhat\nDem",
    "True\nIndep",
    "Lean\nRep", "Somewhat\nRep", "Strong\nRep",
]

# Correct order: sorted by numeric value
ORDERED_VALS   = sorted(STRENGTH_ORDER)  # [1, 1.5, 2, 3, 4, 4.5, 5]
ORDERED_LABELS = [
    "Strong\nDem", "Lean\nDem", "Somewhat\nDem",
    "True\nIndep",
    "Somewhat\nRep", "Lean\nRep", "Strong\nRep",
]

COMPONENTS = {
    "Moral Identity":  ("ap_moral",    "#1a9850", "o"),
    "Othering":        ("ap_othering", "#d73027", "s"),
    "Social Aversion": ("ap_aversion", "#f46d43", "^"),
}

fig, ax = plt.subplots(figsize=(9, 5.5))

for label, (col, color, marker) in COMPONENTS.items():
    x_pos, y_means, y_ci = [], [], []
    for i, val in enumerate(ORDERED_VALS):
        g = df.loc[np.isclose(df["party_combined"], val), col].dropna()
        if len(g) < 3:
            continue
        m  = g.mean()
        se = g.std(ddof=1) / np.sqrt(len(g))
        ci = se * stats.t.ppf(0.975, df=len(g) - 1)
        x_pos.append(i)
        y_means.append(m)
        y_ci.append(ci)

    x_pos   = np.array(x_pos)
    y_means = np.array(y_means)
    y_ci    = np.array(y_ci)

    ax.plot(x_pos, y_means, color=color, linewidth=2.2, marker=marker,
            markersize=8, label=label, zorder=4)
    ax.fill_between(x_pos, y_means - y_ci, y_means + y_ci,
                    color=color, alpha=0.15, zorder=2)

# Vertical dividers
ax.axvline(2.5, color="grey", linewidth=0.9, linestyle="--", alpha=0.5)
ax.axvline(3.5, color="grey", linewidth=0.9, linestyle="--", alpha=0.5)

# Shaded partisan regions
ax.axvspan(-0.5, 2.5, alpha=0.04, color="#2166ac")   # Democrat
ax.axvspan(3.5,  6.5, alpha=0.04, color="#d6604d")   # Republican

ax.text(1.0, ax.get_ylim()[0] + 0.05 if ax.get_ylim()[0] else 1.02,
        "← Democrat", fontsize=9, color="#2166ac", fontstyle="italic", alpha=0.8)
ax.text(4.5, ax.get_ylim()[0] + 0.05 if ax.get_ylim()[0] else 1.02,
        "Republican →", fontsize=9, color="#d6604d", fontstyle="italic", alpha=0.8)

ax.set_xticks(range(len(ORDERED_LABELS)))
ax.set_xticklabels(ORDERED_LABELS, fontsize=9.5)
ax.set_ylabel("Mean Score (1–5 scale)", fontsize=11)
ax.set_xlim(-0.4, len(ORDERED_LABELS) - 0.6)
ax.set_ylim(1, 5)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax.grid(axis="y", linewidth=0.4, alpha=0.4)
ax.set_title(
    "Figure 7: AP Component Means by Party ID Strength\n"
    "Shaded bands = 95% CI; dashed lines mark partisan boundaries",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=10, framealpha=0.85, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_07_partisan_strength_gradient.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_07_partisan_strength_gradient.png")

# Print cell means table
print("\nCell means by party strength:")
print(f"{'Group':<18} {'Moral':>7} {'Othering':>9} {'Aversion':>9} {'N':>5}")
for val, label in zip(ORDERED_VALS, ORDERED_LABELS):
    mask = np.isclose(df["party_combined"], val)
    row_label = label.replace("\n", " ")
    ns, ms = [], []
    for col in ["ap_moral", "ap_othering", "ap_aversion"]:
        g = df.loc[mask, col].dropna()
        ms.append(g.mean() if len(g) > 0 else np.nan)
        ns.append(len(g))
    n = df.loc[mask, "ap_moral"].dropna().__len__()
    print(f"{row_label:<18} {ms[0]:>7.3f} {ms[1]:>9.3f} {ms[2]:>9.3f} {ns[0]:>5}")
