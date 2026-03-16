"""
fig07_partisan_strength_gradient.py
=====================================
Figure 7: Do Stronger Partisans Show More Hostility Toward the Other Party?

X-axis: How strongly a student identifies with their party (7 categories,
        Strong Democrat on the left to Strong Republican on the right)
Y-axis: Mean score on each AP component (1–5)
Lines:  Moral Identity, Othering, Social Aversion

Output: visualizations/figure_07_partisan_strength_gradient.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("data/polarization_clean.csv")

# Ordered from most Democratic (1) to most Republican (5)
# 1.5 = Lean Dem, 4.5 = Lean Rep
ORDERED_VALS   = [1, 1.5, 2, 4, 4.5, 5]
ORDERED_LABELS = [
    "Strong\nDemocrat", "Lean\nDemocrat", "Somewhat\nDemocrat",
    "Somewhat\nRepublican", "Lean\nRepublican", "Strong\nRepublican",
]
# Note: "True Independent" (party_combined==3) excluded — too few respondents

COMPONENTS = {
    "Moral Identity\n(party = moral cause)":  ("ap_moral",    "#1a9850", "o", "--"),
    "Othering\n(out-party = alien)":           ("ap_othering", "#756bb1", "s", "-."),
    "Social Aversion\n(avoids out-party)":     ("ap_aversion", "#e6550d", "^", "-"),
}

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")

for label, (col, color, marker, ls) in COMPONENTS.items():
    x_pos, y_means, y_ci = [], [], []
    for i, val in enumerate(ORDERED_VALS):
        g = df.loc[np.isclose(df["party_combined"], val), col].dropna()
        if len(g) < 3:
            continue
        m  = g.mean()
        ci = stats.t.ppf(0.975, df=len(g)-1) * g.std(ddof=1) / np.sqrt(len(g))
        x_pos.append(i); y_means.append(m); y_ci.append(ci)

    x_arr = np.array(x_pos); y_arr = np.array(y_means); ci_arr = np.array(y_ci)
    ax.plot(x_arr, y_arr, color=color, lw=2.3, marker=marker,
            markersize=9, label=label, linestyle=ls, zorder=4)
    ax.fill_between(x_arr, y_arr - ci_arr, y_arr + ci_arr,
                    color=color, alpha=0.13, zorder=2)

# Shaded regions
ax.axvspan(-0.4, 2.4,  alpha=0.05, color="#2166ac", zorder=1)
ax.axvspan(2.6, 5.4,   alpha=0.05, color="#d6604d",  zorder=1)
ax.axvline(2.5, color="grey", lw=0.9, ls="--", alpha=0.45)

# Region labels
ax.text(1.0, 4.75, "← Democrats", fontsize=10.5, color="#2166ac",
        fontstyle="italic", fontweight="bold", ha="center", alpha=0.8)
ax.text(3.85, 4.75, "Republicans →", fontsize=10.5, color="#d6604d",
        fontstyle="italic", fontweight="bold", ha="center", alpha=0.8)

ax.set_xticks(range(len(ORDERED_LABELS)))
ax.set_xticklabels(ORDERED_LABELS, fontsize=10.5)
ax.set_ylabel("Average Score  (1 = Low  →  5 = High)", fontsize=11)
ax.set_ylim(1, 5.2)
ax.set_xlim(-0.4, len(ORDERED_LABELS) - 0.6)
ax.grid(axis="y", lw=0.4, alpha=0.35)

ax.set_title("Figure 7: Do Stronger Partisans Show More Hostility Toward the Other Party?\n"
             "Each point = average score for students at that level of party identification",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=10, framealpha=0.9, loc="upper right")

note = (
    "How to read: Each line traces one type of political hostility.\n"
    "Higher = more hostility. Strong Democrats (left) vs Strong Republicans (right).\n"
    "Shaded bands = 95% confidence intervals around each group's average."
)
ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f0f0f0", alpha=0.9, ec="grey"))

ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("visualizations/figure_07_partisan_strength_gradient.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_07_partisan_strength_gradient.png")
