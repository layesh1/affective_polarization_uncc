"""
fig12_within_party_scatter.py
==============================
Figure 12: Within-Party Scatter — Each Polarization Component vs. Free Speech
Restriction, Separately for Democrats and Republicans.

3 columns (Othering, Moralizing, Social Aversion) × 2 rows (Dems, Reps) = 6 panels.
Each panel shows:
  - Jittered scatter of individual students
  - OLS regression line with 95% bootstrap CI band
  - Pearson r and p-value annotation

Key insight: The slope and significance of the polarization→speech link differs
by component AND by party — particularly Aversion for Republicans.

Output: visualizations/figure_12_within_party_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")
partisan = df[df["party_3cat"].isin(["Democrat", "Republican"])].copy()

COMPONENTS = [
    ("ap_othering",  "Othering"),
    ("ap_moral",     "Moralizing"),
    ("ap_aversion",  "Social Aversion"),
]
PARTIES = [
    ("Democrat",   "#2166ac", 0),
    ("Republican", "#d6604d", 1),
]
DV = "free_speech_restriction_index"

np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
fig.patch.set_facecolor("white")

for col_idx, (iv, iv_label) in enumerate(COMPONENTS):
    for row_idx, (party, color, _) in enumerate(PARTIES):
        ax = axes[row_idx][col_idx]
        sub = partisan[partisan["party_3cat"] == party][[iv, DV]].dropna()

        x = sub[iv].values
        y = sub[DV].values

        # Jitter
        jx = x + np.random.uniform(-0.08, 0.08, len(x))
        jy = y + np.random.uniform(-0.08, 0.08, len(y))

        ax.scatter(jx, jy, alpha=0.22, s=14, color=color, linewidths=0)

        # Regression line
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = slope * xs + intercept

        # Bootstrap CI
        boot = []
        n = len(x)
        for _ in range(600):
            idx = np.random.choice(n, n, replace=True)
            s2, i2, *_ = stats.linregress(x[idx], y[idx])
            boot.append(s2 * xs + i2)
        boot = np.array(boot)
        ax.fill_between(xs, np.percentile(boot, 2.5, 0),
                        np.percentile(boot, 97.5, 0),
                        color=color, alpha=0.15)
        ax.plot(xs, ys, color=color, lw=2.2)

        # Annotation
        p_str = "< .001" if p < .001 else f"= {p:.3f}"
        sig_str = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "(n.s.)"
        ann = f"r = {r:.2f} {sig_str}\np {p_str}\nN = {len(x)}"
        ax.text(0.04, 0.96, ann, transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85, ec="grey"))

        # Mean lines
        ax.axhline(y.mean(), color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.axvline(x.mean(), color="grey", lw=0.8, ls=":", alpha=0.6)

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 7.2)

        # Labels
        if row_idx == 1:
            ax.set_xlabel(f"{iv_label}\n(1 = low  →  5 = high)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(
                f"{party}s\n\nSpeech Restriction\n(1 = pro-speech  →  7 = pro-restriction)",
                fontsize=9.5, color=color, fontweight="bold"
            )
        if row_idx == 0:
            ax.set_title(iv_label, fontsize=12, fontweight="bold", pad=8)

fig.suptitle(
    "Figure 12: Does Affective Polarization Predict Speech Restriction? Within-Party Analysis\n"
    "Each dot = one student. Shading = 95% bootstrap CI. Dotted lines = group means.",
    fontsize=12, fontweight="bold", y=1.01
)

note = (
    "Key finding: Social Aversion is the only component that significantly predicts speech restriction — "
    "and only among Republicans (r = {:.2f}, p < .001).\n"
    "Among Democrats, moralizing shows a small positive association with free speech support "
    "(lower restriction). Othering is non-significant in both parties."
)

# Compute the Rep aversion r for the note
rep_sub = partisan[partisan["party_3cat"] == "Republican"][["ap_aversion", DV]].dropna()
r_rep_av, _ = stats.pearsonr(rep_sub["ap_aversion"], rep_sub[DV])
fig.text(0.5, -0.04, note.format(r_rep_av), ha="center", fontsize=9, color="#333333",
         bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="grey", alpha=0.9))

plt.tight_layout(rect=[0, 0.0, 1, 1])
plt.savefig("visualizations/figure_12_within_party_scatter.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_12_within_party_scatter.png")
