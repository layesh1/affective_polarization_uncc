"""
fig11_regression_coef_plot.py
==============================
Figure 11: Does Political Hostility Predict Who Wants to Restrict Speech?
           Regression Results by Component and Party Group

Shows OLS regression coefficients + 95% CIs for three models:
  - Full model (all students, controlling for party)
  - Democrats only
  - Republicans only

Plain-language reading guide is built into the figure.

Output: visualizations/figure_11_regression_coef_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ─── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")
partisan = df[df["party_3cat"].isin(["Democrat", "Republican"])].copy()
partisan["party_binary"] = (partisan["party_3cat"] == "Republican").astype(float)
dems = partisan[partisan["party_binary"] == 0].copy()
reps = partisan[partisan["party_binary"] == 1].copy()

# ─── Fit models ───────────────────────────────────────────────────────────────
m_full = smf.ols(
    "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion + party_binary",
    data=partisan).fit()
m_dem = smf.ols(
    "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion",
    data=dems).fit()
m_rep = smf.ols(
    "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion",
    data=reps).fit()

# ─── Setup ────────────────────────────────────────────────────────────────────
PLOT_VARS = ["ap_othering", "ap_moral", "ap_aversion"]
VAR_LABELS = {
    "ap_othering": "Othering\n(Sees out-party\nas alien)",
    "ap_moral":    "Moralizing\n(Sees party\nas moral cause)",
    "ap_aversion": "Social Aversion\n(Avoids out-party\nmembers)",
}
MODELS = [
    ("All students\n(controlling for party)", m_full, "#555555"),
    ("Democrats only",                         m_dem,  "#2166ac"),
    ("Republicans only",                        m_rep,  "#d6604d"),
]

ci = {label: m.conf_int() for label, m, _ in MODELS}

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=False)
fig.patch.set_facecolor("white")

y_pos     = np.arange(len(MODELS))
y_labels  = [label for label, _, _ in MODELS]

for col_idx, var in enumerate(PLOT_VARS):
    ax = axes[col_idx]

    for row_idx, (label, m, color) in enumerate(MODELS):
        if var not in m.params.index:
            continue
        b  = m.params[var]
        lo = ci[label].loc[var, 0]
        hi = ci[label].loc[var, 1]
        p  = m.pvalues[var]
        sig = p < 0.05

        # CI line
        ax.plot([lo, hi], [row_idx, row_idx], color=color, lw=2.5,
                solid_capstyle="round", alpha=0.85)
        # Point
        marker = "D" if sig else "o"
        ms     = 9 if sig else 8
        ax.plot(b, row_idx, marker=marker, ms=ms, color=color, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)

        # B label
        sig_str = "*" if p < .05 else ""
        if p < .001: sig_str = "***"
        elif p < .01: sig_str = "**"
        offset = max(abs(hi - lo) * 0.12, 0.04)
        ha = "left" if b >= 0 else "right"
        ax.text(b + (offset if b >= 0 else -offset), row_idx,
                f"B = {b:+.3f}{sig_str}",
                va="center", ha=ha, fontsize=8.5, color=color, fontweight="bold")

    # Zero line (no effect)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.55, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        y_labels if col_idx == 0 else [""] * len(MODELS),
        fontsize=9.5
    )
    ax.set_title(VAR_LABELS[var], fontsize=11, fontweight="bold", pad=8,
                 linespacing=1.4)
    ax.set_xlabel("Effect on speech restriction\n(+ = predicts more restriction  |  − = predicts less restriction)",
                  fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.7, len(MODELS) - 0.3)

    # Shade the "no effect" region
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)

# ─── Legend ───────────────────────────────────────────────────────────────────
sig_marker   = mlines.Line2D([], [], marker="D", color="black", ms=8,
                              linestyle="None", label="Statistically significant (p < .05)")
insig_marker = mlines.Line2D([], [], marker="o", color="black", ms=8,
                              linestyle="None", label="Not significant (p ≥ .05)")
fig.legend(handles=[sig_marker, insig_marker], loc="lower center", ncol=2,
           fontsize=9.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

# ─── Title ────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 11: Does Partisan Hostility Predict Support for Speech Restrictions?\n"
    "OLS Regression Coefficients (B) ± 95% Confidence Intervals for Each Hostility Component",
    fontsize=12, fontweight="bold", y=1.02
)

# ─── How-to-read box ──────────────────────────────────────────────────────────
note = (
    "How to read:  Each panel shows one type of partisan hostility.\n"
    "Each row = one group of students.  The dot is the estimated effect;\n"
    "the line is the range of plausible values (95% confidence interval).\n"
    "◆ filled diamond = statistically significant  |  ● circle = not significant\n"
    "If the line crosses the dashed zero line → no reliable effect.\n"
    "DV: Free speech restriction scale (1–7); higher = more pro-restriction."
)
fig.text(0.5, -0.12, note, ha="center", fontsize=8.5, color="#333333",
         bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="grey", alpha=0.9))

plt.tight_layout(rect=[0, 0.0, 1, 1.0])
plt.savefig("visualizations/figure_11_regression_coef_plot.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_11_regression_coef_plot.png")
