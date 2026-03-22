"""
fig11_regression_coef_plot.py
==============================
Figure 11: Regression Coefficient Plot — Affective Polarization Predicting
Free Speech Restriction

Shows OLS coefficients (+ 95% CI) for M1 (combined model) and the two
within-party models (M4-Dem, M4-Rep), plotted side by side as a forest plot.

Interpretation guide baked into the figure:
  - Dot to the RIGHT of zero → predictor increases restriction
  - Dot to the LEFT  of zero → predictor decreases restriction (more pro-speech)
  - CI crossing zero → not significant

Output: visualizations/figure_11_regression_coef_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
formula_full = ("free_speech_restriction_index ~ "
                "ap_othering + ap_moral + ap_aversion + party_binary")
formula_sub  = "free_speech_restriction_index ~ ap_othering + ap_moral + ap_aversion"

m_full = smf.ols(formula_full, data=partisan).fit()
m_dem  = smf.ols(formula_sub,  data=dems).fit()
m_rep  = smf.ols(formula_sub,  data=reps).fit()

# ─── Extract coefficients (skip Intercept) ────────────────────────────────────
VAR_LABELS = {
    "ap_othering":   "Othering",
    "ap_moral":      "Moralizing",
    "ap_aversion":   "Social Aversion",
    "party_binary":  "Party (Rep = 1)",
}
PLOT_VARS = ["ap_othering", "ap_moral", "ap_aversion", "party_binary"]

models = {
    "M1 — Full model\n(Dems + Reps,\nw/ party control)": (m_full, "#555555"),
    "M4-Dem\n(Democrats only)":                           (m_dem,  "#2166ac"),
    "M4-Rep\n(Republicans only)":                         (m_rep,  "#d6604d"),
}

# Build dataframe of [model, variable, B, lo, hi, p]
rows = []
for model_label, (m, color) in models.items():
    ci = m.conf_int()
    for v in PLOT_VARS:
        if v not in m.params.index:
            continue
        rows.append({
            "model": model_label, "color": color,
            "variable": v, "label": VAR_LABELS[v],
            "B": m.params[v], "lo": ci.loc[v, 0], "hi": ci.loc[v, 1],
            "p": m.pvalues[v],
        })
coef_df = pd.DataFrame(rows)

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(PLOT_VARS), figsize=(14, 5), sharey=False)
fig.patch.set_facecolor("white")

model_labels = list(models.keys())
n_models = len(model_labels)
y_positions = np.arange(n_models)
spacing = 0.28

for ax_idx, var in enumerate(PLOT_VARS):
    ax = axes[ax_idx]
    sub = coef_df[coef_df["variable"] == var].reset_index(drop=True)

    for i, row in sub.iterrows():
        y = y_positions[i]
        sig = row["p"] < 0.05
        marker = "D" if sig else "o"
        ms = 8 if sig else 7
        ax.plot([row["lo"], row["hi"]], [y, y],
                color=row["color"], lw=2.0, solid_capstyle="round")
        ax.plot(row["B"], y, marker=marker, ms=ms,
                color=row["color"], zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)

        # Annotate B value
        offset = 0.06 if row["B"] >= 0 else -0.06
        ha = "left" if row["B"] >= 0 else "right"
        sig_str = "*" if row["p"] < .05 else ""
        ax.text(row["B"] + offset, y, f"{row['B']:.3f}{sig_str}",
                va="center", ha=ha, fontsize=8.5, color=row["color"])

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [ml.split("\n")[0] for ml in model_labels] if ax_idx == 0 else [""] * n_models,
        fontsize=9
    )
    ax.set_title(VAR_LABELS[var], fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Coefficient (B)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.6, n_models - 0.4)

    # Shade region around zero
    ax_xlim = ax.get_xlim()

# Add model color legend
legend_patches = [
    mpatches.Patch(color=color, label=label.replace("\n", " "))
    for label, (_, color) in models.items()
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "Figure 11: Regression Coefficients — Affective Polarization Predicting Free Speech Restriction\n"
    "Diamonds (◆) = p < .05 · Circles (●) = n.s. · Error bars = 95% CI · "
    "Right of zero = more restriction",
    fontsize=11, fontweight="bold", y=1.01
)

note = (
    "DV: free_speech_restriction_index (1–7, higher = more pro-restriction).\n"
    "M1 controls for party; M4-Dem and M4-Rep are within-party models.\n"
    "* p < .05,  ** p < .01,  *** p < .001"
)
fig.text(0.5, -0.09, note, ha="center", fontsize=8.5, color="#444444",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="grey", alpha=0.9))

plt.tight_layout(rect=[0, 0.0, 1, 1])
plt.savefig("visualizations/figure_11_regression_coef_plot.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_11_regression_coef_plot.png")
