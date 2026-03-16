"""
fig04_component_effect_sizes.py
================================
Figure 4: Component-Level Effect Sizes — Grouped Bar Chart

Shows Democrat and Republican means (± 95% CI) for each of the three
affective polarization components: Moral Identity, Othering, and Social
Aversion.  Cohen's d effect sizes are annotated above each pair.

Output:
    visualizations/figure_04_component_effect_sizes.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")

DEM_COLOR = "#2166ac"
REP_COLOR = "#d6604d"

COMPONENTS = {
    "Moral\nIdentity":   "ap_moral",
    "Othering":          "ap_othering",
    "Social\nAversion":  "ap_aversion",
}

dems = df[df["party_3cat"] == "Democrat"]
reps = df[df["party_3cat"] == "Republican"]


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled_sd = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled_sd

def ci95(vals):
    """Return 95% CI half-width using t-distribution."""
    n = len(vals)
    se = vals.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return se * t_crit


# ─── COMPUTE STATS ────────────────────────────────────────────────────────────
component_stats = {}
for label, col in COMPONENTS.items():
    if col not in df.columns:
        continue
    d_vals = dems[col].dropna()
    r_vals = reps[col].dropna()
    t, p   = stats.ttest_ind(d_vals, r_vals)
    d_val  = cohens_d(d_vals, r_vals)
    component_stats[label] = {
        "dem_mean": d_vals.mean(),
        "dem_ci":   ci95(d_vals),
        "rep_mean": r_vals.mean(),
        "rep_ci":   ci95(r_vals),
        "d":        d_val,
        "p":        p,
        "t":        t,
    }


# ─── PLOT ─────────────────────────────────────────────────────────────────────
labels   = list(component_stats.keys())
n_groups = len(labels)
x        = np.arange(n_groups)
bar_w    = 0.32

fig, ax = plt.subplots(figsize=(8, 5.5))

# Bars
dem_bars = ax.bar(x - bar_w / 2,
                  [component_stats[l]["dem_mean"] for l in labels],
                  bar_w, label="Democrats",
                  color=DEM_COLOR, alpha=0.88, edgecolor="white", linewidth=0.8)
rep_bars = ax.bar(x + bar_w / 2,
                  [component_stats[l]["rep_mean"] for l in labels],
                  bar_w, label="Republicans",
                  color=REP_COLOR, alpha=0.88, edgecolor="white", linewidth=0.8)

# Error bars (95% CI)
ax.errorbar(x - bar_w / 2,
            [component_stats[l]["dem_mean"] for l in labels],
            yerr=[component_stats[l]["dem_ci"]  for l in labels],
            fmt="none", color="black", capsize=4, linewidth=1.3, capthick=1.3)
ax.errorbar(x + bar_w / 2,
            [component_stats[l]["rep_mean"] for l in labels],
            yerr=[component_stats[l]["rep_ci"]  for l in labels],
            fmt="none", color="black", capsize=4, linewidth=1.3, capthick=1.3)

# ── Cohen's d annotation above each pair ──
for i, label in enumerate(labels):
    s = component_stats[label]
    y_top = max(s["dem_mean"] + s["dem_ci"], s["rep_mean"] + s["rep_ci"]) + 0.08
    d_str = f"d = {s['d']:.2f}"
    sig   = "***" if s["p"] < .001 else "**" if s["p"] < .01 else "*" if s["p"] < .05 else "ns"
    # Significance bracket
    ax.plot([i - bar_w / 2, i + bar_w / 2], [y_top, y_top], color="black", linewidth=1.1)
    ax.text(i, y_top + 0.025, f"{d_str}  {sig}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

# ─── FORMATTING ───────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Mean Score (1–5 scale)", fontsize=11)
ax.set_title(
    "Figure 4: Component-Level Affective Polarization by Party\n"
    "Error bars = 95% CI; d = Cohen's d (positive = Dems > Reps)",
    fontsize=12, fontweight="bold",
)
ax.set_ylim(1, ax.get_ylim()[1] + 0.25)

legend_patches = [
    mpatches.Patch(color=DEM_COLOR, alpha=0.88, label="Democrats"),
    mpatches.Patch(color=REP_COLOR, alpha=0.88, label="Republicans"),
]
ax.legend(handles=legend_patches, fontsize=10, framealpha=0.85, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# Reference line at scale midpoint
ax.axhline(3, color="grey", linewidth=0.7, linestyle=":", alpha=0.6,
           label="Scale midpoint (3)")

plt.tight_layout()
plt.savefig("visualizations/figure_04_component_effect_sizes.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_04_component_effect_sizes.png")

# Print full stats table
print("\nComponent-Level Effect Sizes:")
print(f"{'Component':<20} {'Dem M':>7} {'Dem 95%CI':>10} {'Rep M':>7} {'Rep 95%CI':>10} "
      f"{'t':>7} {'p':>8} {'d':>7}")
for label, s in component_stats.items():
    clean = label.replace("\n", " ")
    print(f"{clean:<20} {s['dem_mean']:>7.3f} {'±'+str(round(s['dem_ci'],3)):>10} "
          f"{s['rep_mean']:>7.3f} {'±'+str(round(s['rep_ci'],3)):>10} "
          f"{s['t']:>7.3f} {s['p']:>8.4f} {s['d']:>7.3f}")
