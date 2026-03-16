"""
fig03_violin_distributions.py
==============================
Figure 3: Violin Plot — Affective Polarization Index by Party

Shows the full distribution of the combined affective polarization index for
Democrats (blue), Independents (grey), and Republicans (red).
Individual data points are overlaid as a jittered strip plot.
Means marked with a horizontal bar; medians with a dot.

Output:
    visualizations/figure_03_violin_distributions.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")

PARTY_ORDER  = ["Democrat", "Independent", "Republican"]
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}

INDEX_COL = "affective_polarization_index"

# Keep only respondents with a valid AP index and known party
plot_df = df[df[INDEX_COL].notna() & df["party_3cat"].notna()].copy()
plot_df["party_3cat"] = pd.Categorical(plot_df["party_3cat"], categories=PARTY_ORDER, ordered=True)
plot_df = plot_df.sort_values("party_3cat")


# ─── MANUAL VIOLIN + JITTER USING MATPLOTLIB ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

np.random.seed(42)
positions = {p: i for i, p in enumerate(PARTY_ORDER)}

violin_parts = []

for party in PARTY_ORDER:
    vals = plot_df.loc[plot_df["party_3cat"] == party, INDEX_COL].dropna().values
    if len(vals) < 5:
        continue
    pos = positions[party]
    color = PARTY_COLORS[party]

    # Kernel density estimate
    kde = stats.gaussian_kde(vals, bw_method=0.35)
    y_range = np.linspace(vals.min(), vals.max(), 200)
    density = kde(y_range)
    density_norm = density / density.max() * 0.38   # half-width of violin

    # Draw violin (symmetric)
    ax.fill_betweenx(y_range,
                     pos - density_norm, pos + density_norm,
                     alpha=0.55, color=color, linewidth=0)
    ax.plot(pos - density_norm, y_range, color=color, linewidth=0.7, alpha=0.9)
    ax.plot(pos + density_norm, y_range, color=color, linewidth=0.7, alpha=0.9)

    # IQR box overlay
    q25, q75 = np.percentile(vals, [25, 75])
    median    = np.median(vals)
    mean_val  = vals.mean()
    ax.vlines(pos, q25, q75, color="white", linewidth=3.5, zorder=4)
    ax.scatter([pos], [median], color="white", s=40, zorder=5, marker="o")

    # Mean marker
    ax.scatter([pos], [mean_val], color="black", s=70, zorder=6, marker="D",
               linewidths=1.2, edgecolors="white")

    # Jittered individual points
    jitter_width = 0.12
    jitter = np.random.uniform(-jitter_width, jitter_width, size=len(vals))
    ax.scatter(pos + jitter, vals, color=color, alpha=0.22, s=10,
               linewidths=0, zorder=2)

    # Sample size label
    ax.text(pos, vals.min() - 0.08, f"n={len(vals)}", ha="center",
            va="top", fontsize=9, color=color, fontweight="bold")

# ─── STATISTICAL ANNOTATION ───────────────────────────────────────────────────
dem_vals = plot_df.loc[plot_df["party_3cat"] == "Democrat",  INDEX_COL].dropna()
rep_vals = plot_df.loc[plot_df["party_3cat"] == "Republican", INDEX_COL].dropna()

if len(dem_vals) > 1 and len(rep_vals) > 1:
    t, p = stats.ttest_ind(dem_vals, rep_vals)
    pooled_sd = np.sqrt(((len(dem_vals) - 1) * dem_vals.var(ddof=1) +
                         (len(rep_vals) - 1) * rep_vals.var(ddof=1)) /
                        (len(dem_vals) + len(rep_vals) - 2))
    d = (dem_vals.mean() - rep_vals.mean()) / pooled_sd
    sig_text = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

    # Bracket + annotation
    y_max = max(dem_vals.max(), rep_vals.max()) * 1.05
    y_br  = y_max + 0.05
    x0, x1 = positions["Democrat"], positions["Republican"]
    ax.annotate("", xy=(x1, y_br), xytext=(x0, y_br),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
    ax.vlines([x0, x1], y_max, y_br, color="black", linewidth=1.2)
    ax.text((x0 + x1) / 2, y_br + 0.02,
            f"t = {t:.2f}, p {'< .001' if p < .001 else f'= {p:.3f}'}, d = {d:.2f}",
            ha="center", va="bottom", fontsize=9)


# ─── FORMATTING ───────────────────────────────────────────────────────────────
ax.set_xticks(list(positions.values()))
ax.set_xticklabels(PARTY_ORDER, fontsize=12)
ax.set_ylabel("Affective Polarization Index (1–5)", fontsize=11)
ax.set_title(
    "Figure 3: Distribution of Affective Polarization Index by Party\n"
    "Violin = density; ◆ = mean; ● = median; dots = individual respondents",
    fontsize=12, fontweight="bold"
)

# Legend
legend_patches = [mpatches.Patch(color=c, label=p) for p, c in PARTY_COLORS.items()]
legend_patches += [plt.scatter([], [], marker="D", color="black", s=50, label="Mean")]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.85)

ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.6, len(PARTY_ORDER) - 0.4)

plt.tight_layout()
plt.savefig("visualizations/figure_03_violin_distributions.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: visualizations/figure_03_violin_distributions.png")

# Print summary stats
print("\nAffective Polarization Index by Party:")
for party in PARTY_ORDER:
    g = plot_df.loc[plot_df["party_3cat"] == party, INDEX_COL].dropna()
    if len(g) > 0:
        print(f"  {party:<15}  M={g.mean():.3f}  SD={g.std():.3f}  N={len(g)}")
