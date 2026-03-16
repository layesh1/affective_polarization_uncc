"""
fig02_pca_biplot.py
====================
Figure 2: PCA Biplot — Affective Polarization Components

Runs PCA on the 9 affective polarization items (own-party version per
respondent).  Plots respondent scores on PC1 × PC2, overlaid with item
loading vectors.

Expected pattern:
  PC1 — general polarization: all 9 items load positively
  PC2 — component differentiation: aversion items vs. moral/othering items
         load in opposite directions

Output:
    visualizations/figure_02_pca_biplot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/polarization_clean.csv")

is_dem = df["party_3cat"] == "Democrat"
is_rep = df["party_3cat"] == "Republican"

ITEM_LABELS = [
    "Moral 1", "Moral 2", "Moral 3",
    "Othering 1", "Othering 2", "Othering 3",
    "Aversion 1", "Aversion 2", "Aversion 3",
]

ITEM_CLUSTERS = {
    "Moral Identity": [0, 1, 2],
    "Othering":       [3, 4, 5],
    "Social Aversion":[6, 7, 8],
}

CLUSTER_COLORS = {
    "Moral Identity":  "#1a9850",
    "Othering":        "#d73027",
    "Social Aversion": "#f46d43",
}

features = pd.DataFrame({
    "moral_1":    np.where(is_dem, df.get("moral1D"), np.where(is_rep, df.get("moral1R"), np.nan)),
    "moral_2":    np.where(is_dem, df.get("moral2D"), np.where(is_rep, df.get("moral2R"), np.nan)),
    "moral_3":    np.where(is_dem, df.get("moral3D"), np.where(is_rep, df.get("moral3R"), np.nan)),
    "other_1":    np.where(is_dem, df.get("other1D"), np.where(is_rep, df.get("other1R"), np.nan)),
    "other_2":    np.where(is_dem, df.get("other2D"), np.where(is_rep, df.get("other2R"), np.nan)),
    "other_3":    np.where(is_dem, df.get("other3D"), np.where(is_rep, df.get("other3R"), np.nan)),
    "aversion_1": np.where(is_dem, df.get("Q138_s"), np.where(is_rep, df.get("Q135_s"), np.nan)),
    "aversion_2": np.where(is_dem, df.get("Q139_s"), np.where(is_rep, df.get("Q136_s"), np.nan)),
    "aversion_3": np.where(is_dem, df.get("Q140_s"), np.where(is_rep, df.get("Q137_s"), np.nan)),
})

mask = features.notna().all(axis=1) & df["party_combined"].notna()
X = features[mask].values
party_cats = df.loc[mask, "party_3cat"].values

# ─── PCA ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)
loadings = pca.components_.T          # shape (9, 2)

pct_var = pca.explained_variance_ratio_ * 100

# ─── BIPLOT ───────────────────────────────────────────────────────────────────
PARTY_COLORS = {"Democrat": "#2166ac", "Independent": "#969696", "Republican": "#d6604d"}
point_colors = [PARTY_COLORS.get(p, "#969696") for p in party_cats]

# Scale factor for loading arrows
scale = max(np.abs(scores[:, 0]).max(), np.abs(scores[:, 1]).max()) * 0.42

fig, ax = plt.subplots(figsize=(8.5, 7))

# ── Scatter: respondent scores ──
for party, color in PARTY_COLORS.items():
    idx = party_cats == party
    ax.scatter(scores[idx, 0], scores[idx, 1],
               c=color, alpha=0.35, s=14, linewidths=0, label=party, zorder=2)

# ── Loading arrows ──
for i, (label, sx, sy) in enumerate(zip(ITEM_LABELS, loadings[:, 0], loadings[:, 1])):
    # Determine cluster color for this item
    arrow_color = "#333333"
    for cluster, indices in ITEM_CLUSTERS.items():
        if i in indices:
            arrow_color = CLUSTER_COLORS[cluster]
            break

    ax.annotate(
        "", xy=(sx * scale, sy * scale), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.8),
        zorder=4,
    )
    # Offset label slightly beyond arrow tip
    nudge = 1.18
    ax.text(sx * scale * nudge, sy * scale * nudge, label,
            fontsize=8.5, color=arrow_color, fontweight="bold",
            ha="center", va="center", zorder=5)

# ── Reference lines ──
ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

# ── Cluster legend patches ──
cluster_patches = [mpatches.Patch(color=c, label=n) for n, c in CLUSTER_COLORS.items()]
party_patches   = [mpatches.Patch(color=c, label=p) for p, c in PARTY_COLORS.items()]

leg1 = ax.legend(handles=party_patches, title="Respondent Party",
                 loc="upper right", fontsize=8.5, framealpha=0.85)
ax.add_artist(leg1)
ax.legend(handles=cluster_patches, title="Item Cluster",
          loc="lower right", fontsize=8.5, framealpha=0.85)

ax.set_xlabel(f"PC1 — General Polarization ({pct_var[0]:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 — Aversion vs. Moral/Othering ({pct_var[1]:.1f}% variance)", fontsize=11)
ax.set_title("Figure 2: PCA Biplot — Affective Polarization Items\n"
             "(arrows = item loadings; colored by cluster)", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("visualizations/figure_02_pca_biplot.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: visualizations/figure_02_pca_biplot.png")
print(f"PC1 explains {pct_var[0]:.1f}% of variance")
print(f"PC2 explains {pct_var[1]:.1f}% of variance")
print(f"Total (PC1+PC2): {sum(pct_var):.1f}%")
print("\nItem loadings (PC1, PC2):")
for label, lx, ly in zip(ITEM_LABELS, loadings[:, 0], loadings[:, 1]):
    print(f"  {label:<15}  PC1={lx:+.3f}  PC2={ly:+.3f}")
