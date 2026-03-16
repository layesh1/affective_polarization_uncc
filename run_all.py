"""
run_all.py
==========
Run every analysis and generate all figures in the correct order.

Usage:
    python3 run_all.py
"""

import subprocess
import sys

SCRIPTS = [
    ("Data preparation",         "scripts/00_data_preparation.py"),
    ("Descriptive statistics",   "scripts/01_descriptive_statistics.py"),
    ("Core analysis",            "scripts/02_core_analysis.py"),
    ("Supplemental analyses",    "scripts/03_supplemental_analyses.py"),
    ("Figure 1 — UMAP",          "scripts/figures/fig01_umap_projection.py"),
    ("Figure 2 — PCA biplot",    "scripts/figures/fig02_pca_biplot.py"),
    ("Figure 3 — Violin plot",   "scripts/figures/fig03_violin_distributions.py"),
    ("Figure 4 — Effect sizes",  "scripts/figures/fig04_component_effect_sizes.py"),
]

print("=" * 60)
print("Running all scripts")
print("=" * 60)

for label, path in SCRIPTS:
    print(f"\n── {label} ──")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR in {path} — stopping.")
        sys.exit(1)

print("\n" + "=" * 60)
print("Done. Figures saved to: visualizations/")
print("=" * 60)
