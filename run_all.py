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
    ("Data preparation",                    "scripts/00_data_preparation.py"),
    ("Descriptive statistics",              "scripts/01_descriptive_statistics.py"),
    ("Core analysis",                       "scripts/02_core_analysis.py"),
    ("Supplemental analyses",               "scripts/03_supplemental_analyses.py"),
    ("Figure 1  — UMAP projection",         "scripts/figures/fig01_umap_projection.py"),
    ("Figure 2  — PCA biplot",              "scripts/figures/fig02_pca_biplot.py"),
    ("Figure 3  — Violin: AP index",        "scripts/figures/fig03_violin_distributions.py"),
    ("Figure 4  — Component effect sizes",  "scripts/figures/fig04_component_effect_sizes.py"),
    ("Figure 5  — Free speech by battery",  "scripts/figures/fig05_free_speech_by_battery.py"),
    ("Figure 6  — Aversion vs. speech",     "scripts/figures/fig06_aversion_vs_speech.py"),
    ("Figure 7  — Partisan strength gradient", "scripts/figures/fig07_partisan_strength_gradient.py"),
    ("Figure 8  — Ideology distribution",   "scripts/figures/fig08_ideology_distribution.py"),
    ("Figure 9  — Distrust violin",         "scripts/figures/fig09_distrust_violin.py"),
    ("Figure 10 — FT vs AP validation",     "scripts/figures/fig10_ft_vs_ap_validation.py"),
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
print("Done. All figures saved to: visualizations/")
print("=" * 60)
