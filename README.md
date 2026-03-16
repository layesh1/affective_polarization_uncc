# Affective Polarization Study — UNC Charlotte

Analysis of affective polarization among undergraduate students at UNC Charlotte
(Fall 2025 POLS Lab Student Omnibus). Examines emotional hostility, social aversion,
moral othering, and out-party distrust across partisan groups.

> **Data**: Raw survey data are **not included** in this repository per IRB protocol.
> Place the Qualtrics CSV at `data/POLS Lab Student Omnibus Fall 25- text.csv` to run any script.

---

## Key Finding

Democrats in this sample score approximately **2× higher** on the combined affective
polarization index than Republicans (M ≈ 3.8 vs. 1.97 on a 1–5 scale), driven
primarily by the **social aversion** component (Cohen's d > 1.0).

---

## Repository Structure

```
affective_polarization_uncc/
│
├── scripts/                        # All analysis code
│   ├── 00_data_preparation.py      # ← Run first: loads CSV, reverse-codes, saves clean data
│   ├── 01_descriptive_statistics.py
│   ├── 02_core_analysis.py         # t-tests, Cohen's d, correlations
│   ├── 03_supplemental_analyses.py # Race, free speech, partisan sorting
│   │
│   └── figures/                    # Publication figures (run after 00_)
│       ├── fig01_umap_projection.py
│       ├── fig02_pca_biplot.py
│       ├── fig03_violin_distributions.py
│       └── fig04_component_effect_sizes.py
│
├── visualizations/                 # Generated figures (PNG, 300 dpi)
│   ├── figure_01_umap_projection.png
│   ├── figure_02_pca_biplot.png
│   ├── figure_03_violin_distributions.png
│   ├── figure_04_component_effect_sizes.png
│   ├── descriptive_party_distribution.png
│   ├── descriptive_correlation_heatmap.png
│   ├── supp_partisan_sorting.png
│   ├── supp_free_speech_vs_ap.png
│   └── supp_race_party.png
│
├── docs/                           # Written documentation
│   ├── methodology.md              # Step-by-step walkthrough of all statistical decisions
│   ├── statistical_concepts.md     # Explanation of key concepts (UMAP, PCA, Cohen's d, etc.)
│   └── codebook.md                 # Every variable: scale, direction, reverse-coding
│
└── data/                           # Gitignored — place raw CSV here
    └── .gitkeep
```

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn umap-learn
```

### 2. Place data file

```
data/POLS Lab Student Omnibus Fall 25- text.csv
```

### 3. Run in order

```bash
# Step 1: prep & clean
python scripts/00_data_preparation.py

# Step 2: descriptives
python scripts/01_descriptive_statistics.py

# Step 3: core inferential tests
python scripts/02_core_analysis.py

# Step 4: supplemental
python scripts/03_supplemental_analyses.py

# Step 5: publication figures (any order)
python scripts/figures/fig01_umap_projection.py
python scripts/figures/fig02_pca_biplot.py
python scripts/figures/fig03_violin_distributions.py
python scripts/figures/fig04_component_effect_sizes.py
```

---

## Affective Polarization Components

| Component | Items | Scale | Reverse Coding |
|-----------|-------|-------|---------------|
| **Moral Identity** | moral1–3 (R & D versions) | 1–5 | None |
| **Othering** | other1–3 (R & D versions) | 1–5 | None |
| **Social Aversion** | Q135–Q140 | 1–5 | Q137, Q140 reversed (`6 − x`) |
| **Combined AP Index** | Mean of all three | 1–5 | — |
| **FT Gap** | Q148 (Rep warmth), Q149 (Dem warmth) | 0–100 | — |
| **Free Speech** | Q92–Q106 | 1–7 | Q92, Q100, Q103–Q106 reversed (`8 − x`) |
| **Distrust** | Q110–Q122 | 1–7 | Q110, Q112–Q118, Q121 reversed (`8 − x`) |

Full details: see [`docs/codebook.md`](docs/codebook.md)

---

## Publication Figures

| Figure | Description |
|--------|-------------|
| **Fig. 1** | UMAP 2-D scatter — respondents colored by party along Strong Dem → Strong Rep gradient |
| **Fig. 2** | PCA biplot — item loadings on PC1 (general polarization) × PC2 (aversion vs. moral/othering) |
| **Fig. 3** | Violin plot — full AP index distribution by party (Democrat/Independent/Republican) with jittered points |
| **Fig. 4** | Grouped bar chart — component means ± 95% CI for Dems vs. Reps, annotated with Cohen's d |

---

## Documentation

| File | Contents |
|------|----------|
| [`docs/methodology.md`](docs/methodology.md) | Full statistical walkthrough: data loading, scale construction, index formulas, inferential tests |
| [`docs/statistical_concepts.md`](docs/statistical_concepts.md) | Plain-English explanations of affective polarization theory, UMAP, PCA biplots, Cohen's d, violin plots |
| [`docs/codebook.md`](docs/codebook.md) | Every variable with scale, direction, and reverse-coding decisions documented |
