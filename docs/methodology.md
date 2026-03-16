# Statistical Methodology Walkthrough

## Study Overview
This project examines **affective polarization** — the psychological distance between partisan groups defined not by policy disagreement but by negative emotions, social aversion, and moral othering — among undergraduate students at UNC Charlotte (Fall 2025 POLS Lab Student Omnibus).

Raw data are excluded from this repository per IRB protocol. Only analysis scripts and aggregate visualizations are shared.

---

## Data Preparation (`scripts/00_data_preparation.py`)

### Source File
`POLS Lab Student Omnibus Fall 25- text.csv`

The file is a Qualtrics export with a text-label format (`-text`). Qualtrics includes a second row of question text; this row is skipped with `skiprows=[1]`.

### Party Identification
Respondents are coded on a **5-point scale**:
| Value | Label |
|-------|-------|
| 1 | Strongly Democrat |
| 2 | Somewhat Democrat |
| 3 | Not sure / neither / other |
| 4 | Somewhat Republican |
| 5 | Strongly Republican |

True independents (value = 3) who indicate a partisan lean are recoded:
- Lean Democrat → **1.5**
- Lean Republican → **4.5**

For most analyses respondents are grouped into three categories: **Democrat**, **Independent**, **Republican** (cut at 2.5 and 3.5 on the combined scale).

---

## Scale Construction

### 1. Moral Identity (1–5, higher = stronger moral-partisan link)

Items ask how much the respondent's *own* party identity is tied to core moral beliefs.

| Item | Wording |
|------|---------|
| moral1 | "My party identity is connected to my core moral beliefs." |
| moral2 | "My party reflects my beliefs about right and wrong." |
| moral3 | "My party identity is rooted in my moral principles." |

**Reverse coding:** None required — all items are scored in the same direction (higher = more moral-identity fusion).

**Index:** Simple mean of all three items after mapping text responses to 1–5 via `{"None at all": 1, "A little": 2, "A moderate amount": 3, "A lot": 4, "A great deal": 5}`.

Republicans and Democrats each received parallel versions (moral1R/moral1D, etc.) because the survey presented them as separate question blocks.

---

### 2. Othering (1–5, higher = more dehumanization of out-party)

Items ask how different / alien the respondent perceives the *opposing* party to be.

| Item | Wording |
|------|---------|
| other1 | "The opposing party is very different from people like me." |
| other2 | "People in the opposing party live in a completely different world." |
| other3 | "I cannot understand why people in the opposing party act the way they do." |

**Reverse coding:** None required.

**Index:** Simple mean of the three items.

---

### 3. Social Aversion (1–5, higher = stronger avoidance of out-party members)

Six items split by party; **two are reverse-coded** because they measure liking rather than aversion.

| Column | Party | Wording | Direction |
|--------|-------|---------|-----------|
| Q135 | Republican (R aversion toward D) | "I would not want to be friends with a Democrat." | **Direct** |
| Q136 | Republican | "I would want to stop spending time with a Democratic friend." | **Direct** |
| Q137 | Republican | "There are people I like who are Democrats." | **REVERSED** (liking → aversion) |
| Q138 | Democrat (D aversion toward R) | "I would not want to be friends with a Republican." | **Direct** |
| Q139 | Democrat | "I would want to stop spending time with a Republican friend." | **Direct** |
| Q140 | Democrat | "There are people I like who are Republicans." | **REVERSED** |

**Reverse coding formula:** `reversed = 6 − original`  (maps 1↔5, 2↔4, 3↔3 on a 1–5 scale)

**Why reverse Q137/Q140?** Agreeing with "There are people I like who are [out-party]" signals *low* social aversion. To include it in a coherent aversion index (where high = more aversion), we flip its direction so that high aversion to out-party members scores consistently high across all items.

**Index:** Mean of the three party-appropriate scaled items.

---

### 4. Combined Affective Polarization Index (1–5)

For each respondent, the composite is the mean of their own-party moral index, their othering index, and their aversion index:

```
AP_index = mean(moral_own, othering_own, aversion_own)
```

Republicans use the R-version items; Democrats use the D-version items; Independents receive NaN and are excluded from partisan comparisons.

---

### 5. Feeling Thermometer Gap (0–100)

- **Q148**: Warmth toward Republicans (0 = very cold, 100 = very warm)
- **Q149**: Warmth toward Democrats

FT gap = inparty warmth − outparty warmth. Higher gap = greater affective polarization via the classic thermometer method.

---

### 6. Free Speech Support Index (1–7, higher = greater support for free speech)

Thirteen items on a 7-point agreement scale (1 = Strongly Agree, 7 = Strongly Disagree).

**Pro-freedom questions** (high agreement = support free speech → **REVERSED** so that high score = support):
`Q92, Q100, Q103, Q104, Q105, Q106`
`reversed = 8 − original`

**Pro-restriction questions** (high agreement = oppose restriction = support free speech → **kept as-is**):
`Q95, Q96, Q97, Q98, Q99, Q101, Q102`

Index = mean of all 13 scaled items.

---

### 7. Out-Party Distrust Index (1–7, higher = greater distrust)

**Pro-trust questions** (high agreement = trust → **REVERSED** to distrust direction):
`Q110, Q112, Q113, Q114, Q115, Q116, Q117, Q118, Q121`
`reversed = 8 − original`

**Pro-distrust questions** (high agreement = distrust → **kept as-is**):
`Q119, Q120, Q122`

Index = mean of all scaled items.

---

## Reliability

Cronbach's alpha (α) is computed for each scale to verify internal consistency:
- α ≥ .80 = excellent
- α ≥ .70 = acceptable
- α < .60 = questionable

All three AP components (moral identity, othering, aversion) are expected to exceed α = .70 based on prior literature.

---

## Inferential Tests (`scripts/02_core_analysis.py`)

### Independent-Samples t-test
Compares Democrat vs. Republican means on each index. Assumes approximately equal variances (Levene's test should be inspected if heteroscedasticity is suspected).

### Cohen's d (Effect Size)
```
d = (M_Dem − M_Rep) / SD_pooled
```
Where `SD_pooled = sqrt([(n1−1)·var1 + (n2−1)·var2] / (n1+n2−2))`.

Benchmarks: |d| < 0.2 trivial, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large.

### Spearman Rank Correlation
Used for party × ideology because both are ordinal scales and the relationship is not assumed linear.

---

## Dimensionality Reduction (`scripts/figures/`)

### UMAP (Figure 1)
Uniform Manifold Approximation and Projection. A non-linear technique that preserves local neighbor structure in high-dimensional data. Applied to the 9 AP items (own-party version per respondent) to reveal natural clustering.

Parameters: `n_neighbors=15`, `min_dist=0.1`, `random_state=42` for reproducibility.

### PCA Biplot (Figure 2)
Principal Component Analysis. A linear technique that identifies orthogonal axes of maximum variance. The biplot overlays respondent scores with item loading vectors:
- **PC1** captures the primary axis of variance — expected to be general polarization (all items load in the same direction).
- **PC2** captures secondary variance — expected to differentiate aversion items from moral/othering items.

Items are standardized (mean=0, SD=1) before PCA so that scale differences do not inflate loadings.
