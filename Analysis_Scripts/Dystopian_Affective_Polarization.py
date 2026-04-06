"""
DYSTOPIAN MEDIA × AFFECTIVE POLARIZATION ANALYSES
UNC Charlotte Political Polarization Study - Fall 2025
Author: Lean

Research Questions:
  RQ1: Which dystopian works correlate with affective polarization measures?
  RQ2: Do Democrats and Republicans differ in dystopian media liking?
  RQ3: Do movies/TV vs. books show different polarization signatures?
  RQ4: Do high vs. low dystopian consumers differ in polarization?
  RQ5: Does dystopian consumption predict polarization after controls?

Sections:
  A1 — Correlation heatmap: all 9 works + index × all polarization measures
  A2 — All 9 dystopian works by party (grouped bar chart)
  A3 — Movies/TV sub-index vs. Books sub-index × polarization (r comparison)
  A4 — High vs. Low dystopian consumers: polarization profiles
  A5 — Scatter plots: dystopian_index vs. key polarization indices (by party)
  A6 — Multiple regression: dystopian_index → polarization (with controls)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, f_oneway
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

DATA_PATH = '../../../Data_Cleaned_Thesis.csv'   # relative to Analysis_Scripts/
# Fallback: absolute path
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/Users/lena/Downloads/Data_Cleaned_Thesis.csv'

OUT_DIR = 'Dystopian_Analyses'
os.makedirs(OUT_DIR, exist_ok=True)

blue, red, gray = '#0015BC', '#E81B23', '#808080'
PARTY_COLORS = {'Democrat': blue, 'Republican': red, 'Independent': gray}

print("=" * 80)
print("DYSTOPIAN MEDIA × AFFECTIVE POLARIZATION ANALYSES")
print("=" * 80)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Party category ─────────────────────────────────────────────────────────────
df['party_cat'] = pd.cut(
    df['party_combined'],
    bins=[0, 2.5, 3.5, 6],
    labels=['Democrat', 'Independent', 'Republican']
)

# ── Clean age (remove clearly invalid entries) ─────────────────────────────────
df['age_clean'] = df['age'].where((df['age'] >= 16) & (df['age'] <= 80))

# ── Variable definitions ───────────────────────────────────────────────────────
DYSTOPIAN_WORKS = {
    'Hunger_num':     'Hunger Games',
    'Divergent_num':  'Divergent',
    'MazeRun_num':    'Maze Runner',
    'BlackMirror_num':'Black Mirror',
    'LastofUs_num':   'Last of Us',
    'The100_num':     'The 100',
    '1984_num':       '1984',
    'Giver_num':      'The Giver',
    '451_num':        'Fahrenheit 451',
}
MOVIE_COLS = ['Hunger_num', 'Divergent_num', 'MazeRun_num',
              'BlackMirror_num', 'LastofUs_num', 'The100_num']
BOOK_COLS  = ['1984_num', 'Giver_num', '451_num']

POLARIZATION_MEASURES = {
    'aversion_index':             'Aversion\n(social avoidance)',
    'othering_index':             'Othering\n(perceptual gap)',
    'moralizing_index':           'Moralizing\n(moral identity)',
    'partisan_polarization_index':'Partisan\nPolarization',
    'FT_gap_DminusR':             'Feeling Therm.\nGap (D−R)',
    'freespeech_index':           'Free Speech\nPermissiveness',
}

# Build sub-indices
df['movies_index'] = df[MOVIE_COLS].mean(axis=1)
df['books_index']  = df[BOOK_COLS].mean(axis=1)

dem = df[df['party_cat'] == 'Democrat']
rep = df[df['party_cat'] == 'Republican']


def save_fig(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close(fig)


def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'ns'


def cohens_d(a, b):
    pooled_sd = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan


# ────────────────────────────────────────────────────────────────────────────────
# A1: CORRELATION HEATMAP — All dystopian works × all polarization measures
# ────────────────────────────────────────────────────────────────────────────────
print("\n── A1: CORRELATION MATRIX ──")

dyst_cols  = list(DYSTOPIAN_WORKS.keys()) + ['dystopian_index']
dyst_labels = [DYSTOPIAN_WORKS[c] for c in list(DYSTOPIAN_WORKS.keys())] + ['Dystopian\nIndex']
pol_cols   = list(POLARIZATION_MEASURES.keys())
pol_labels = list(POLARIZATION_MEASURES.values())

r_mat = np.full((len(dyst_cols), len(pol_cols)), np.nan)
p_mat = np.full((len(dyst_cols), len(pol_cols)), np.nan)

for i, dc in enumerate(dyst_cols):
    for j, pc in enumerate(pol_cols):
        pair = df[[dc, pc]].dropna()
        if len(pair) >= 20:
            r, p = pearsonr(pair[dc], pair[pc])
            r_mat[i, j] = r
            p_mat[i, j] = p
            stars = sig_stars(p)
            print(f"  {dc:20s} × {pc:30s}: r={r:+.3f}, p={p:.4f} {stars}")

# Annotation: r value + stars
annot = np.empty_like(r_mat, dtype=object)
for i in range(r_mat.shape[0]):
    for j in range(r_mat.shape[1]):
        if not np.isnan(r_mat[i, j]):
            annot[i, j] = f"{r_mat[i,j]:+.2f}{sig_stars(p_mat[i,j]).replace('ns','')}"
        else:
            annot[i, j] = ''

fig, ax = plt.subplots(figsize=(13, 8))
sns.heatmap(
    r_mat, annot=annot, fmt='', cmap='RdBu_r', vmin=-0.4, vmax=0.4,
    xticklabels=pol_labels, yticklabels=dyst_labels,
    linewidths=0.5, ax=ax, cbar_kws={'label': 'Pearson r'}
)
ax.set_title(
    "Dystopian Media Liking × Affective Polarization Measures\n"
    "(* p<.05, ** p<.01, *** p<.001)",
    fontweight='bold', fontsize=13
)
ax.set_xlabel("Polarization Measure", fontweight='bold')
ax.set_ylabel("Dystopian Work", fontweight='bold')
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0)
plt.tight_layout()
save_fig(fig, 'A1_correlation_heatmap.png')


# ────────────────────────────────────────────────────────────────────────────────
# A2: ALL 9 DYSTOPIAN WORKS BY PARTY
# ────────────────────────────────────────────────────────────────────────────────
print("\n── A2: DYSTOPIAN WORKS BY PARTY ──")

work_cols   = list(DYSTOPIAN_WORKS.keys())
work_names  = list(DYSTOPIAN_WORKS.values())
results_a2  = {}

for col, name in DYSTOPIAN_WORKS.items():
    d_vals = dem[col].dropna()
    r_vals = rep[col].dropna()
    if len(d_vals) >= 15 and len(r_vals) >= 15:
        t, p = ttest_ind(d_vals, r_vals)
        d_eff = cohens_d(d_vals, r_vals)
        results_a2[col] = {
            'name': name, 'dem_m': d_vals.mean(), 'dem_se': d_vals.sem(),
            'rep_m': r_vals.mean(), 'rep_se': r_vals.sem(),
            't': t, 'p': p, 'd': d_eff,
            'dem_n': len(d_vals), 'rep_n': len(r_vals)
        }
        print(f"  {name:20s}: Dem={d_vals.mean():.2f}(n={len(d_vals)}), "
              f"Rep={r_vals.mean():.2f}(n={len(r_vals)}), "
              f"t={t:.2f}, {sig_stars(p)}, d={d_eff:.2f}")

if results_a2:
    n_works = len(results_a2)
    x = np.arange(n_works)
    w = 0.35
    names_ordered = [results_a2[c]['name'] for c in work_cols if c in results_a2]

    fig, ax = plt.subplots(figsize=(14, 7))
    dem_means = [results_a2[c]['dem_m'] for c in work_cols if c in results_a2]
    dem_sems  = [results_a2[c]['dem_se'] for c in work_cols if c in results_a2]
    rep_means = [results_a2[c]['rep_m'] for c in work_cols if c in results_a2]
    rep_sems  = [results_a2[c]['rep_se'] for c in work_cols if c in results_a2]
    p_vals    = [results_a2[c]['p']     for c in work_cols if c in results_a2]

    ax.bar(x - w/2, dem_means, w, yerr=dem_sems, capsize=4,
           color=blue, alpha=0.82, label='Democrats', edgecolor='black', linewidth=0.7)
    ax.bar(x + w/2, rep_means, w, yerr=rep_sems, capsize=4,
           color=red,  alpha=0.82, label='Republicans', edgecolor='black', linewidth=0.7)

    for i, p in enumerate(p_vals):
        stars = sig_stars(p).replace('ns', '')
        if stars:
            y_top = max(dem_means[i] + dem_sems[i], rep_means[i] + rep_sems[i]) + 0.12
            ax.text(i, y_top, stars, ha='center', fontsize=13, fontweight='bold')

    # Divider between movies and books
    ax.axvline(5.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2.5, 5.25, 'TV / Movies', ha='center', fontsize=10,
            color='gray', style='italic')
    ax.text(7.0, 5.25, 'Books', ha='center', fontsize=10,
            color='gray', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(names_ordered, fontsize=10.5)
    ax.set_ylabel('Mean Liking (1 = Dislike, 5 = Like)', fontweight='bold')
    ax.set_ylim(1, 5.6)
    ax.set_title('Dystopian Media Liking by Party\n'
                 '(Error bars = SEM; * p<.05, ** p<.01, *** p<.001)',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig(fig, 'A2_dystopian_by_party.png')


# ────────────────────────────────────────────────────────────────────────────────
# A3: MOVIES/TV vs BOOKS SUB-INDICES × POLARIZATION
# ────────────────────────────────────────────────────────────────────────────────
print("\n── A3: MOVIES vs BOOKS SUB-INDICES × POLARIZATION ──")

r_movies, r_books, p_movies, p_books = [], [], [], []
pol_short = ['Aversion', 'Othering', 'Moralizing', 'Part. Polar.', 'FT Gap', 'Free Speech']

for pc in pol_cols:
    for label, idx_col, r_list, p_list in [
        ('Movies', 'movies_index', r_movies, p_movies),
        ('Books',  'books_index',  r_books,  p_books),
    ]:
        pair = df[[idx_col, pc]].dropna()
        if len(pair) >= 20:
            r, p = pearsonr(pair[idx_col], pair[pc])
            r_list.append(r); p_list.append(p)
            print(f"  {label:6s} × {pc:30s}: r={r:+.3f} {sig_stars(p)}")
        else:
            r_list.append(np.nan); p_list.append(np.nan)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(pol_cols)); w = 0.35

bars_m = ax.bar(x - w/2, r_movies, w, color='#5B8DB8', alpha=0.85,
                label='TV/Movies Index', edgecolor='black', linewidth=0.7)
bars_b = ax.bar(x + w/2, r_books,  w, color='#C4704F', alpha=0.85,
                label='Books Index',    edgecolor='black', linewidth=0.7)

for i, (rm, rb, pm, pb) in enumerate(zip(r_movies, r_books, p_movies, p_books)):
    if not np.isnan(rm):
        s = sig_stars(pm).replace('ns', '')
        if s: ax.text(i - w/2, rm + (0.005 if rm >= 0 else -0.015), s,
                      ha='center', fontsize=11, fontweight='bold')
    if not np.isnan(rb):
        s = sig_stars(pb).replace('ns', '')
        if s: ax.text(i + w/2, rb + (0.005 if rb >= 0 else -0.015), s,
                      ha='center', fontsize=11, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pol_short, fontsize=11)
ax.set_ylabel("Pearson r", fontweight='bold')
ax.set_title("Dystopian TV/Movies vs. Books: Correlations with Polarization Measures\n"
             "(* p<.05, ** p<.01, *** p<.001)",
             fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'A3_movies_vs_books_correlations.png')


# ────────────────────────────────────────────────────────────────────────────────
# A4: HIGH vs LOW DYSTOPIAN CONSUMERS — Polarization Profiles
# ────────────────────────────────────────────────────────────────────────────────
print("\n── A4: HIGH vs LOW DYSTOPIAN CONSUMERS ──")

median_dyst = df['dystopian_index'].median()
print(f"  Median dystopian_index = {median_dyst:.2f}")
df['dyst_group'] = np.where(df['dystopian_index'] > median_dyst, 'High',
                   np.where(df['dystopian_index'] < median_dyst, 'Low', None))

high = df[df['dyst_group'] == 'High']
low  = df[df['dyst_group'] == 'Low']
print(f"  High group: n={len(high)}, Low group: n={len(low)}")

pol_display = {
    'aversion_index':             ('Aversion', '1–5'),
    'othering_index':             ('Othering', '1–5'),
    'moralizing_index':           ('Moralizing', '1–5'),
    'partisan_polarization_index':('Partisan\nPolar.', '1–7'),
    'FT_gap_DminusR':             ('FT Gap\n(D−R)', '0–100'),
    'freespeech_index':           ('Free Speech\nPermissive', '1–7'),
}

n_pol = len(pol_display)
fig, axes = plt.subplots(1, n_pol, figsize=(15, 6))
fig.suptitle("Affective Polarization: High vs. Low Dystopian Media Consumers\n"
             f"(Split at median dystopian_index = {median_dyst:.2f}; error bars = SEM)",
             fontweight='bold', fontsize=13)

for ax, (pc, (short, scale)) in zip(axes, pol_display.items()):
    h_vals = high[pc].dropna()
    l_vals = low[pc].dropna()
    if len(h_vals) >= 10 and len(l_vals) >= 10:
        t, p = ttest_ind(h_vals, l_vals)
        d_eff = cohens_d(h_vals, l_vals)
        stars = sig_stars(p)
        print(f"  {pc:32s}: High={h_vals.mean():.2f}, Low={l_vals.mean():.2f}, "
              f"t={t:.2f}, {stars}, d={d_eff:.2f}")

        vals   = [h_vals.mean(), l_vals.mean()]
        sems   = [h_vals.sem(),  l_vals.sem()]
        colors = ['#2ca25f', '#756bb1']
        bars   = ax.bar(['High\n(n={})'.format(len(h_vals)),
                          'Low\n(n={})'.format(len(l_vals))],
                         vals, yerr=sems, capsize=5, color=colors, alpha=0.82,
                         edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + sems[vals.index(val)] + 0.05,
                    f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
        y_max = max(v + s for v, s in zip(vals, sems))
        sig_str = stars.replace('ns', 'n.s.')
        ax.text(0.5, y_max + 0.12, sig_str, ha='center', transform=ax.get_xaxis_transform(),
                fontsize=12, fontweight='bold')
        ax.set_title(f"{short}\n({scale})", fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('')

plt.tight_layout()
save_fig(fig, 'A4_high_vs_low_dystopian.png')


# ────────────────────────────────────────────────────────────────────────────────
# A5: SCATTER PLOTS — dystopian_index vs key polarization indices (by party)
# ────────────────────────────────────────────────────────────────────────────────
print("\n── A5: SCATTER PLOTS (dystopian_index × polarization, by party) ──")

scatter_targets = [
    ('aversion_index',              'Aversion Index (1–5)'),
    ('othering_index',              'Othering Index (1–5)'),
    ('moralizing_index',            'Moralizing Index (1–5)'),
    ('FT_gap_DminusR',              'Feeling Thermometer Gap (D−R)'),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Dystopian Media Index vs. Affective Polarization Measures\n"
             "(Points colored by party; lines = OLS fits)",
             fontweight='bold', fontsize=13)

for ax, (pc, ylabel) in zip(axes.flatten(), scatter_targets):
    pair_all = df[['dystopian_index', pc, 'party_cat']].dropna()

    # Overall OLS line
    x_all = pair_all['dystopian_index'].values
    y_all = pair_all[pc].values
    if len(x_all) >= 20:
        m, b = np.polyfit(x_all, y_all, 1)
        r_all, p_all = pearsonr(x_all, y_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, m * x_line + b, color='black', linewidth=1.8,
                linestyle='--', label=f'Overall r={r_all:+.2f}{sig_stars(p_all).replace("ns","")}',
                zorder=5)

    # By party
    for party, color in [('Democrat', blue), ('Republican', red), ('Independent', gray)]:
        sub = pair_all[pair_all['party_cat'] == party]
        if len(sub) < 10:
            continue
        ax.scatter(sub['dystopian_index'], sub[pc], color=color, alpha=0.35,
                   s=25, edgecolors='none', zorder=3)
        m2, b2 = np.polyfit(sub['dystopian_index'], sub[pc], 1)
        r2, p2  = pearsonr(sub['dystopian_index'], sub[pc])
        ax.plot(x_line, m2 * x_line + b2, color=color, linewidth=1.5,
                label=f'{party} r={r2:+.2f}{sig_stars(p2).replace("ns","")}',
                zorder=4)
        print(f"  {party:12s} × {pc:32s}: r={r2:+.3f} {sig_stars(p2)}")

    ax.set_xlabel('Dystopian Index (1–5)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f'Dystopian Index vs.\n{ylabel}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig(fig, 'A5_scatter_dystopian_polarization.png')


# ────────────────────────────────────────────────────────────────────────────────
# A6: MULTIPLE REGRESSION — dystopian_index → polarization (with controls)
# ────────────────────────────────────────────────────────────────────────────────
print("\n── A6: MULTIPLE REGRESSION ──")

REG_CONTROLS = ['party_combined', 'ideology_num', 'age_clean', 'gender_num', 'religiosity_num']
REG_DVS = [
    ('aversion_index',  'DV1: Social Aversion'),
    ('FT_gap_DminusR',  'DV2: Feeling Thermometer Gap'),
]

reg_results = {}
for dv_col, dv_label in REG_DVS:
    ivs = ['dystopian_index'] + REG_CONTROLS
    model_df = df[[dv_col] + ivs].dropna()
    X = sm.add_constant(model_df[ivs])
    y = model_df[dv_col]
    model = sm.OLS(y, X).fit()
    reg_results[dv_col] = {'label': dv_label, 'model': model}
    print(f"\n  {dv_label} (n={len(model_df)})")
    print(f"  R² = {model.rsquared:.3f}, Adj. R² = {model.rsquared_adj:.3f}")
    print(f"  {'Variable':<25} {'β':>8} {'SE':>8} {'p':>8}")
    print("  " + "-"*52)
    for var in ivs:
        b  = model.params[var]
        se = model.bse[var]
        p  = model.pvalues[var]
        print(f"  {var:<25} {b:>8.3f} {se:>8.3f} {p:>8.4f} {sig_stars(p)}")

# Coefficient plot
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
fig.suptitle("Multiple Regression: Predicting Affective Polarization\n"
             "from Dystopian Media Consumption (controlling for party, ideology, demographics)",
             fontweight='bold', fontsize=13)

var_labels = {
    'dystopian_index': 'Dystopian Index',
    'party_combined':  'Party (1=Dem → 5=Rep)',
    'ideology_num':    'Ideology (1=Liberal → 7=Con)',
    'age_clean':       'Age',
    'gender_num':      'Gender',
    'religiosity_num': 'Religiosity',
}

ivs_order = ['dystopian_index'] + REG_CONTROLS

for ax, (dv_col, (dv_label, _)) in zip(axes, [(k, (v['label'], None)) for k, v in reg_results.items()]):
    model = reg_results[dv_col]['model']
    coefs = [model.params[v] for v in ivs_order]
    ci_lo = [model.conf_int().loc[v, 0] for v in ivs_order]
    ci_hi = [model.conf_int().loc[v, 1] for v in ivs_order]
    pvals = [model.pvalues[v] for v in ivs_order]
    labels = [var_labels[v] for v in ivs_order]

    y_pos = np.arange(len(ivs_order))
    bar_colors = ['#2ca25f' if v == 'dystopian_index' else
                  (blue if c < 0 else red) for v, c in zip(ivs_order, coefs)]

    ax.barh(y_pos, coefs,
            xerr=np.array([[c - lo for c, lo in zip(coefs, ci_lo)],
                           [hi - c for c, hi in zip(coefs, ci_hi)]]),
            color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.6,
            capsize=4, height=0.6)
    ax.axvline(0, color='black', linewidth=1)

    for y, (c, p) in enumerate(zip(coefs, pvals)):
        s = sig_stars(p).replace('ns', '')
        if s:
            offset = 0.02 * (max(ci_hi) - min(ci_lo)) if max(ci_hi) != min(ci_lo) else 0.02
            ax.text(ci_hi[y] + offset, y, s, va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel('Unstandardized Coefficient (β)', fontweight='bold')
    ax.set_title(f"{reg_results[dv_col]['label']}\n"
                 f"R²={model.rsquared:.3f}, n={int(model.nobs)}",
                 fontweight='bold', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig(fig, 'A6_regression_coefplot.png')


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("✅  DYSTOPIAN × AFFECTIVE POLARIZATION ANALYSES COMPLETE")
print(f"    All figures saved to: {OUT_DIR}/")
print("=" * 80)
print("""
Files generated:
  A1_correlation_heatmap.png          — 9 works + index × 6 polarization measures (r heatmap)
  A2_dystopian_by_party.png           — Mean liking for all 9 works: Democrats vs Republicans
  A3_movies_vs_books_correlations.png — TV/movie index vs. book index: different r profiles?
  A4_high_vs_low_dystopian.png        — High vs. Low consumers: polarization profile comparison
  A5_scatter_dystopian_polarization.png — Scatter: dystopian_index vs. key measures, by party
  A6_regression_coefplot.png          — OLS coefficient plots controlling for demographics
""")
