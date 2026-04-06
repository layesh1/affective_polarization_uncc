"""
DYSTOPIAN MEDIA × AFFECTIVE POLARIZATION — STATISTICAL SUMMARIES & BAR CHARTS
UNC Charlotte Political Polarization Study - Fall 2025
Author: Lean

Outputs (all to Dystopian_Analyses/):

  STATISTICAL SUMMARIES (CSV + printed tables)
    summary_descriptives.csv       — Mean, SD, n for all 9 works overall & by party
    summary_familiarity.csv        — % familiar with each work (overall & by party)
    summary_correlations.csv       — Pearson r, p, n for every dystopian × polarization pair
    summary_party_ttests.csv       — t, p, Cohen's d for Dem vs Rep on every work
    summary_regression.csv         — OLS β, SE, p, 95% CI for both regression models
    summary_highlow.csv            — High vs Low consumer group means, t, d, p

  BAR CHARTS
    B1_mean_liking_overall.png     — Horizontal sorted bar: mean liking all 9 works
    B2_familiarity_rates.png       — % of respondents familiar with each work (by party)
    B3_polarization_by_party.png   — Aversion / Othering / Moralizing means: D vs R
    B4_dystopian_corr_bar.png      — Bar chart of r values: dystopian_index × 6 measures
    B5_liking_by_party_all.png     — Simple grouped bar for ALL 9 works × party
    B6_item_level_aversion.png     — Aversion item means: Dem vs Rep (3 items)
    B7_books_liking_bar.png        — Books-only mean liking (overall, clean)
    B8_movies_liking_bar.png       — TV/Movies-only mean liking (overall, clean)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

DATA_PATH = '/Users/lena/Downloads/Data_Cleaned_Thesis.csv'
OUT_DIR   = '/Users/lena/Downloads/Thesis/Dystopian_Analyses'
os.makedirs(OUT_DIR, exist_ok=True)

blue, red, gray = '#0015BC', '#E81B23', '#808080'

print("=" * 80)
print("DYSTOPIAN × POLARIZATION — STATISTICAL SUMMARIES & BAR CHARTS")
print("=" * 80)

# ── Load & prep ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

df['party_cat'] = pd.cut(
    df['party_combined'],
    bins=[0, 2.5, 3.5, 6],
    labels=['Democrat', 'Independent', 'Republican']
)
df['age_clean'] = df['age'].where((df['age'] >= 16) & (df['age'] <= 80))

WORKS = {
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
MOVIE_COLS = ['Hunger_num','Divergent_num','MazeRun_num',
              'BlackMirror_num','LastofUs_num','The100_num']
BOOK_COLS  = ['1984_num','Giver_num','451_num']
MOVIE_NAMES = {k: WORKS[k] for k in MOVIE_COLS}
BOOK_NAMES  = {k: WORKS[k] for k in BOOK_COLS}

POL = {
    'aversion_index':             'Aversion',
    'othering_index':             'Othering',
    'moralizing_index':           'Moralizing',
    'partisan_polarization_index':'Partisan Polarization',
    'FT_gap_DminusR':             'FT Gap (D−R)',
    'freespeech_index':           'Free Speech',
}

dem = df[df['party_cat'] == 'Democrat']
rep = df[df['party_cat'] == 'Republican']
ind = df[df['party_cat'] == 'Independent']

TOTAL_N = len(df)   # denominator for familiarity %


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
    ps = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Descriptive stats ───────────────────────────────────────────────────────
print("\n── SUMMARY 1: DESCRIPTIVE STATS ──")
rows = []
for col, name in WORKS.items():
    overall = df[col].dropna()
    d_vals  = dem[col].dropna()
    r_vals  = rep[col].dropna()
    rows.append({
        'Work': name,
        'Overall_n': len(overall), 'Overall_M': round(overall.mean(), 3), 'Overall_SD': round(overall.std(), 3),
        'Dem_n': len(d_vals),      'Dem_M': round(d_vals.mean(), 3),      'Dem_SD': round(d_vals.std(), 3),
        'Rep_n': len(r_vals),      'Rep_M': round(r_vals.mean(), 3),      'Rep_SD': round(r_vals.std(), 3),
    })

desc_df = pd.DataFrame(rows)
desc_df.to_csv(os.path.join(OUT_DIR, 'summary_descriptives.csv'), index=False)
print(desc_df[['Work','Overall_n','Overall_M','Overall_SD','Dem_M','Rep_M']].to_string(index=False))
print(f"  ✓ Saved: summary_descriptives.csv")


# ── 2. Familiarity rates ───────────────────────────────────────────────────────
print("\n── SUMMARY 2: FAMILIARITY RATES ──")
fam_rows = []
for col, name in WORKS.items():
    n_familiar_all = df[col].notna().sum()
    n_familiar_dem = dem[col].notna().sum()
    n_familiar_rep = rep[col].notna().sum()
    fam_rows.append({
        'Work': name,
        'Pct_Familiar_Overall': round(n_familiar_all / TOTAL_N * 100, 1),
        'Pct_Familiar_Dem':     round(n_familiar_dem / len(dem) * 100, 1) if len(dem) > 0 else np.nan,
        'Pct_Familiar_Rep':     round(n_familiar_rep / len(rep) * 100, 1) if len(rep) > 0 else np.nan,
        'n_familiar': n_familiar_all,
    })

fam_df = pd.DataFrame(fam_rows)
fam_df.to_csv(os.path.join(OUT_DIR, 'summary_familiarity.csv'), index=False)
print(fam_df[['Work','Pct_Familiar_Overall','Pct_Familiar_Dem','Pct_Familiar_Rep']].to_string(index=False))
print(f"  ✓ Saved: summary_familiarity.csv")


# ── 3. Correlation table ───────────────────────────────────────────────────────
print("\n── SUMMARY 3: CORRELATIONS ──")
corr_rows = []
dyst_cols  = list(WORKS.keys()) + ['dystopian_index']
dyst_names = list(WORKS.values()) + ['Dystopian Index']

for col, name in zip(dyst_cols, dyst_names):
    for pcol, pname in POL.items():
        pair = df[[col, pcol]].dropna()
        if len(pair) >= 20:
            r, p = pearsonr(pair[col], pair[pcol])
        else:
            r, p = np.nan, np.nan
        corr_rows.append({
            'Dystopian_Work': name,
            'Polarization_Measure': pname,
            'r': round(r, 3), 'p': round(p, 4), 'n': len(pair),
            'sig': sig_stars(p) if not np.isnan(p) else '',
        })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(os.path.join(OUT_DIR, 'summary_correlations.csv'), index=False)
# Print only significant rows
sig_corrs = corr_df[corr_df['sig'].isin(['*','**','***'])]
print(sig_corrs.to_string(index=False))
print(f"  ✓ Saved: summary_correlations.csv ({len(sig_corrs)}/{len(corr_df)} significant)")


# ── 4. Party t-tests ───────────────────────────────────────────────────────────
print("\n── SUMMARY 4: PARTY T-TESTS (Dem vs Rep) ──")
ttest_rows = []
for col, name in WORKS.items():
    d_vals = dem[col].dropna()
    r_vals = rep[col].dropna()
    if len(d_vals) >= 15 and len(r_vals) >= 15:
        t, p = ttest_ind(d_vals, r_vals)
        d_eff = cohens_d(d_vals, r_vals)
    else:
        t, p, d_eff = np.nan, np.nan, np.nan
    ttest_rows.append({
        'Work': name,
        'Dem_M': round(d_vals.mean(), 3), 'Dem_SD': round(d_vals.std(), 3), 'Dem_n': len(d_vals),
        'Rep_M': round(r_vals.mean(), 3), 'Rep_SD': round(r_vals.std(), 3), 'Rep_n': len(r_vals),
        't': round(t, 3), 'p': round(p, 4), "Cohen's_d": round(d_eff, 3),
        'sig': sig_stars(p) if not np.isnan(p) else '',
    })

ttest_df = pd.DataFrame(ttest_rows)
ttest_df.to_csv(os.path.join(OUT_DIR, 'summary_party_ttests.csv'), index=False)
print(ttest_df[['Work','Dem_M','Rep_M','t','p','sig',"Cohen's_d"]].to_string(index=False))
print(f"  ✓ Saved: summary_party_ttests.csv")


# ── 5. Regression summaries ────────────────────────────────────────────────────
print("\n── SUMMARY 5: REGRESSION MODELS ──")
REG_IVS = ['dystopian_index','party_combined','ideology_num','age_clean','gender_num','religiosity_num']
IV_LABELS = {
    'dystopian_index':'Dystopian Index',
    'party_combined': 'Party (1=Dem→5=Rep)',
    'ideology_num':   'Ideology (1=Lib→7=Con)',
    'age_clean':      'Age',
    'gender_num':     'Gender',
    'religiosity_num':'Religiosity',
}
reg_rows = []
for dv_col, dv_label in [('aversion_index','Social Aversion'), ('FT_gap_DminusR','FT Gap (D−R)')]:
    model_df = df[[dv_col] + REG_IVS].dropna()
    X = sm.add_constant(model_df[REG_IVS])
    model = sm.OLS(model_df[dv_col], X).fit()
    print(f"\n  {dv_label}  R²={model.rsquared:.3f}, n={len(model_df)}")
    for iv in REG_IVS:
        ci = model.conf_int().loc[iv]
        row = {
            'DV': dv_label, 'IV': IV_LABELS[iv],
            'beta': round(model.params[iv], 3),
            'SE':   round(model.bse[iv], 3),
            'p':    round(model.pvalues[iv], 4),
            'CI_lo': round(ci[0], 3), 'CI_hi': round(ci[1], 3),
            'sig':  sig_stars(model.pvalues[iv]),
            'R2':   round(model.rsquared, 3),
            'n':    len(model_df),
        }
        reg_rows.append(row)
        print(f"    {IV_LABELS[iv]:30s}  β={row['beta']:+.3f}  SE={row['SE']:.3f}  "
              f"p={row['p']:.4f}  {row['sig']}")

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(os.path.join(OUT_DIR, 'summary_regression.csv'), index=False)
print(f"  ✓ Saved: summary_regression.csv")


# ── 6. High vs Low group summary ──────────────────────────────────────────────
print("\n── SUMMARY 6: HIGH vs LOW DYSTOPIAN CONSUMERS ──")
med = df['dystopian_index'].median()
df['dyst_group'] = np.where(df['dystopian_index'] > med, 'High',
                   np.where(df['dystopian_index'] < med, 'Low', None))
high = df[df['dyst_group'] == 'High']
low  = df[df['dyst_group'] == 'Low']

hl_rows = []
for pcol, pname in POL.items():
    h_vals = high[pcol].dropna(); l_vals = low[pcol].dropna()
    if len(h_vals) >= 10 and len(l_vals) >= 10:
        t, p = ttest_ind(h_vals, l_vals)
        d_eff = cohens_d(h_vals, l_vals)
    else:
        t, p, d_eff = np.nan, np.nan, np.nan
    hl_rows.append({
        'Measure': pname,
        'High_M': round(h_vals.mean(), 3), 'High_SD': round(h_vals.std(), 3), 'High_n': len(h_vals),
        'Low_M':  round(l_vals.mean(), 3), 'Low_SD':  round(l_vals.std(), 3), 'Low_n':  len(l_vals),
        't': round(t, 3), 'p': round(p, 4), "Cohen's_d": round(d_eff, 3),
        'sig': sig_stars(p) if not np.isnan(p) else '',
    })

hl_df = pd.DataFrame(hl_rows)
hl_df.to_csv(os.path.join(OUT_DIR, 'summary_highlow.csv'), index=False)
print(hl_df[['Measure','High_M','Low_M','t','p','sig',"Cohen's_d"]].to_string(index=False))
print(f"  ✓ Saved: summary_highlow.csv")


# ══════════════════════════════════════════════════════════════════════════════
# BAR CHARTS
# ══════════════════════════════════════════════════════════════════════════════

# ── B1: Mean liking overall — horizontal sorted bar ───────────────────────────
print("\n── B1: MEAN LIKING OVERALL (sorted) ──")
means_all  = {WORKS[c]: df[c].dropna().mean() for c in WORKS}
sems_all   = {WORKS[c]: df[c].dropna().sem()  for c in WORKS}
sorted_items = sorted(means_all.items(), key=lambda x: x[1])

names_s = [x[0] for x in sorted_items]
vals_s  = [x[1] for x in sorted_items]
sems_s  = [sems_all[n] for n in names_s]
colors_s = ['#C4704F' if n in BOOK_NAMES.values() else '#5B8DB8' for n in names_s]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(names_s, vals_s, xerr=sems_s, capsize=4,
               color=colors_s, alpha=0.85, edgecolor='black', linewidth=0.7, height=0.65)
for bar, val in zip(bars, vals_s):
    ax.text(val + 0.04, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=10)

from matplotlib.patches import Patch
legend_handles = [Patch(color='#5B8DB8', alpha=0.85, label='TV / Movies'),
                  Patch(color='#C4704F', alpha=0.85, label='Books')]
ax.legend(handles=legend_handles, fontsize=11, loc='lower right')

ax.set_xlabel('Mean Liking (1 = Dislike, 5 = Like)', fontweight='bold')
ax.set_title('Overall Mean Liking for Dystopian Works\n(Error bars = SEM; sorted ascending)',
             fontweight='bold', fontsize=13)
ax.set_xlim(1, 5.5)
ax.axvline(3, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Neutral (3)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'B1_mean_liking_overall.png')


# ── B2: Familiarity rates by party ────────────────────────────────────────────
print("\n── B2: FAMILIARITY RATES ──")
work_order = list(WORKS.values())
fam_all  = [df[c].notna().sum() / TOTAL_N * 100         for c in WORKS]
fam_dem  = [dem[c].notna().sum() / len(dem) * 100 if len(dem) > 0 else 0 for c in WORKS]
fam_rep  = [rep[c].notna().sum() / len(rep) * 100 if len(rep) > 0 else 0 for c in WORKS]

x = np.arange(len(WORKS)); w = 0.28
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w, fam_dem, w, color=blue, alpha=0.8, label='Democrats', edgecolor='black', linewidth=0.7)
ax.bar(x,     fam_all, w, color='#4a4a4a', alpha=0.7, label='Overall',    edgecolor='black', linewidth=0.7)
ax.bar(x + w, fam_rep, w, color=red,  alpha=0.8, label='Republicans', edgecolor='black', linewidth=0.7)

ax.axvline(5.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
ax.text(2.5, 102, 'TV / Movies', ha='center', fontsize=10, color='gray', style='italic')
ax.text(7.0, 102, 'Books',       ha='center', fontsize=10, color='gray', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(work_order, fontsize=10.5)
ax.set_ylabel('% Familiar with Work', fontweight='bold')
ax.set_ylim(0, 115)
ax.set_title('Familiarity with Dystopian Works by Party\n(% who have seen/read the work)',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'B2_familiarity_rates.png')


# ── B3: Affective polarization components by party ────────────────────────────
print("\n── B3: POLARIZATION COMPONENTS BY PARTY ──")
pol3 = {
    'aversion_index':   ('Aversion\n(1–5)',   1, 5),
    'othering_index':   ('Othering\n(1–5)',    1, 5),
    'moralizing_index': ('Moralizing\n(1–5)',  1, 5),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 6))
fig.suptitle('Affective Polarization Components: Democrats vs. Republicans\n'
             '(Error bars = SEM)', fontweight='bold', fontsize=13)

for ax, (pcol, (label, ymin, ymax)) in zip(axes, pol3.items()):
    d_vals = dem[pcol].dropna()
    r_vals = rep[pcol].dropna()
    t, p   = ttest_ind(d_vals, r_vals)
    d_eff  = cohens_d(d_vals, r_vals)
    stars  = sig_stars(p)
    means  = [d_vals.mean(), r_vals.mean()]
    sems   = [d_vals.sem(),  r_vals.sem()]
    labels = [f'Dem\n(n={len(d_vals)})', f'Rep\n(n={len(r_vals)})']

    bars = ax.bar(labels, means, yerr=sems, capsize=5,
                  color=[blue, red], alpha=0.82, edgecolor='black', linewidth=0.7, width=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + max(sems) + 0.06,
                f'{m:.2f}', ha='center', fontsize=11, fontweight='bold')

    y_top = max(m + s for m, s in zip(means, sems)) + 0.18
    ax.text(0.5, y_top, f'{stars}\nd={d_eff:.2f}', ha='center',
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylim(ymin, ymax + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig(fig, 'B3_polarization_by_party.png')


# ── B4: Correlation bar — dystopian_index × polarization measures ─────────────
print("\n── B4: CORRELATION BAR (dystopian_index × polarization) ──")
r_vals_bar, p_vals_bar, pol_names_bar = [], [], []
for pcol, pname in POL.items():
    pair = df[['dystopian_index', pcol]].dropna()
    if len(pair) >= 20:
        r, p = pearsonr(pair['dystopian_index'], pair[pcol])
        r_vals_bar.append(r); p_vals_bar.append(p); pol_names_bar.append(pname)

colors_bar = [blue if r < 0 else red for r in r_vals_bar]
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(pol_names_bar, r_vals_bar, color=colors_bar, alpha=0.82,
              edgecolor='black', linewidth=0.7, width=0.55)
for bar, r, p in zip(bars, r_vals_bar, p_vals_bar):
    s = sig_stars(p).replace('ns', '')
    y = r + 0.005 if r >= 0 else r - 0.012
    ax.text(bar.get_x() + bar.get_width() / 2, y, f'{r:+.2f}{s}',
            ha='center', va='bottom' if r >= 0 else 'top', fontsize=10, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.9)
ax.set_ylabel('Pearson r', fontweight='bold')
ax.set_title('Dystopian Media Index: Correlations with Affective Polarization Measures\n'
             '(* p<.05, ** p<.01, *** p<.001)',
             fontweight='bold', fontsize=13)
ax.set_ylim(-0.25, 0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'B4_dystopian_corr_bar.png')


# ── B5: All 9 works × party — clean grouped bar ───────────────────────────────
print("\n── B5: ALL WORKS × PARTY (clean grouped bar) ──")
work_keys   = list(WORKS.keys())
work_labels = list(WORKS.values())

d_means = [dem[c].dropna().mean() for c in work_keys]
r_means = [rep[c].dropna().mean() for c in work_keys]
d_sems  = [dem[c].dropna().sem()  for c in work_keys]
r_sems  = [rep[c].dropna().sem()  for c in work_keys]
t_ps    = [ttest_ind(dem[c].dropna(), rep[c].dropna())[1]
           if dem[c].dropna().shape[0] >= 15 and rep[c].dropna().shape[0] >= 15 else np.nan
           for c in work_keys]

x = np.arange(len(work_keys)); w = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w/2, d_means, w, yerr=d_sems, capsize=4, color=blue, alpha=0.82,
       label='Democrats', edgecolor='black', linewidth=0.7)
ax.bar(x + w/2, r_means, w, yerr=r_sems, capsize=4, color=red,  alpha=0.82,
       label='Republicans', edgecolor='black', linewidth=0.7)

for i, p in enumerate(t_ps):
    if not np.isnan(p):
        s = sig_stars(p).replace('ns', '')
        if s:
            y_top = max(d_means[i] + d_sems[i], r_means[i] + r_sems[i]) + 0.1
            ax.text(i, y_top, s, ha='center', fontsize=13, fontweight='bold')

ax.axvline(5.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
ax.text(2.5, 5.35, 'TV / Movies', ha='center', fontsize=10, color='gray', style='italic')
ax.text(7.0, 5.35, 'Books',       ha='center', fontsize=10, color='gray', style='italic')
ax.set_xticks(x)
ax.set_xticklabels(work_labels, fontsize=10.5)
ax.set_ylabel('Mean Liking (1–5)', fontweight='bold')
ax.set_ylim(1, 5.7)
ax.set_title('Dystopian Media Liking: Democrats vs. Republicans\n'
             '(Error bars = SEM; * p<.05, ** p<.01, *** p<.001)',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'B5_liking_by_party_all.png')


# ── B6: Aversion item-level means by party ────────────────────────────────────
print("\n── B6: AVERSION ITEM MEANS BY PARTY ──")
# aversion1/2/3 are in the cleaned data (composite items)
aversion_items = {
    'aversion1': 'Would not befriend\nan out-partisan',
    'aversion2': 'Would stop spending\ntime with out-partisan',
    'aversion3': 'Like out-partisans\n(reverse-coded)',
}

available_items = {k: v for k, v in aversion_items.items() if k in df.columns}
if available_items:
    n_items = len(available_items)
    item_cols = list(available_items.keys())
    item_labels = list(available_items.values())

    d_means_av = [dem[c].dropna().mean() for c in item_cols]
    r_means_av = [rep[c].dropna().mean() for c in item_cols]
    d_sems_av  = [dem[c].dropna().sem()  for c in item_cols]
    r_sems_av  = [rep[c].dropna().sem()  for c in item_cols]

    x = np.arange(n_items); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, d_means_av, w, yerr=d_sems_av, capsize=5, color=blue, alpha=0.82,
           label='Democrats', edgecolor='black', linewidth=0.7)
    ax.bar(x + w/2, r_means_av, w, yerr=r_sems_av, capsize=5, color=red,  alpha=0.82,
           label='Republicans', edgecolor='black', linewidth=0.7)

    for i, (dm, rm, ds, rs) in enumerate(zip(d_means_av, r_means_av, d_sems_av, r_sems_av)):
        t, p = ttest_ind(dem[item_cols[i]].dropna(), rep[item_cols[i]].dropna())
        s = sig_stars(p).replace('ns', '')
        if s:
            ax.text(i, max(dm + ds, rm + rs) + 0.08, s, ha='center', fontsize=13, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(item_labels, fontsize=11)
    ax.set_ylabel('Mean Score (1–5)', fontweight='bold')
    ax.set_ylim(1, 5.5)
    ax.set_title('Social Aversion Items: Democrats vs. Republicans\n'
                 '(Higher = Greater aversion toward out-party)',
                 fontweight='bold', fontsize=13)
    ax.axhline(3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig(fig, 'B6_item_level_aversion.png')
else:
    print("  ⚠ aversion item columns not found — skipping B6")


# ── B7: Books-only mean liking ─────────────────────────────────────────────────
print("\n── B7: BOOKS-ONLY MEAN LIKING ──")
book_means = {BOOK_NAMES[c]: df[c].dropna().mean() for c in BOOK_COLS}
book_sems  = {BOOK_NAMES[c]: df[c].dropna().sem()  for c in BOOK_COLS}

sorted_books = sorted(book_means.items(), key=lambda x: x[1], reverse=True)
bnames = [x[0] for x in sorted_books]
bvals  = [x[1] for x in sorted_books]
bsems  = [book_sems[n] for n in bnames]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(bnames, bvals, yerr=bsems, capsize=5, color='#C4704F', alpha=0.85,
              edgecolor='black', linewidth=0.8, width=0.5)
for bar, val in zip(bars, bvals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + max(bsems) + 0.05,
            f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('Mean Liking (1–5)', fontweight='bold')
ax.set_ylim(1, 5.2)
ax.set_title('Mean Liking for Dystopian Books\n(Error bars = SEM)',
             fontweight='bold', fontsize=13)
ax.axhline(3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'B7_books_liking_bar.png')


# ── B8: TV/Movies-only mean liking ────────────────────────────────────────────
print("\n── B8: TV/MOVIES-ONLY MEAN LIKING ──")
movie_means = {MOVIE_NAMES[c]: df[c].dropna().mean() for c in MOVIE_COLS}
movie_sems  = {MOVIE_NAMES[c]: df[c].dropna().sem()  for c in MOVIE_COLS}

sorted_movies = sorted(movie_means.items(), key=lambda x: x[1], reverse=True)
mnames = [x[0] for x in sorted_movies]
mvals  = [x[1] for x in sorted_movies]
msems  = [movie_sems[n] for n in mnames]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(mnames, mvals, yerr=msems, capsize=5, color='#5B8DB8', alpha=0.85,
              edgecolor='black', linewidth=0.8, width=0.55)
for bar, val in zip(bars, mvals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + max(msems) + 0.05,
            f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('Mean Liking (1–5)', fontweight='bold')
ax.set_ylim(1, 5.2)
ax.set_title('Mean Liking for Dystopian TV Shows & Films\n(Error bars = SEM)',
             fontweight='bold', fontsize=13)
ax.axhline(3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'B8_movies_liking_bar.png')


# ── Done ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("✅  SUMMARIES & BAR CHARTS COMPLETE")
print(f"    All outputs saved to: {OUT_DIR}/")
print("=" * 80)
print("""
CSV tables:
  summary_descriptives.csv    — Mean, SD, n for all 9 works (overall & by party)
  summary_familiarity.csv     — % familiar with each work (overall & by party)
  summary_correlations.csv    — r, p, n for all dystopian × polarization pairs
  summary_party_ttests.csv    — t, p, Cohen's d: Dem vs Rep for each work
  summary_regression.csv      — β, SE, p, 95% CI for OLS models
  summary_highlow.csv         — High vs Low consumer group comparison

Bar charts:
  B1_mean_liking_overall.png  — Horizontal sorted bar: overall mean liking
  B2_familiarity_rates.png    — % familiar by party
  B3_polarization_by_party.png— Aversion / Othering / Moralizing: D vs R
  B4_dystopian_corr_bar.png   — r values: dystopian_index × 6 measures
  B5_liking_by_party_all.png  — All 9 works × party (clean grouped bar)
  B6_item_level_aversion.png  — Aversion item means: Dem vs Rep
  B7_books_liking_bar.png     — Books-only mean liking
  B8_movies_liking_bar.png    — TV/Movies-only mean liking
""")
