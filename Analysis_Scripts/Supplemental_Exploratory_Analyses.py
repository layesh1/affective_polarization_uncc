"""
SUPPLEMENTAL EXPLORATORY ANALYSES
UNC Charlotte Political Polarization Study - Fall 2025
Author: Lean

Analyses that may be useful for thesis discussion, limitations section,
or future papers — not primary RQs but potentially interesting/reviewable.

Topics:
1. Partisan sorting (ideology × party correlation)
2. Dystopian media preferences by party
3. Climate change beliefs by party & ideology
4. Out-party trust (Q119, Q120, Q122)
5. Government trust by race
6. Racial fairness perceptions × party
7. Gender × polarization
8. Full feeling thermometer breakdown (Q148, Q149) with party × ideology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, pearsonr, spearmanr, chi2_contingency
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

DATA_PATH = 'Data_csv/POLS Lab Student Omnibus Fall 25- text.csv'
OUT_DIR   = 'Supplemental_Exploratory'
os.makedirs(OUT_DIR, exist_ok=True)

print("="*80)
print("SUPPLEMENTAL EXPLORATORY ANALYSES")
print("="*80)

# ── Load & prep (mirrors main script) ──────────────────────
df = pd.read_csv(DATA_PATH, skiprows=[1])

map5 = {'None at all':1,'A little':2,'A moderate amount':3,'A lot':4,'A great deal':5}
map7 = {'Strongly agree':1,'Agree':2,'Somewhat agree':3,'Neither agree nor disagree':4,
        'Somewhat disagree':5,'Disagree':6,'Strongly disagree':7}

party_map = {'Strongly Democrat':1,'Somewhat Democrat':2,
             'Not sure/neither one/other':3,'Somewhat Republican':4,'Strongly Republican':5}
partylean_map = {'Probably Democrats':1,'Still not sure/neither one/other':2,'Probably Republicans':3}

df['party_num']    = df['party'].map(party_map) if 'party' in df.columns else np.nan
df['partylean_num']= df['partylean'].map(partylean_map) if 'partylean' in df.columns else np.nan
df['party_combined']= df['party_num'].copy()
df.loc[(df['party_num']==3)&(df['partylean_num']==1),'party_combined'] = 1.5
df.loc[(df['party_num']==3)&(df['partylean_num']==3),'party_combined'] = 4.5

df['party_cat'] = pd.cut(df['party_combined'], bins=[0,2.5,3.5,6],
                          labels=['Democrat','Independent','Republican'])

# Ideology
ideo_map = {'Very liberal':1,'Liberal':2,'Somewhat liberal':3,'Moderate':4,
            'Somewhat conservative':5,'Conservative':6,'Very conservative':7}
df['ideo_num'] = df['ideology'].map(ideo_map) if 'ideology' in df.columns else np.nan

# Free speech
def rev7(s): return 8 - s
for q in ['Q92','Q100','Q103','Q104','Q105','Q106']:
    if q in df.columns: df[f'{q}_s'] = df[q].map(map7)
for q in ['Q95','Q96','Q97','Q98','Q99','Q101','Q102']:
    if q in df.columns: df[f'{q}_s'] = rev7(df[q].map(map7))

# Trust
for q in [f'Q{i}' for i in range(110,123)]:
    if q in df.columns: df[q] = df[q].map(map7)

# Feeling thermometers
df['FT_R'] = pd.to_numeric(df['Q148'], errors='coerce') if 'Q148' in df.columns else np.nan
df['FT_D'] = pd.to_numeric(df['Q149'], errors='coerce') if 'Q149' in df.columns else np.nan

# Aversion
def rev5(s): return 6 - s
for q in ['Q135','Q136','Q137','Q138','Q139','Q140']:
    if q in df.columns: df[q] = df[q].map(map5)
df['Q137_s'] = rev5(df['Q137']); df['Q140_s'] = rev5(df['Q140'])
df['Q135_s'] = df['Q135']; df['Q136_s'] = df['Q136']
df['Q138_s'] = df['Q138']; df['Q139_s'] = df['Q139']

df['aversion_R'] = df[['Q135_s','Q136_s','Q137_s']].mean(axis=1)
df['aversion_D'] = df[['Q138_s','Q139_s','Q140_s']].mean(axis=1)
df['aversion_idx'] = np.where(df['party_cat']=='Democrat', df['aversion_D'],
                    np.where(df['party_cat']=='Republican', df['aversion_R'], np.nan))

# Moral / othering
for q in ['moral1R','moral2R','moral3R','moral1D','moral2D','moral3D',
          'other1R','other2R','other3R','other1D','other2D','other3D']:
    if q in df.columns: df[q] = df[q].map(map5)

df['moral_R'] = df[['moral1R','moral2R','moral3R']].mean(axis=1)
df['moral_D'] = df[['moral1D','moral2D','moral3D']].mean(axis=1)

# Affective polarization (FT-based)
df['FT_inparty']  = np.where(df['party_cat']=='Democrat', df['FT_D'],
                    np.where(df['party_cat']=='Republican', df['FT_R'], np.nan))
df['FT_outparty'] = np.where(df['party_cat']=='Democrat', df['FT_R'],
                    np.where(df['party_cat']=='Republican', df['FT_D'], np.nan))
df['affpol_FT'] = df['FT_inparty'] - df['FT_outparty']

# Race
def cat_race(s):
    if pd.isna(s): return 'Missing'
    s = str(s).lower()
    if 'white' in s and ',' not in s: return 'White'
    elif 'black' in s or 'african' in s: return 'Black/African American'
    elif 'hispanic' in s or 'latino' in s: return 'Hispanic/Latino'
    elif 'asian' in s: return 'Asian'
    elif ',' in s: return 'Multiracial'
    else: return 'Other'

df['race'] = df['Q62'].apply(cat_race) if 'Q62' in df.columns else 'Unknown'
df['gender'] = df['Q63'] if 'Q63' in df.columns else np.nan

rep = df[df['party_cat']=='Republican']
dem = df[df['party_cat']=='Democrat']
ind = df[df['party_cat']=='Independent']

blue, red, gray = '#0015BC', '#E81B23', '#808080'

def save_fig(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# S1: PARTISAN SORTING
# ────────────────────────────────────────────────────────────
print("\n── S1: PARTISAN SORTING ──")
valid = df[df['party_num'].notna() & df['ideo_num'].notna()]
r, p = spearmanr(valid['party_num'], valid['ideo_num'])
print(f"  Spearman r_s = {r:.3f}, p = {p:.4f} (n={len(valid)})")
print(f"  Interpretation: {'Strong' if abs(r)>.7 else 'Moderate' if abs(r)>.4 else 'Weak'} partisan sorting")

fig, ax = plt.subplots(figsize=(11, 7))
ideo_order = ['Very liberal','Liberal','Somewhat liberal','Moderate',
              'Somewhat conservative','Conservative','Very conservative']
party_order = ['Strongly Democrat','Somewhat Democrat','Not sure/neither one/other',
               'Somewhat Republican','Strongly Republican']

ct = pd.crosstab(df['party'], df['ideology'])
ct = ct.reindex([p for p in party_order if p in ct.index])
ct = ct.reindex(columns=[i for i in ideo_order if i in ct.columns])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

sns.heatmap(ct_pct, annot=True, fmt='.0f', cmap='RdBu_r', vmin=0, vmax=60,
            linewidths=0.5, ax=ax, cbar_kws={'label': '% within party row'})
ax.set_title(f"Partisan Sorting: Party × Ideology (Spearman r={r:.2f}***)\n"
             f"Values = % of each party row", fontweight='bold')
ax.set_xlabel('Ideological Self-Placement', fontweight='bold')
ax.set_ylabel('Party Identification', fontweight='bold')
plt.xticks(rotation=40, ha='right')
plt.yticks(rotation=0)
save_fig(fig, 'S1_partisan_sorting.png')


# ────────────────────────────────────────────────────────────
# S2: DYSTOPIAN MEDIA BY PARTY
# ────────────────────────────────────────────────────────────
print("\n── S2: DYSTOPIAN MEDIA PREFERENCES BY PARTY ──")
like_map = {'Dislike a great deal':1,'Dislike somewhat':2,'Neither like nor dislike':3,
            'Like somewhat':4,'Like a great deal':5}
media_vars = ['Hunger','1984','BlackMirror'] if all(v in df.columns for v in ['Hunger','1984','BlackMirror']) else []
media_vars += [c for c in df.columns if any(k in c.lower() for k in ['hunger','1984','blackmirror','handmaid','brave'])
               if c not in media_vars]

media_results = {}
for v in media_vars[:6]:
    if v in df.columns:
        col_name = f'{v}_n'
        df[col_name] = df[v].map(like_map)
        # Re-slice after adding column so dem/rep see it
        d  = df[df['party_cat']=='Democrat'][col_name].dropna()
        r2 = df[df['party_cat']=='Republican'][col_name].dropna()
        if len(d) > 15 and len(r2) > 15:
            t, p = ttest_ind(d, r2)
            media_results[v] = {'dem_m': d.mean(), 'rep_m': r2.mean(), 't': t, 'p': p}
            sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
            print(f"  {v}: Dem={d.mean():.2f}, Rep={r2.mean():.2f}, {sig}")

if media_results:
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(media_results.keys())
    dem_m = [media_results[n]['dem_m'] for n in names]
    rep_m = [media_results[n]['rep_m'] for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, dem_m, w, color=blue, alpha=0.8, label='Democrats', edgecolor='black')
    ax.bar(x+w/2, rep_m, w, color=red,  alpha=0.8, label='Republicans', edgecolor='black')
    for i, n in enumerate(names):
        p = media_results[n]['p']
        sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
        if sig: ax.text(i, max(dem_m[i], rep_m[i])+0.1, sig, ha='center', fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Liking (1=Dislike, 5=Like)', fontweight='bold')
    ax.set_title('Dystopian Media Preferences by Party', fontweight='bold')
    ax.legend(); ax.set_ylim(1, 6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_fig(fig, 'S2_dystopian_media.png')


# ────────────────────────────────────────────────────────────
# S3: CLIMATE CHANGE BY PARTY & IDEOLOGY
# ────────────────────────────────────────────────────────────
print("\n── S3: CLIMATE CHANGE BELIEFS ──")
climate_col = 'Q167' if 'Q167' in df.columns else None
if climate_col:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Climate Change Beliefs: Party & Ideology", fontsize=14, fontweight='bold')

    for ax_idx, (group_col, title, order) in enumerate([
        ('party_cat',  'By Party',    ['Democrat','Independent','Republican']),
        ('ideology',   'By Ideology', ['Very liberal','Liberal','Somewhat liberal','Moderate',
                                        'Somewhat conservative','Conservative','Very conservative']),
    ]):
        ct = pd.crosstab(df[group_col], df[climate_col], normalize='index') * 100
        target_resp = 'Caused mostly by human activities'
        if target_resp in ct.columns:
            vals = ct[target_resp].reindex([o for o in order if o in ct.index])
            colors_bar = [blue if 'Dem' in str(i) or 'liberal' in str(i).lower()
                          else red if 'Rep' in str(i) or 'conservative' in str(i).lower()
                          else gray for i in vals.index]
            axes[ax_idx].barh(range(len(vals)), vals.values, color=colors_bar, alpha=0.8, edgecolor='black')
            axes[ax_idx].set_yticks(range(len(vals)))
            axes[ax_idx].set_yticklabels(vals.index, fontsize=9)
            axes[ax_idx].set_xlabel('% Believing Human-Caused Climate Change', fontweight='bold')
            axes[ax_idx].set_title(title, fontweight='bold')
            for i, v in enumerate(vals.values):
                axes[ax_idx].text(v + 0.5, i, f'{v:.0f}%', va='center', fontsize=9)
            axes[ax_idx].spines['top'].set_visible(False)
            axes[ax_idx].spines['right'].set_visible(False)
    save_fig(fig, 'S3_climate_change.png')


# ────────────────────────────────────────────────────────────
# S4: OUT-PARTY TRUST
# ────────────────────────────────────────────────────────────
print("\n── S4: OUT-PARTY TRUST (Q119, Q120, Q122) ──")
trust_questions = {
    'Q119': '"Opposite party cannot be trusted in government"',
    'Q120': '"Political divisions make trust difficult"',
    'Q122': '"Frustrated/angry with opposite party in power"',
}
# These are pro-distrust items: 1=Strongly agree (high distrust), 7=Strongly disagree
# REVERSE so high = high distrust
for q in ['Q119','Q120','Q122']:
    if q in df.columns:
        df[f'{q}_distrust'] = rev7(df[q])  # now 7=high distrust

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Out-Party Distrust: Democrats vs Republicans\n"
             "(After reversal: 7 = Maximum distrust)", fontsize=14, fontweight='bold')

for ax, (q, desc) in zip(axes, trust_questions.items()):
    col = f'{q}_distrust'
    if col not in df.columns: continue
    d_vals = df[df['party_cat']=='Democrat'][col].dropna()
    r_vals = df[df['party_cat']=='Republican'][col].dropna()
    t, p = ttest_ind(d_vals, r_vals)
    sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
    bp = ax.boxplot([d_vals, r_vals],
                    labels=[f'Dem\n(n={len(d_vals)})', f'Rep\n(n={len(r_vals)})'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(blue); bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(red);  bp['boxes'][1].set_alpha(0.5)
    for i, vals in enumerate([d_vals, r_vals]):
        ax.plot(i+1, vals.mean(), 'D', markersize=9, color=['#002299','#990000'][i],
                markeredgecolor='black', zorder=10)
        ax.text(i+1, vals.mean()+0.2, f'M={vals.mean():.2f}',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'{q}: {sig}\n{desc[:40]}...', fontsize=9, fontweight='bold')
    ax.set_ylabel('Distrust Score (1-7)')
    ax.set_ylim(0.5, 7.5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

save_fig(fig, 'S4_outparty_distrust.png')


# ────────────────────────────────────────────────────────────
# S5: GOVERNMENT TRUST BY RACE
# ────────────────────────────────────────────────────────────
print("\n── S5: GOVERNMENT TRUST BY RACE ──")
trust_gov_map = {'Strongly agree':7,'Agree':6,'Somewhat agree':5,
                 'Neither agree nor disagree':4,'Somewhat disagree':3,
                 'Disagree':2,'Strongly disagree':1}
for q in ['Q110','Q112']:
    if q in df.columns:
        df[f'{q}_trust'] = df[q].map(trust_gov_map) if df[q].dtype == object else df[q]

main_races = ['White','Black/African American','Hispanic/Latino','Asian']

print("  Federal govt trust by race:")
race_trust_data = {}
for r_label in main_races:
    sub = df[df['race']==r_label]['Q110_trust'].dropna() if 'Q110_trust' in df.columns else pd.Series()
    if len(sub) > 10:
        race_trust_data[r_label] = sub
        print(f"    {r_label}: M={sub.mean():.2f}, n={len(sub)}")

if len(race_trust_data) >= 2:
    f_stat, p_val = f_oneway(*race_trust_data.values())
    print(f"  ANOVA: F={f_stat:.3f}, p={p_val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    race_colors = ['#4472C4','#ED7D31','#A9D18E','#FF0000']
    bp = ax.boxplot(list(race_trust_data.values()),
                    labels=list(race_trust_data.keys()),
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], race_colors[:len(race_trust_data)]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(4, color='gray', linestyle='--', alpha=0.5, label='Neutral (4)')
    ax.set_ylabel('Federal Govt Trust (1=Low, 7=High)', fontweight='bold')
    ax.set_title('Trust in Federal Government by Race/Ethnicity\n'
                 f'ANOVA: F={f_stat:.2f}, p={p_val:.3f}', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_fig(fig, 'S5_govtrust_by_race.png')


# ────────────────────────────────────────────────────────────
# S6: RACIAL FAIRNESS PERCEPTIONS × PARTY
# ────────────────────────────────────────────────────────────
print("\n── S6: RACIAL FAIRNESS PERCEPTIONS × PARTY ──")
fairness_qs = {'Q115':'Govt treats all races fairly',
               'Q116':'My race well represented',
               'Q117':'Policies benefit all races equally',
               'Q118':'Govt responds fairly to all communities'}
for q in fairness_qs:
    if q in df.columns:
        df[f'{q}_n'] = df[q].map(trust_gov_map) if df[q].dtype == object else df[q]

fairness_cols = [f'{q}_n' for q in fairness_qs if f'{q}_n' in df.columns]
if fairness_cols:
    df['fairness_idx'] = df[fairness_cols].mean(axis=1)

    print("  Racial Fairness Index by race × party:")
    for r_label in ['White','Black/African American','Hispanic/Latino']:
        for party_label, subdf in [('Dem',dem),('Rep',rep)]:
            sub = subdf[subdf['race']==r_label]['fairness_idx'].dropna()
            if len(sub) > 5:
                print(f"    {r_label} {party_label}: M={sub.mean():.2f}, n={len(sub)}")

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_data = []
    for r_label in ['White','Black/African American','Hispanic/Latino']:
        for party_label, subdf, color in [('Democrat',dem,blue),('Republican',rep,red)]:
            sub = subdf[subdf['race']==r_label]['fairness_idx'].dropna()
            if len(sub) > 5:
                plot_data.append({'Race': r_label, 'Party': party_label,
                                  'Mean': sub.mean(), 'SEM': sub.sem(),
                                  'Color': color})

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        x_pos = {}
        offset = {'Democrat': -0.2, 'Republican': 0.2}
        race_pos = {'White': 0, 'Black/African American': 1, 'Hispanic/Latino': 2}
        for _, row in plot_df.iterrows():
            x = race_pos.get(row['Race'], 0) + offset.get(row['Party'], 0)
            ax.bar(x, row['Mean'], 0.35, color=row['Color'], alpha=0.8,
                   yerr=row['SEM'], capsize=5, edgecolor='black',
                   label=row['Party'] if row['Race']=='White' else '')
            ax.text(x, row['Mean'] + 0.1, f"{row['Mean']:.2f}",
                    ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['White', 'Black/\nAfrican American', 'Hispanic/\nLatino'], fontsize=11)
        ax.set_ylabel('Racial Fairness Index (1=Low, 7=High)', fontweight='bold')
        ax.set_title('Perceptions of Racial Fairness in Government: Race × Party\n'
                     '(Higher = More belief government treats races fairly)', fontweight='bold')
        ax.axhline(4, color='gray', linestyle='--', alpha=0.5, label='Neutral (4)')
        ax.legend(fontsize=10)
        ax.set_ylim(1, 7)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        save_fig(fig, 'S6_racial_fairness_party.png')


# ────────────────────────────────────────────────────────────
# S7: GENDER × POLARIZATION
# ────────────────────────────────────────────────────────────
print("\n── S7: GENDER × AFFECTIVE POLARIZATION ──")
# Re-slice here so gender subsets see all columns added above
men   = df[df['gender']=='Man']
women = df[df['gender']=='Woman']

for label, col in [('Aversion','aversion_idx'), ('FT-based Polarization','affpol_FT')]:
    men_v   = men[col].dropna();   women_v = women[col].dropna()
    if len(men_v) > 20 and len(women_v) > 20:
        t, p = ttest_ind(men_v, women_v)
        d_val = (men_v.mean()-women_v.mean()) / np.sqrt((men_v.std()**2+women_v.std()**2)/2)
        sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
        print(f"  {label}: Men M={men_v.mean():.2f} vs Women M={women_v.mean():.2f}, t={t:.3f}, {sig}, d={d_val:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Gender × Polarization", fontsize=14, fontweight='bold')

for ax, (col, label) in zip(axes, [('aversion_idx','Social Aversion (1-5)'),
                                     ('affpol_FT','FT-based Polarization (0-100)')]):
    data_plot = [g[col].dropna() for g in [men, women] if g[col].notna().sum() > 5]
    labels_bp = [f'{lbl}\n(n={len(g[col].dropna())})' for lbl, g in
                 [('Men', men), ('Women', women)] if g[col].notna().sum() > 5]
    bp = ax.boxplot(data_plot, labels=labels_bp, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], ['#6495ED','#FF69B4']):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(label.split(' (')[0], fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

save_fig(fig, 'S7_gender_polarization.png')


# ────────────────────────────────────────────────────────────
# S8: FEELING THERMOMETERS FULL BREAKDOWN
# ────────────────────────────────────────────────────────────
print("\n── S8: FEELING THERMOMETERS — Full Party × Ideology Breakdown ──")
ideo_groups = ['Very liberal','Liberal','Somewhat liberal','Moderate',
               'Somewhat conservative','Conservative','Very conservative']

ft_data = df[df['ideology'].isin(ideo_groups)].copy()
ft_means = ft_data.groupby(['party_cat','ideology'])[['FT_D','FT_R']].mean()
print(ft_means.round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feeling Thermometers: Party × Ideology\n"
             "(How warm each group feels toward Democrats and Republicans)", fontsize=13, fontweight='bold')

for ax, (col, title, cmap) in zip(axes, [('FT_D','Warmth toward Democrats','Blues'),
                                          ('FT_R','Warmth toward Republicans','Reds')]):
    pivot = ft_data.pivot_table(values=col, index='ideology', columns='party_cat', aggfunc='mean')
    pivot = pivot.reindex([i for i in ideo_groups if i in pivot.index])
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap=cmap, vmin=20, vmax=90,
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Mean FT Score'})
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Party', fontweight='bold')
    ax.set_ylabel('Ideology', fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

save_fig(fig, 'S8_feeling_thermometers_party_ideo.png')


print("\n" + "="*80)
print("✅ SUPPLEMENTAL ANALYSES COMPLETE")
print(f"   All figures saved to: {OUT_DIR}/")
print("="*80)
print("""
Files generated:
  S1_partisan_sorting.png             — Party × ideology heat map
  S2_dystopian_media.png              — Media preferences by party
  S3_climate_change.png               — Climate beliefs by party/ideology
  S4_outparty_distrust.png            — Q119, Q120, Q122 distrust items
  S5_govtrust_by_race.png             — Federal govt trust by race
  S6_racial_fairness_party.png        — Racial fairness × party interaction
  S7_gender_polarization.png          — Gender × polarization
  S8_feeling_thermometers_party_ideo.png — Full FT breakdown
""")