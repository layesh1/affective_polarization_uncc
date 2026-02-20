"""
WEEK 3 EXTENDED ANALYSES
UNC Charlotte Political Polarization Study - Fall 2025
Author: Lean

NEW ANALYSES:
1. Students vs Faculty Free Speech (separate out the two batteries)
2. Three Separate Affective Polarization Questions (moral identity, othering, aversion individually)
3. Independents' Free Speech Opinions
4. Weak partisans (leaners from partylean question) on Affective Polarization
5. Aversion Deep Dive (strongest effect, item-level analysis)
6. Religious Importance × Polarization / Free Speech
7. Scale verification for all new constructs

SUPPLEMENTAL (saved to exploratory/ folder):
- Dystopian media preferences by party
- Climate change beliefs by party/ideology
- Out-party trust (Q119, Q120, Q122)
- Partisan sorting (ideology × party)
- Government trust by race
- Racial fairness perceptions × party
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, pearsonr, spearmanr, chi2_contingency
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================
# PATHS — adjust if running from a different working directory
# ============================================================
DATA_PATH = 'Data_csv/POLS Lab Student Omnibus Fall 25- text.csv'
OUT_MAIN   = 'Week3_Analyses'
OUT_SUPP   = 'Supplemental_Exploratory'

os.makedirs(OUT_MAIN, exist_ok=True)
os.makedirs(OUT_SUPP, exist_ok=True)

print("="*80)
print("WEEK 3 EXTENDED POLARIZATION ANALYSES")
print("="*80)

# ============================================================
# LOAD & CLEAN
# ============================================================
df = pd.read_csv(DATA_PATH, skiprows=[1])
print(f"✓ Loaded: {len(df)} rows, {len(df.columns)} columns")

# ── Likert maps ──────────────────────────────────────────────
map5 = {'None at all': 1, 'A little': 2, 'A moderate amount': 3,
        'A lot': 4, 'A great deal': 5}
map7_agree = {'Strongly agree': 1, 'Agree': 2, 'Somewhat agree': 3,
              'Neither agree nor disagree': 4, 'Somewhat disagree': 5,
              'Disagree': 6, 'Strongly disagree': 7}
map7_imp = {'Not important at all': 1, 'Not very important': 2,
            'Somewhat important': 3, 'Very important': 4,
            'Extremely important': 5}   # for religion if 5-pt; adjust if 7-pt

def rev5(s):  return 6 - s
def rev7(s):  return 8 - s

# ── Party coding ─────────────────────────────────────────────
party_map = {'Strongly Democrat': 1, 'Somewhat Democrat': 2,
             'Not sure/neither one/other': 3,
             'Somewhat Republican': 4, 'Strongly Republican': 5}
partylean_map = {'Probably Democrats': 1, 'Still not sure/neither one/other': 2,
                 'Probably Republicans': 3}

if 'party' in df.columns:
    df['party_num'] = df['party'].map(party_map)
if 'partylean' in df.columns:
    df['partylean_num'] = df['partylean'].map(partylean_map)

# Combined party: leaners assigned to their lean
df['party_combined'] = df['party_num'].copy()
df.loc[(df['party_num'] == 3) & (df['partylean_num'] == 1), 'party_combined'] = 1.5  # Lean Dem
df.loc[(df['party_num'] == 3) & (df['partylean_num'] == 3), 'party_combined'] = 4.5  # Lean Rep

df['party_cat'] = pd.cut(df['party_combined'],
                         bins=[0, 2.5, 3.5, 6],
                         labels=['Democrat', 'Independent', 'Republican'])

# Leaner flags (the "weak" partisans who needed the second question)
df['is_leaner'] = df['partylean_num'].notna()
# Use None (object) instead of np.nan to avoid dtype promotion error with str
df['leaner_dir'] = np.where(df['partylean_num'] == 1, 'Lean Democrat',
                   np.where(df['partylean_num'] == 3, 'Lean Republican', None))
df['leaner_dir'] = df['leaner_dir'].astype(object)

# ── Convert key question batteries ──────────────────────────
# Aversion (Q135-Q140)
for q in ['Q135', 'Q136', 'Q137', 'Q138', 'Q139', 'Q140']:
    if q in df.columns:
        df[q] = df[q].map(map5)

# Moral / Othering
for q in ['moral1R','moral2R','moral3R','moral1D','moral2D','moral3D',
          'other1R','other2R','other3R','other1D','other2D','other3D']:
    if q in df.columns:
        df[q] = df[q].map(map5)

# Free speech (Q92-Q106)
for q in [f'Q{i}' for i in range(92, 107)]:
    if q in df.columns:
        df[q] = df[q].map(map7_agree)

# Trust battery (Q110-Q122)
for q in [f'Q{i}' for i in range(110, 123)]:
    if q in df.columns:
        df[q] = df[q].map(map7_agree)

# Religion importance — try common column names
relig_col = None
for col in ['Q_relig', 'religion_importance', 'Q164', 'Q165', 'relig']:
    if col in df.columns:
        relig_col = col
        break
# Also scan for anything with 'relig' in name
if relig_col is None:
    cands = [c for c in df.columns if 'relig' in c.lower()]
    if cands:
        relig_col = cands[0]
print(f"Religion column detected: {relig_col}")

# ── Aversion: apply reversals ────────────────────────────────
# High score = MORE aversion to out-party
df['Q135_s'] = df['Q135']               # Rep: "would NOT want Dem friend"
df['Q136_s'] = df['Q136']               # Rep: "stop spending time with Dems"
df['Q137_s'] = rev5(df['Q137'])         # Rep: "like some Dems" → REVERSED
df['Q138_s'] = df['Q138']               # Dem: "would NOT want Rep friend"
df['Q139_s'] = df['Q139']               # Dem: "stop spending time with Reps"
df['Q140_s'] = rev5(df['Q140'])         # Dem: "like some Reps" → REVERSED

# ── Moral identity indices ───────────────────────────────────
df['moral_R'] = df[['moral1R','moral2R','moral3R']].mean(axis=1)
df['moral_D'] = df[['moral1D','moral2D','moral3D']].mean(axis=1)

df['othering_R'] = df[['other1R','other2R','other3R']].mean(axis=1)
df['othering_D'] = df[['other1D','other2D','other3D']].mean(axis=1)

df['aversion_R'] = df[['Q135_s','Q136_s','Q137_s']].mean(axis=1)
df['aversion_D'] = df[['Q138_s','Q139_s','Q140_s']].mean(axis=1)

# Unified index per respondent (use D for Democrats, R for Republicans)
df['moral_idx']    = np.where(df['party_cat']=='Democrat', df['moral_D'],
                    np.where(df['party_cat']=='Republican', df['moral_R'], np.nan))
df['othering_idx'] = np.where(df['party_cat']=='Democrat', df['othering_D'],
                    np.where(df['party_cat']=='Republican', df['othering_R'], np.nan))
df['aversion_idx'] = np.where(df['party_cat']=='Democrat', df['aversion_D'],
                    np.where(df['party_cat']=='Republican', df['aversion_R'], np.nan))

df['affpol_idx'] = df[['moral_idx','othering_idx','aversion_idx']].mean(axis=1)

# ── Free speech: pro-freedom questions kept as-is (1=agree=low restriction)
#    Pro-restriction questions REVERSED so high = more restriction
freedom_qs     = ['Q92','Q100','Q103','Q104','Q105','Q106']   # direct
restriction_qs = ['Q95','Q96','Q97','Q98','Q99','Q101','Q102'] # need reversal

# Faculty-specific questions
faculty_freedom_qs     = ['Q92','Q100']
faculty_restriction_qs = ['Q95','Q96','Q97','Q98','Q99','Q101','Q102']

# Student-specific questions
student_freedom_qs     = ['Q103','Q104','Q105','Q106']
student_restriction_qs = []   # all student items are pro-freedom

for q in restriction_qs:
    if q in df.columns:
        df[f'{q}_s'] = rev7(df[q])
    else:
        df[f'{q}_s'] = np.nan

for q in freedom_qs:
    if q in df.columns:
        df[f'{q}_s'] = df[q]
    else:
        df[f'{q}_s'] = np.nan

# Faculty free speech restriction index
fac_items = [f'{q}_s' for q in faculty_freedom_qs + faculty_restriction_qs]
df['fs_faculty'] = df[[c for c in fac_items if c in df.columns]].mean(axis=1)

# Student free speech restriction index
stu_items = [f'{q}_s' for q in student_freedom_qs]
df['fs_student'] = df[[c for c in stu_items if c in df.columns]].mean(axis=1)

# Combined
all_items = [f'{q}_s' for q in freedom_qs + restriction_qs]
df['fs_combined'] = df[[c for c in all_items if c in df.columns]].mean(axis=1)

print(f"\nParty distribution:")
print(df['party_cat'].value_counts().sort_index())
print(f"Leaners: {df['is_leaner'].sum()} ({df['leaner_dir'].value_counts().to_dict()})")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def cohen_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)

def ttest_report(a, b, label_a='A', label_b='B'):
    t, p = ttest_ind(a.dropna(), b.dropna())
    d = cohen_d(a.dropna(), b.dropna())
    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
    print(f"  {label_a}: M={a.mean():.3f}, SD={a.std():.3f}, n={a.notna().sum()}")
    print(f"  {label_b}: M={b.mean():.3f}, SD={b.std():.3f}, n={b.notna().sum()}")
    print(f"  t({a.notna().sum()+b.notna().sum()-2:.0f}) = {t:.3f}, p = {p:.4f} {sig}, d = {d:.3f}")
    return t, p, d

def save_fig(fig, folder, fname, dpi=300):
    path = os.path.join(folder, fname)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved: {path}")
    plt.close(fig)


# ============================================================
# ANALYSIS 1: STUDENTS vs FACULTY FREE SPEECH — SEPARATED
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 1: STUDENT vs FACULTY FREE SPEECH ATTITUDES (Separated)")
print("="*80)
print("""
Scale: 1 = Support freedom (low restriction) → 7 = Support restriction (high restriction)
Faculty index: Q92, Q95-Q99, Q100-Q102  (faculty speech rights)
Student index: Q103-Q106                 (student speech rights)
""")

rep = df[df['party_cat']=='Republican']
dem = df[df['party_cat']=='Democrat']
ind = df[df['party_cat']=='Independent']

print("── FACULTY FREE SPEECH RESTRICTION INDEX ──")
ttest_report(dem['fs_faculty'], rep['fs_faculty'], 'Democrats', 'Republicans')

print("\n── STUDENT FREE SPEECH RESTRICTION INDEX ──")
ttest_report(dem['fs_student'], rep['fs_student'], 'Democrats', 'Republicans')

# Paired comparison: within-party, does faculty vs student differ?
print("\n── WITHIN-PARTY: Faculty vs Student Restriction Index ──")
for party_label, subdf in [('Democrats', dem), ('Republicans', rep)]:
    valid = subdf[['fs_faculty','fs_student']].dropna()
    t, p = stats.ttest_rel(valid['fs_faculty'], valid['fs_student'])
    print(f"\n  {party_label}:")
    print(f"    Faculty index: M={valid['fs_faculty'].mean():.3f}")
    print(f"    Student index: M={valid['fs_student'].mean():.3f}")
    print(f"    Paired t = {t:.3f}, p = {p:.4f} {'***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'}")

# Figure 1a: Faculty vs Student, by party
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Free Speech Restriction: Faculty vs. Student Rights by Party\n"
             "(1 = Support Freedom ← → 7 = Support Restriction)",
             fontsize=14, fontweight='bold')

categories = ['Faculty\nRights', 'Student\nRights']
colors = {'Democrat': '#0015BC', 'Republican': '#E81B23', 'Independent': '#808080'}

# Bar chart comparison
party_groups = [('Democrat', dem), ('Republican', rep), ('Independent', ind)]
x = np.arange(len(categories))
width = 0.25

ax = axes[0]
for i, (label, subdf) in enumerate(party_groups):
    means = [subdf['fs_faculty'].mean(), subdf['fs_student'].mean()]
    sems  = [subdf['fs_faculty'].sem(),  subdf['fs_student'].sem()]
    bars = ax.bar(x + (i-1)*width, means, width,
                  label=label, color=colors[label], alpha=0.8,
                  yerr=sems, capsize=4)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.08,
                f'{m:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel('Restriction Index (1=Low, 7=High)', fontweight='bold')
ax.set_title('Mean Restriction Support by Topic & Party')
ax.axhline(4, color='gray', linestyle='--', alpha=0.5, label='Neutral (4)')
ax.set_ylim(1, 5.5)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Scatter: faculty vs student per respondent
ax2 = axes[1]
for label, subdf in party_groups:
    valid = subdf[['fs_faculty','fs_student']].dropna()
    ax2.scatter(valid['fs_faculty'], valid['fs_student'],
                alpha=0.3, s=25, color=colors[label], label=label)

ax2.plot([1,7],[1,7], 'k--', alpha=0.3, label='Equal attitudes')
ax2.set_xlabel('Faculty Restriction Index', fontweight='bold')
ax2.set_ylabel('Student Restriction Index', fontweight='bold')
ax2.set_title('Individual: Faculty vs Student Restriction Attitudes')
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

save_fig(fig, OUT_MAIN, '01_faculty_vs_student_freespeech.png')

# Figure 1b: Item-level heatmap for each question
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Item-Level Free Speech Attitudes by Party\n"
             "(Mean restriction score per question; higher = more restriction)",
             fontsize=14, fontweight='bold')

question_labels = {
    'Q92_s': 'Q92: Faculty right\nto express views (F)',
    'Q95_s': 'Q95: No politics if\nprohibited (F)',
    'Q96_s': 'Q96: Refrain from\npolitics (F)',
    'Q97_s': 'Q97: No social equity\ndiscussion (F)',
    'Q98_s': 'Q98: No social media\nrights (F)',
    'Q99_s': 'Q99: No social media\nat all (F)',
    'Q100_s':'Q100: Constitutional\nright to comment (F)',
    'Q101_s':'Q101: No protest\non campus (F)',
    'Q102_s':'Q102: No protest\noff campus (F)',
    'Q103_s':'Q103: Students speak\nfreely (S)',
    'Q104_s':'Q104: Political\nclothing (S)',
    'Q105_s':'Q105: Protest on\ncampus (S)',
    'Q106_s':'Q106: Protest on/off\ncampus (S)',
}

all_fs_items = [f'{q}_s' for q in freedom_qs + restriction_qs if f'{q}_s' in df.columns]
labels = [question_labels.get(q, q) for q in all_fs_items]

for ax_idx, (party_label, subdf, cmap) in enumerate([
        ('Democrats', dem, 'Blues'), ('Republicans', rep, 'Reds')]):
    means = [subdf[q].mean() for q in all_fs_items]
    df_heat = pd.DataFrame({'Question': labels, 'Mean': means}).set_index('Question')
    sns.heatmap(df_heat, annot=True, fmt='.2f', cmap=cmap, vmin=1, vmax=7,
                ax=axes[ax_idx], cbar_kws={'label': 'Restriction Score'},
                linewidths=0.5)
    axes[ax_idx].set_title(f'{party_label} (n={len(subdf)})', fontweight='bold')
    axes[ax_idx].set_yticklabels(labels, rotation=0, fontsize=8)
    axes[ax_idx].set_xticklabels([])

save_fig(fig, OUT_MAIN, '01b_freespeech_item_heatmap.png')


# ============================================================
# ANALYSIS 2: THREE AFFECTIVE POLARIZATION COMPONENTS — SEPARATELY
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 2: THREE AFFECTIVE POLARIZATION COMPONENTS (Separate)")
print("="*80)

components = [
    ('moral_idx',    'Moral Identity',  'Party as moral foundation (1-5)'),
    ('othering_idx', 'Othering',        'Perceiving out-party as alien (1-5)'),
    ('aversion_idx', 'Social Aversion', 'Unwillingness to befriend out-party (1-5)'),
]

print("\n── DEMOCRATS vs REPUBLICANS on each component ──")
results = {}
for col, name, desc in components:
    print(f"\n{name} — {desc}")
    d_vals = dem[col].dropna()
    r_vals = rep[col].dropna()
    t, p, d = ttest_report(d_vals, r_vals, 'Democrats', 'Republicans')
    results[name] = {'dem_m': d_vals.mean(), 'dem_sd': d_vals.std(),
                     'rep_m': r_vals.mean(), 'rep_sd': r_vals.std(),
                     't': t, 'p': p, 'd': d}

# Figure 2: Three-panel component comparison
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("Three Components of Affective Polarization: Democrats vs. Republicans\n"
             "Scale: 1 = Low → 5 = High (♦ = Mean)", fontsize=14, fontweight='bold')

blue, red = '#0015BC', '#E81B23'

for ax, (col, name, desc) in zip(axes, components):
    d_vals = dem[col].dropna()
    r_vals = rep[col].dropna()

    bp = ax.boxplot([d_vals, r_vals],
                    labels=[f'Democrats\n(n={len(d_vals)})', f'Republicans\n(n={len(r_vals)})'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2.5))
    bp['boxes'][0].set_facecolor(blue);  bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor(red);   bp['boxes'][1].set_alpha(0.55)

    for i, (vals, color) in enumerate([(d_vals, blue), (r_vals, red)]):
        ax.plot(i+1, vals.mean(), marker='D', markersize=10,
                color=color, markeredgecolor='black', markeredgewidth=1, zorder=10)
        ax.text(i+1, vals.mean() + 0.18,
                f'M={vals.mean():.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    r = results[name]
    sig = '***' if r['p'] < .001 else '**' if r['p'] < .01 else '*' if r['p'] < .05 else 'ns'
    ax.set_title(f'{name}\nd = {r["d"]:.2f}, {sig}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Index Score (1–5)', fontweight='bold')
    ax.set_ylim(0.5, 5.5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

save_fig(fig, OUT_MAIN, '02_three_components_comparison.png')

# Figure 2b: Effect size comparison
fig, ax = plt.subplots(figsize=(8, 5))
names = [r[0] for r in [(n, results[n]) for n in ['Moral Identity','Othering','Social Aversion']]]
ds    = [results[n]['d'] for n in ['Moral Identity','Othering','Social Aversion']]
colors_bar = ['#4472C4' if d >= 0 else '#ED7D31' for d in ds]
bars = ax.barh(names, ds, color=colors_bar, alpha=0.8, edgecolor='black')
ax.axvline(0, color='black', linewidth=1)
ax.axvline(0.2,  color='gray', linestyle=':', alpha=0.5, label='Small (0.2)')
ax.axvline(0.5,  color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
ax.axvline(0.8,  color='gray', linestyle='-', alpha=0.5, label='Large (0.8)')
for bar, d in zip(bars, ds):
    ax.text(d + 0.02, bar.get_y() + bar.get_height()/2,
            f'd = {d:.2f}', va='center', fontweight='bold')
ax.set_xlabel("Cohen's d (Democrats − Republicans)", fontweight='bold')
ax.set_title("Effect Sizes: Which Component Drives Asymmetric Polarization?",
             fontweight='bold')
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, OUT_MAIN, '02b_effect_sizes_components.png')


# ============================================================
# ANALYSIS 3: INDEPENDENTS' FREE SPEECH OPINIONS
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 3: INDEPENDENTS' FREE SPEECH OPINIONS")
print("="*80)

print("""
Comparing TRUE Independents (no lean) to partisan groups.
Also comparing Lean Democrat vs Lean Republican Independents.
""")

true_ind = df[(df['party_cat']=='Independent') & (~df['is_leaner'])]
lean_dem  = df[df['leaner_dir']=='Lean Democrat']
lean_rep  = df[df['leaner_dir']=='Lean Republican']

print("── TRUE Independents vs Partisans ──")
print(f"True Independents: n={len(true_ind)}")

for idx, label in [('fs_faculty','Faculty Rights'), ('fs_student','Student Rights'), ('fs_combined','Combined')]:
    print(f"\n{label}:")
    print(f"  True Independents:  M={true_ind[idx].mean():.3f}, SD={true_ind[idx].std():.3f}")
    print(f"  Democrats:          M={dem[idx].mean():.3f}, SD={dem[idx].std():.3f}")
    print(f"  Republicans:        M={rep[idx].mean():.3f}, SD={rep[idx].std():.3f}")
    f, p = f_oneway(true_ind[idx].dropna(), dem[idx].dropna(), rep[idx].dropna())
    print(f"  One-way ANOVA: F={f:.3f}, p={p:.4f} {'***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'}")

print("\n── LEAN Democrat vs LEAN Republican on Free Speech ──")
if len(lean_dem) > 5 and len(lean_rep) > 5:
    for idx, label in [('fs_faculty','Faculty'), ('fs_student','Student'), ('fs_combined','Combined')]:
        t, p, d = ttest_report(lean_dem[idx], lean_rep[idx], 'Lean Democrat', 'Lean Republican')
        print()

# Figure 3: Free speech across all five groups
fig, ax = plt.subplots(figsize=(13, 6))

groups = [
    ('Strong Dem',    df[df['party_num']==1],  '#0015BC'),
    ('Soft Dem',      df[df['party_num']==2],  '#4169E1'),
    ('Lean Dem',      lean_dem,                '#87CEEB'),
    ('True Ind',      true_ind,                '#808080'),
    ('Lean Rep',      lean_rep,                '#FFA07A'),
    ('Soft Rep',      df[df['party_num']==4],  '#E81B23'),
    ('Strong Rep',    df[df['party_num']==5],  '#8B0000'),
]

faculty_means = [g[1]['fs_faculty'].mean() for g in groups]
student_means = [g[1]['fs_student'].mean() for g in groups]
group_labels  = [g[0] for g in groups]
group_colors  = [g[2] for g in groups]
x = np.arange(len(groups))
width = 0.35

b1 = ax.bar(x - width/2, faculty_means, width, label='Faculty Rights',
            color=group_colors, alpha=0.85, edgecolor='black', linewidth=1)
b2 = ax.bar(x + width/2, student_means, width, label='Student Rights',
            color=group_colors, alpha=0.45, edgecolor='black', linewidth=1, hatch='//')

ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=9)
ax.set_ylabel('Restriction Index (1=Freedom, 7=Restriction)', fontweight='bold')
ax.set_title("Free Speech Restriction Attitudes Across the Full Party Spectrum\n"
             "(Solid = Faculty Rights, Hatched = Student Rights)", fontweight='bold')
ax.axhline(4, color='gray', linestyle='--', alpha=0.5, label='Neutral (4)')

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='gray', label='Faculty Rights (solid)'),
                   Patch(facecolor='gray', alpha=0.4, hatch='//', label='Student Rights (hatched)'),
                   Patch(facecolor='gray', linestyle='--', label='Neutral = 4')]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_ylim(1, 6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, OUT_MAIN, '03_freespeech_full_spectrum.png')


# ============================================================
# ANALYSIS 4: WEAK PARTISANS (LEANERS) ON AFFECTIVE POLARIZATION
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 4: LEANERS vs STRONG PARTISANS ON AFFECTIVE POLARIZATION")
print("="*80)
print("""
Comparing:
  • Strong Democrat (party=1) vs Lean Democrat (partylean→Dem)
  • Strong Republican (party=5) vs Lean Republican (partylean→Rep)
  • Lean Democrats vs Lean Republicans
""")

strong_dem = df[df['party_num']==1]
soft_dem   = df[df['party_num']==2]
strong_rep = df[df['party_num']==5]
soft_rep   = df[df['party_num']==4]

partisan_groups = [
    ('Strong Democrat',  strong_dem,  'aversion_D', '#0015BC'),
    ('Soft Democrat',    soft_dem,    'aversion_D', '#4169E1'),
    ('Lean Democrat',    lean_dem,    'aversion_D', '#87CEEB'),
    ('Lean Republican',  lean_rep,    'aversion_R', '#FFA07A'),
    ('Soft Republican',  soft_rep,    'aversion_R', '#E81B23'),
    ('Strong Republican',strong_rep,  'aversion_R', '#8B0000'),
]

print("\n── AFFECTIVE POLARIZATION COMPONENTS across partisan strength ──")
for name, subdf, avcol, _ in partisan_groups:
    moral_col = 'moral_D' if 'Democrat' in name else 'moral_R'
    other_col = 'othering_D' if 'Democrat' in name else 'othering_R'
    m  = subdf[moral_col].mean()
    o  = subdf[other_col].mean()
    av = subdf[avcol].mean()
    overall = np.nanmean([m, o, av])
    print(f"  {name:22s} | n={len(subdf):3d} | Moral={m:.2f} | Other={o:.2f} | Aversion={av:.2f} | Overall={overall:.2f}")

print("\n── Lean Dem vs Lean Rep (aversion) ──")
if len(lean_dem) > 5 and len(lean_rep) > 5:
    ttest_report(lean_dem['aversion_D'], lean_rep['aversion_R'], 'Lean Dem', 'Lean Rep')

# Figure 4: Gradient chart across partisan strength
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("Affective Polarization by Partisan Strength\n"
             "(Does intensity of party ID predict polarization?)", fontsize=14, fontweight='bold')

component_cols = [
    ('Moral Identity',  ['moral_D','moral_D','moral_D','moral_R','moral_R','moral_R']),
    ('Othering',        ['othering_D','othering_D','othering_D','othering_R','othering_R','othering_R']),
    ('Social Aversion', ['aversion_D','aversion_D','aversion_D','aversion_R','aversion_R','aversion_R']),
]

group_names  = [g[0] for g in partisan_groups]
group_colors = [g[3] for g in partisan_groups]
group_dfs    = [g[1] for g in partisan_groups]

for ax, (comp_name, cols) in zip(axes, component_cols):
    means = [gdf[col].mean() for gdf, col in zip(group_dfs, cols)]
    sems  = [gdf[col].sem()  for gdf, col in zip(group_dfs, cols)]
    bars = ax.bar(range(len(group_names)), means, color=group_colors,
                  alpha=0.8, edgecolor='black', yerr=sems, capsize=4)
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Mean Score (1–5)', fontweight='bold')
    ax.set_title(comp_name, fontweight='bold', fontsize=12)
    ax.set_ylim(0.5, 5)
    ax.axhline(3, color='gray', linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

save_fig(fig, OUT_MAIN, '04_leaners_partisan_strength.png')


# ============================================================
# ANALYSIS 5: AVERSION DEEP DIVE (Strongest Effect)
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 5: AVERSION DEEP DIVE")
print("="*80)

print("""
Q135 (Rep): "I would NOT want to be friends with a Democrat"
Q136 (Rep): "If I found out a friend was a Democrat, I'd STOP spending time with them"
Q137 (Rep): "There are people I LIKE who are Democrats" ← REVERSED
Q138 (Dem): "I would NOT want to be friends with a Republican"
Q139 (Dem): "If I found out a friend was a Republican, I'd STOP spending time with them"
Q140 (Dem): "There are people I LIKE who are Republicans" ← REVERSED
""")

# Item-level analysis
print("── Item-level means (1=None at all, 5=A great deal of aversion) ──")
print("\nRepublicans toward Democrats:")
for q, label in [('Q135_s','Would not want Dem friend'),
                 ('Q136_s','Stop spending time with Dem friend'),
                 ('Q137_s','Like some Dems (reversed)')]:
    vals = rep[q].dropna()
    print(f"  {label}: M={vals.mean():.2f}, SD={vals.std():.2f}, n={len(vals)}")

print("\nDemocrats toward Republicans:")
for q, label in [('Q138_s','Would not want Rep friend'),
                 ('Q139_s','Stop spending time with Rep friend'),
                 ('Q140_s','Like some Reps (reversed)')]:
    vals = dem[q].dropna()
    print(f"  {label}: M={vals.mean():.2f}, SD={vals.std():.2f}, n={len(vals)}")

# Scale reliability for aversion
print("\n── Internal Consistency (Cronbach's α) ──")
def cronbach(df_sub, items):
    data = df_sub[items].dropna()
    if len(data) < 2: return np.nan
    n = len(items)
    item_vars = data.var(ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    return (n/(n-1)) * (1 - item_vars.sum()/total_var)

alpha_rep = cronbach(rep, ['Q135_s','Q136_s','Q137_s'])
alpha_dem = cronbach(dem, ['Q138_s','Q139_s','Q140_s'])
print(f"  Republicans: α = {alpha_rep:.3f}")
print(f"  Democrats:   α = {alpha_dem:.3f}")

# Breakdown: % of each response level
print("\n── Distribution of Aversion Responses (%) ──")
response_labels = {1:'None at all', 2:'A little', 3:'Moderate', 4:'A lot', 5:'A great deal'}
for party_label, subdf, items in [
    ('Republicans', rep, ['Q135_s','Q136_s','Q137_s']),
    ('Democrats',   dem, ['Q138_s','Q139_s','Q140_s'])]:
    pooled = pd.concat([subdf[q] for q in items]).dropna()
    print(f"\n  {party_label} (pooled across 3 items):")
    for val in [1,2,3,4,5]:
        pct = (pooled == val).mean() * 100
        bar = '█' * int(pct/2)
        print(f"    {response_labels[val]:14s}: {pct:5.1f}% {bar}")

# Figure 5a: Item-level grouped bar
fig, ax = plt.subplots(figsize=(13, 7))

items_rep = ['Q135_s','Q136_s','Q137_s']
items_dem = ['Q138_s','Q139_s','Q140_s']
item_labels_short = ["Would not\nwant friend", "Stop seeing\nfriend", "Like some\n(reversed)"]

rep_means = [rep[q].mean() for q in items_rep]
dem_means = [dem[q].mean() for q in items_dem]
rep_sems  = [rep[q].sem()  for q in items_rep]
dem_sems  = [dem[q].sem()  for q in items_dem]

x = np.arange(3)
width = 0.35
ax.bar(x - width/2, rep_means, width, label=f'Republicans (n={len(rep)})',
       color=red, alpha=0.8, yerr=rep_sems, capsize=5, edgecolor='black')
ax.bar(x + width/2, dem_means, width, label=f'Democrats (n={len(dem)})',
       color=blue, alpha=0.8, yerr=dem_sems, capsize=5, edgecolor='black')

for i, (rm, dm) in enumerate(zip(rep_means, dem_means)):
    ax.text(i - width/2, rm + 0.12, f'{rm:.2f}', ha='center', fontsize=10, fontweight='bold', color=red)
    ax.text(i + width/2, dm + 0.12, f'{dm:.2f}', ha='center', fontsize=10, fontweight='bold', color=blue)

ax.set_xticks(x)
ax.set_xticklabels(item_labels_short, fontsize=12)
ax.set_ylabel('Aversion Score (1=None at all, 5=A great deal)', fontweight='bold')
ax.set_title("Aversion Deep Dive: Item-Level Comparison\n"
             "Democrats show higher aversion on all three items", fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0.5, 4.5)
ax.axhline(3, color='gray', linestyle=':', alpha=0.4, label='Midpoint (3)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, OUT_MAIN, '05a_aversion_items.png')

# Figure 5b: Stacked distribution
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Aversion Response Distributions (Item-Level)\n"
             "Scale: 1=None at all → 5=A great deal", fontsize=14, fontweight='bold')

all_aversion_items = [
    (rep, 'Q135_s', 'Rep: Would NOT want\nDem friend',         red),
    (rep, 'Q136_s', 'Rep: Stop spending\ntime with Dems',      red),
    (rep, 'Q137_s', 'Rep: Like some Dems\n(REVERSED)',         red),
    (dem, 'Q138_s', 'Dem: Would NOT want\nRep friend',         blue),
    (dem, 'Q139_s', 'Dem: Stop spending\ntime with Reps',      blue),
    (dem, 'Q140_s', 'Dem: Like some Reps\n(REVERSED)',         blue),
]

for ax, (subdf, col, title, color) in zip(axes.flatten(), all_aversion_items):
    vals = subdf[col].dropna()
    counts = vals.value_counts().reindex([1,2,3,4,5], fill_value=0)
    pcts = counts / len(vals) * 100
    bars = ax.bar([1,2,3,4,5], pcts, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    for bar, pct in zip(bars, pcts):
        if pct > 3:
            ax.text(bar.get_x() + bar.get_width()/2, pct + 0.5,
                    f'{pct:.0f}%', ha='center', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Response (1=None → 5=A great deal)', fontsize=9)
    ax.set_ylabel('% of Respondents', fontsize=9)
    ax.set_xticks([1,2,3,4,5])
    ax.text(0.97, 0.97, f'M={vals.mean():.2f}\nSD={vals.std():.2f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

save_fig(fig, OUT_MAIN, '05b_aversion_distributions.png')


# ============================================================
# ANALYSIS 6: RELIGIOUS IMPORTANCE × POLARIZATION / FREE SPEECH
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 6: RELIGIOUS IMPORTANCE × AFFECTIVE POLARIZATION / FREE SPEECH")
print("="*80)

if relig_col is not None:
    # Try to map to numeric if text
    relig_text_maps = [
        {'Not at all important': 1, 'Not very important': 2, 'Somewhat important': 3,
         'Very important': 4, 'Extremely important': 5},
        {'Not important at all': 1, 'Not very important': 2, 'Somewhat important': 3,
         'Very important': 4, 'Extremely important': 5},
    ]
    df['relig_num'] = pd.to_numeric(df[relig_col], errors='coerce')
    if df['relig_num'].isna().all():
        for m in relig_text_maps:
            df['relig_num'] = df[relig_col].map(m)
            if df['relig_num'].notna().sum() > 50:
                break

    print(f"\nReligion variable: {relig_col}")
    print(f"  Valid responses: {df['relig_num'].notna().sum()}")
    print(f"  Range: {df['relig_num'].min()} – {df['relig_num'].max()}")
    print(f"  Mean: {df['relig_num'].mean():.2f}, SD: {df['relig_num'].std():.2f}")

    # Correlations within party
    print("\n── Correlation: Religious Importance × Affective Polarization ──")
    for party_label, subdf in [('Democrats', dem), ('Republicans', rep), ('All Partisans', df[df['party_cat'].isin(['Democrat','Republican'])])]:
        valid = subdf[['relig_num','aversion_idx','moral_idx','othering_idx','affpol_idx','fs_combined']].dropna()
        if len(valid) > 20:
            print(f"\n  {party_label} (n={len(valid)}):")
            for outcome, label in [('aversion_idx','Aversion'), ('moral_idx','Moral Identity'),
                                    ('othering_idx','Othering'), ('affpol_idx','Overall Polarization'),
                                    ('fs_combined','Free Speech Restriction')]:
                r, p = pearsonr(valid['relig_num'], valid[outcome])
                sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
                print(f"    × {label:25s}: r = {r:.3f}, p = {p:.4f} {sig}")

    # High vs low religiosity groups
    df['relig_group'] = pd.cut(df['relig_num'], bins=[0,2,3,6],
                               labels=['Low (1-2)', 'Moderate (3)', 'High (4-5)'])
    print("\n── Aversion Index by Religiosity (within party) ──")
    for party_label, subdf in [('Democrats', dem), ('Republicans', rep)]:
        print(f"\n  {party_label}:")
        for rg in ['Low (1-2)','Moderate (3)','High (4-5)']:
            sub = subdf[subdf['relig_group']==rg]['aversion_idx'].dropna()
            if len(sub) > 5:
                print(f"    {rg}: M={sub.mean():.2f}, SD={sub.std():.2f}, n={len(sub)}")

    # Figure 6: Scatter + boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Religious Importance × Polarization & Free Speech\n"
                 "(by Party Affiliation)", fontsize=14, fontweight='bold')

    for ax_idx, (outcome, label, ylim) in enumerate([
        ('aversion_idx',  'Social Aversion (1-5)',          (0.5, 5.5)),
        ('affpol_idx',    'Overall Polarization (1-5)',     (0.5, 5.5)),
        ('fs_combined',   'Free Speech Restriction (1-7)', (1, 7)),
    ]):
        ax = axes[ax_idx]
        for party_label, subdf, color in [('Democrats', dem, blue), ('Republicans', rep, red)]:
            valid = subdf[['relig_num', outcome]].dropna()
            if len(valid) > 10:
                ax.scatter(valid['relig_num'], valid[outcome],
                          alpha=0.25, s=20, color=color, label=party_label)
                # Regression line
                z = np.polyfit(valid['relig_num'], valid[outcome], 1)
                xl = np.linspace(valid['relig_num'].min(), valid['relig_num'].max(), 50)
                ax.plot(xl, np.poly1d(z)(xl), color=color, linewidth=2.5)
                r, p = pearsonr(valid['relig_num'], valid[outcome])
                sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
                print(f"  {party_label} × {label}: r={r:.3f} {sig}")

        ax.set_xlabel('Religious Importance (1=Low, 5=High)', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(label.split(' (')[0], fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    save_fig(fig, OUT_MAIN, '06_religion_polarization.png')

else:
    print("\n⚠️  No religion column detected. Check column name in your dataset.")
    print("   Common names: 'Q164', 'religion_importance', 'relig'")
    print("   Once identified, update `relig_col` at top of script.")


# ============================================================
# SCALE VERIFICATION SUMMARY
# ============================================================
print("\n" + "="*80)
print("SCALE VERIFICATION — ALL CONSTRUCTS")
print("="*80)

def verify_scale(df_sub, items, label):
    data = df_sub[items].dropna()
    corr = data.corr()
    upper = corr.values[np.triu_indices_from(corr.values, k=1)]
    alpha = cronbach(df_sub, items)
    print(f"\n  {label}:")
    print(f"    Items: {items}")
    print(f"    n = {len(data)}")
    print(f"    Mean inter-item r = {upper.mean():.3f}")
    print(f"    Min inter-item r  = {upper.min():.3f}")
    print(f"    Cronbach's α      = {alpha:.3f}")
    ok = "✓ PASS" if upper.min() > 0 and alpha > 0.5 else "⚠ REVIEW"
    print(f"    Status: {ok}")

verify_scale(rep, ['Q135_s','Q136_s','Q137_s'], "Republicans: Aversion")
verify_scale(dem, ['Q138_s','Q139_s','Q140_s'], "Democrats: Aversion")
verify_scale(rep, ['moral1R','moral2R','moral3R'], "Republicans: Moral Identity")
verify_scale(dem, ['moral1D','moral2D','moral3D'], "Democrats: Moral Identity")
verify_scale(rep, ['other1R','other2R','other3R'], "Republicans: Othering")
verify_scale(dem, ['other1D','other2D','other3D'], "Democrats: Othering")

fac_qs_avail = [q+'_s' for q in faculty_freedom_qs + faculty_restriction_qs if q+'_s' in df.columns]
stu_qs_avail = [q+'_s' for q in student_freedom_qs if q+'_s' in df.columns]
verify_scale(df, fac_qs_avail, "All: Faculty Free Speech Restriction Index")
verify_scale(df, stu_qs_avail, "All: Student Free Speech Restriction Index")


print("\n" + "="*80)
print("✅ ALL MAIN ANALYSES COMPLETE")
print(f"   Figures saved to: {OUT_MAIN}/")
print("="*80)