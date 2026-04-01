"""
Final Thesis Figures - Simplified for Non-Technical Audience
Corrected data values verified against cleaned dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
OUTPUT_DIR = 'Thesis_Figures_Final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
BLUE = '#4878CF'    # Democrats
RED = '#D65F5F'     # Republicans
GRAY = '#999999'    # Independents
LIGHT_BLUE = '#A8C4E0'
LIGHT_RED = '#E8B4B4'

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load data
df = pd.read_csv('Cleaned_Data/polarization_cleaned_CORRECTED.csv')
rep = df[df['party_category'] == 'Republican'].copy()
dem = df[df['party_category'] == 'Democrat'].copy()
ind = df[df['party_category'] == 'Independent'].copy()

print(f"Loaded: {len(df)} total, {len(dem)} Dem, {len(rep)} Rep, {len(ind)} Ind")

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1-1)*g1.std(ddof=1)**2 + (n2-1)*g2.std(ddof=1)**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled_std

# ============================================================================
# FIGURE 1: UMAP Projection (simplified)
# ============================================================================
print("\nFigure 1: UMAP - Skipping (requires UMAP recomputation, keep existing)")
# We'll regenerate this from the existing script if needed; for now keep existing.

# ============================================================================
# FIGURE 2: PCA Biplot (simplified)
# ============================================================================
print("Figure 2: PCA - Skipping (requires PCA recomputation, keep existing)")

# ============================================================================
# FIGURE 3: Overall Affective Polarization - BAR CHART (replaces violin)
# ============================================================================
print("Figure 3: Overall Polarization Bar Chart")

dem_ap = dem['affective_polarization_D'].dropna()
rep_ap = rep['affective_polarization_R'].dropna()
d_overall = cohens_d(dem_ap, rep_ap)

fig, ax = plt.subplots(figsize=(8, 6))
means = [dem_ap.mean(), rep_ap.mean()]
sds = [dem_ap.std(ddof=1), rep_ap.std(ddof=1)]
ses = [sds[0]/np.sqrt(len(dem_ap)), sds[1]/np.sqrt(len(rep_ap))]
ci95 = [1.96 * se for se in ses]

bars = ax.bar(['Democrats\n(n = {})'.format(len(dem_ap)),
               'Republicans\n(n = {})'.format(len(rep_ap))],
              means, yerr=ci95, capsize=8, color=[BLUE, RED],
              edgecolor='white', linewidth=1.5, width=0.5)

ax.set_ylabel('Average Polarization Score\n(1 = Low, 5 = High)')
ax.set_title('How Emotionally Distant Are Students\nFrom the Other Party?',
             fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(1, 4.0)
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.text(1.05, 3.02, 'Scale midpoint', color='gray', fontsize=10, ha='left')

# Add value labels
for bar, mean, ci in zip(bars, means, ci95):
    ax.text(bar.get_x() + bar.get_width()/2., mean + ci + 0.05,
            f'{mean:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Add effect size annotation
ax.annotate(f'Democrats score significantly higher\n(effect size d = {abs(d_overall):.2f}, p < .001)',
            xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=11,
            style='italic', color='#444444')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig3_polarization_bar.png')
plt.close()
print(f"  Saved. Dem M={dem_ap.mean():.3f}, Rep M={rep_ap.mean():.3f}, d={d_overall:.4f}")

# ============================================================================
# FIGURE 4: Component Decomposition Bar Chart (CORRECTED d values)
# ============================================================================
print("Figure 4: Component Decomposition")

dem_moral = dem['moral_index_D'].dropna()
rep_moral = rep['moral_index_R'].dropna()
dem_other = dem['othering_index_D'].dropna()
rep_other = rep['othering_index_R'].dropna()
dem_aversion = dem['aversion_index_D'].dropna()
rep_aversion = rep['aversion_index_R'].dropna()

d_moral = cohens_d(dem_moral, rep_moral)
d_other = cohens_d(dem_other, rep_other)
d_aversion = cohens_d(dem_aversion, rep_aversion)

fig, ax = plt.subplots(figsize=(10, 6))

components = ['Moral Identity\n(sees party as\nmoral cause)',
              'Othering\n(sees out-party\nas alien)',
              'Social Aversion\n(avoids out-party\nmembers)']
dem_means = [dem_moral.mean(), dem_other.mean(), dem_aversion.mean()]
rep_means = [rep_moral.mean(), rep_other.mean(), rep_aversion.mean()]
dem_ci = [1.96 * dem_moral.std(ddof=1)/np.sqrt(len(dem_moral)),
          1.96 * dem_other.std(ddof=1)/np.sqrt(len(dem_other)),
          1.96 * dem_aversion.std(ddof=1)/np.sqrt(len(dem_aversion))]
rep_ci = [1.96 * rep_moral.std(ddof=1)/np.sqrt(len(rep_moral)),
          1.96 * rep_other.std(ddof=1)/np.sqrt(len(rep_other)),
          1.96 * rep_aversion.std(ddof=1)/np.sqrt(len(rep_aversion))]

x = np.arange(len(components))
width = 0.35

bars1 = ax.bar(x - width/2, dem_means, width, yerr=dem_ci, capsize=6,
               label='Democrats', color=BLUE, edgecolor='white', linewidth=1)
bars2 = ax.bar(x + width/2, rep_means, width, yerr=rep_ci, capsize=6,
               label='Republicans', color=RED, edgecolor='white', linewidth=1)

ax.set_ylabel('Average Score (1 = Low, 5 = High)')
ax.set_title('Where Do Democrats and Republicans Differ Most?',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.set_ylim(1.0, 4.2)
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.text(2.55, 3.02, 'Scale midpoint', color='gray', fontsize=10)
ax.legend(loc='upper right', framealpha=0.9)

# Effect size labels
d_values = [d_moral, d_other, d_aversion]
d_labels = ['small', 'small', 'large']
for i, (d_val, d_lab) in enumerate(zip(d_values, d_labels)):
    max_y = max(dem_means[i] + dem_ci[i], rep_means[i] + rep_ci[i])
    ax.text(i, max_y + 0.1,
            f'd = {abs(d_val):.2f}  ({d_lab} difference) ***',
            ha='center', fontsize=10, fontweight='bold')

# Bottom note
ax.text(0.02, -0.12,
        "d = Cohen's d = standardized gap between groups\n"
        "Small: d < 0.5 | Medium: d = 0.5-0.8 | Large: d > 0.8\n"
        "*** = statistically significant at p < .001",
        transform=ax.transAxes, fontsize=9, color='#666666', va='top')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig4_components_bar.png')
plt.close()
print(f"  Saved. d_moral={d_moral:.4f}, d_other={d_other:.4f}, d_aversion={d_aversion:.4f}")

# ============================================================================
# FIGURE 5: Free Speech Restriction by Party (with midpoint line)
# ============================================================================
print("Figure 5: Free Speech Restriction")

# Compute faculty and student batteries
faculty_items = [f'Q{q}_scaled' for q in [92, 95, 96, 97, 98, 99, 100, 101, 102]]
student_items = [f'Q{q}_scaled' for q in [103, 104, 105, 106]]
all_fs_items = faculty_items + student_items

existing_faculty = [c for c in faculty_items if c in df.columns]
existing_student = [c for c in student_items if c in df.columns]

df['faculty_restriction'] = 8 - df[existing_faculty].mean(axis=1)
df['student_restriction'] = 8 - df[existing_student].mean(axis=1)

# Re-filter
rep5 = df[df['party_category'] == 'Republican']
dem5 = df[df['party_category'] == 'Democrat']
ind5 = df[df['party_category'] == 'Independent']

fig, ax = plt.subplots(figsize=(10, 6))

batteries = ['Faculty Speech\n(about faculty\nexpression rights)',
             'Student Speech\n(about campus\nspeech norms)']

dem_fac = dem5['faculty_restriction'].dropna()
rep_fac = rep5['faculty_restriction'].dropna()
ind_fac = ind5['faculty_restriction'].dropna()
dem_stu = dem5['student_restriction'].dropna()
rep_stu = rep5['student_restriction'].dropna()
ind_stu = ind5['student_restriction'].dropna()

x = np.arange(2)
width = 0.25

dem_m = [dem_fac.mean(), dem_stu.mean()]
ind_m = [ind_fac.mean(), ind_stu.mean()]
rep_m = [rep_fac.mean(), rep_stu.mean()]
dem_e = [1.96*dem_fac.std(ddof=1)/np.sqrt(len(dem_fac)),
         1.96*dem_stu.std(ddof=1)/np.sqrt(len(dem_stu))]
ind_e = [1.96*ind_fac.std(ddof=1)/np.sqrt(len(ind_fac)),
         1.96*ind_stu.std(ddof=1)/np.sqrt(len(ind_stu))]
rep_e = [1.96*rep_fac.std(ddof=1)/np.sqrt(len(rep_fac)),
         1.96*rep_stu.std(ddof=1)/np.sqrt(len(rep_stu))]

b1 = ax.bar(x - width, dem_m, width, yerr=dem_e, capsize=5,
            label='Democrats', color=BLUE, edgecolor='white')
b2 = ax.bar(x, ind_m, width, yerr=ind_e, capsize=5,
            label='Independents', color=GRAY, edgecolor='white')
b3 = ax.bar(x + width, rep_m, width, yerr=rep_e, capsize=5,
            label='Republicans', color=RED, edgecolor='white')

# Value labels
for bars_group in [b1, b2, b3]:
    for bar in bars_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Average Restriction Score\n(1 = Strongly pro-free-speech,\n7 = Strongly pro-restriction)')
ax.set_title('Who Supports More Speech Restrictions?',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(batteries)
ax.set_ylim(1, 5.5)

# Midpoint line
ax.axhline(y=4, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.text(1.45, 4.05, 'Scale midpoint (4 = Neutral)', color='gray', fontsize=10)

ax.legend(loc='upper right', framealpha=0.9)

# Note about general support for free speech
ax.text(0.02, -0.12,
        "Note: All groups score below the midpoint, meaning students generally\n"
        "support free speech. Republicans score higher = more support for restrictions.",
        transform=ax.transAxes, fontsize=10, color='#444444', va='top', style='italic')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig5_free_speech_bars.png')
plt.close()
print(f"  Saved. Faculty: Dem={dem_fac.mean():.3f}, Rep={rep_fac.mean():.3f}")
print(f"         Student: Dem={dem_stu.mean():.3f}, Rep={rep_stu.mean():.3f}")

# ============================================================================
# FIGURE 6: Scatter - Aversion vs Speech Restriction (CORRECTED key finding)
# ============================================================================
print("Figure 6: Aversion vs Speech Restriction Scatter")

df['combined_restriction'] = 8 - df['free_speech_support_index']
rep6 = df[df['party_category'] == 'Republican'].copy()
dem6 = df[df['party_category'] == 'Democrat'].copy()

fig, ax = plt.subplots(figsize=(9, 7))

# Democrats
dem_x = dem6['aversion_index_D'].dropna()
dem_y = dem6.loc[dem_x.index, 'combined_restriction'].dropna()
common_d = dem_x.index.intersection(dem_y.index)
ax.scatter(dem6.loc[common_d, 'aversion_index_D'],
           dem6.loc[common_d, 'combined_restriction'],
           color=BLUE, alpha=0.25, s=25, label='_nolegend_')

# Republicans
rep_x = rep6['aversion_index_R'].dropna()
rep_y = rep6.loc[rep_x.index, 'combined_restriction'].dropna()
common_r = rep_x.index.intersection(rep_y.index)
ax.scatter(rep6.loc[common_r, 'aversion_index_R'],
           rep6.loc[common_r, 'combined_restriction'],
           color=RED, alpha=0.35, s=25, label='_nolegend_')

# Regression lines
from numpy.polynomial.polynomial import polyfit
# Democrats
d_coeffs = np.polyfit(dem6.loc[common_d, 'aversion_index_D'],
                       dem6.loc[common_d, 'combined_restriction'], 1)
d_line_x = np.linspace(1, 5, 100)
ax.plot(d_line_x, np.polyval(d_coeffs, d_line_x), color=BLUE, linewidth=2.5)

# Republicans
r_coeffs = np.polyfit(rep6.loc[common_r, 'aversion_index_R'],
                       rep6.loc[common_r, 'combined_restriction'], 1)
ax.plot(d_line_x, np.polyval(r_coeffs, d_line_x), color=RED, linewidth=2.5)

# Correlations
r_dem, p_dem = stats.pearsonr(dem6.loc[common_d, 'aversion_index_D'],
                               dem6.loc[common_d, 'combined_restriction'])
r_rep, p_rep = stats.pearsonr(rep6.loc[common_r, 'aversion_index_R'],
                               rep6.loc[common_r, 'combined_restriction'])

# Legend
dem_patch = mpatches.Patch(color=BLUE,
    label=f'Democrats: r = {r_dem:.2f} (p = {p_dem:.3f}), slope = {d_coeffs[0]:.2f}')
rep_patch = mpatches.Patch(color=RED,
    label=f'Republicans: r = {r_rep:.2f} (p < .001), slope = {r_coeffs[0]:.2f}')
ax.legend(handles=[dem_patch, rep_patch], loc='upper left', fontsize=11, framealpha=0.9)

ax.set_xlabel('Social Aversion Score\n(1 = Comfortable with out-partisans, 5 = Strongly avoids them)',
              fontsize=12)
ax.set_ylabel('Speech Restriction Score\n(1 = Strongly pro-free-speech, 7 = Strongly pro-restriction)',
              fontsize=12)
ax.set_title('Do Students Who Avoid the Other Party\nAlso Want to Restrict Speech?',
             fontsize=16, fontweight='bold', pad=15)

# Key finding box
textstr = ("Key finding: For Democrats, the line is flat — avoiding\n"
           "Republicans does NOT predict wanting to restrict speech.\n"
           "For Republicans, the relationship IS significant:\n"
           "more aversion = more support for speech restriction.")
props = dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig6_aversion_speech_scatter.png')
plt.close()
print(f"  Saved. Dem r={r_dem:.4f} p={p_dem:.4f}, Rep r={r_rep:.4f} p={p_rep:.6f}")

# ============================================================================
# FIGURE 7: Partisan Strength Gradient (REORDERED per advisor)
# ============================================================================
print("Figure 7: Partisan Strength Gradient")

# Create partisan strength categories
# party_combined: 1=Strong D, 1.5=Lean D, 2=Somewhat D, 3=Independent,
#                 4=Somewhat R, 4.5=Lean R, 5=Strong R
# Correct order: Strong > Somewhat > Lean > Independent > Lean > Somewhat > Strong

strength_map = {
    1.0: 'Strong\nDemocrat',
    2.0: 'Somewhat\nDemocrat',
    1.5: 'Lean\nDemocrat',
    3.0: 'True\nIndependent',
    4.5: 'Lean\nRepublican',
    4.0: 'Somewhat\nRepublican',
    5.0: 'Strong\nRepublican'
}

# Order for x-axis: Strong D, Somewhat D, Lean D, Independent, Lean R, Somewhat R, Strong R
ordered_values = [1.0, 2.0, 1.5, 3.0, 4.5, 4.0, 5.0]
ordered_labels = [strength_map[v] for v in ordered_values]

fig, ax = plt.subplots(figsize=(12, 7))

# For each strength level, compute means for each component
moral_means, moral_cis = [], []
other_means, other_cis = [], []
aversion_means, aversion_cis = [], []

for val in ordered_values:
    subset = df[df['party_combined'] == val]

    # Use party-appropriate columns
    if val <= 2.5:  # Democrats
        m = subset['moral_index_D'].dropna()
        o = subset['othering_index_D'].dropna()
        a = subset['aversion_index_D'].dropna()
    elif val >= 3.5:  # Republicans
        m = subset['moral_index_R'].dropna()
        o = subset['othering_index_R'].dropna()
        a = subset['aversion_index_R'].dropna()
    else:  # Independents - average both
        m_d = subset['moral_index_D'].dropna()
        m_r = subset['moral_index_R'].dropna()
        m = pd.concat([m_d, m_r]).groupby(level=0).mean()
        o_d = subset['othering_index_D'].dropna()
        o_r = subset['othering_index_R'].dropna()
        o = pd.concat([o_d, o_r]).groupby(level=0).mean()
        a_d = subset['aversion_index_D'].dropna()
        a_r = subset['aversion_index_R'].dropna()
        a = pd.concat([a_d, a_r]).groupby(level=0).mean()

    moral_means.append(m.mean() if len(m) > 0 else np.nan)
    moral_cis.append(1.96 * m.std(ddof=1)/np.sqrt(len(m)) if len(m) > 1 else 0)
    other_means.append(o.mean() if len(o) > 0 else np.nan)
    other_cis.append(1.96 * o.std(ddof=1)/np.sqrt(len(o)) if len(o) > 1 else 0)
    aversion_means.append(a.mean() if len(a) > 0 else np.nan)
    aversion_cis.append(1.96 * a.std(ddof=1)/np.sqrt(len(a)) if len(a) > 1 else 0)

x = np.arange(len(ordered_values))

ax.errorbar(x, moral_means, yerr=moral_cis, marker='o', markersize=8,
            linewidth=2.5, color='#2CA02C', label='Moral Identity\n(party = moral cause)',
            capsize=4, capthick=1.5)
ax.errorbar(x, other_means, yerr=other_cis, marker='s', markersize=8,
            linewidth=2.5, color='#7F7FFF', label='Othering\n(out-party = alien)',
            capsize=4, capthick=1.5, linestyle='--')
ax.errorbar(x, aversion_means, yerr=aversion_cis, marker='^', markersize=8,
            linewidth=2.5, color='#FF7F0E', label='Social Aversion\n(avoids out-party)',
            capsize=4, capthick=1.5, linestyle='-.')

# Background shading
ax.axvspan(-0.5, 2.5, alpha=0.06, color='blue')
ax.axvspan(3.5, 6.5, alpha=0.06, color='red')
ax.text(1, 4.6, 'Democrats', ha='center', fontsize=13, color=BLUE, fontweight='bold')
ax.text(5, 4.6, 'Republicans', ha='center', fontsize=13, color=RED, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(ordered_labels, fontsize=11)
ax.set_ylabel('Average Score (1 = Low, 5 = High)', fontsize=13)
ax.set_title('Do Stronger Partisans Show More Hostility\nToward the Other Party?',
             fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(1.0, 5.0)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

ax.text(0.02, -0.1,
        "Each point = average score for students at that level of party identification.\n"
        "Higher = more hostility. Error bars = 95% confidence intervals.",
        transform=ax.transAxes, fontsize=10, color='#666666', va='top')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig7_partisan_gradient.png')
plt.close()
print("  Saved.")

# ============================================================================
# FIGURE 8: OLS Regression Coefficient Plot (simplified)
# ============================================================================
print("Figure 8: Regression Coefficient Plot")

try:
    import statsmodels.api as sm

    # Prepare regression data
    reg_df = df[df['party_category'].isin(['Democrat', 'Republican'])].copy()
    reg_df['moral'] = np.where(reg_df['party_category'] == 'Democrat',
                                reg_df['moral_index_D'], reg_df['moral_index_R'])
    reg_df['othering'] = np.where(reg_df['party_category'] == 'Democrat',
                                   reg_df['othering_index_D'], reg_df['othering_index_R'])
    reg_df['aversion'] = np.where(reg_df['party_category'] == 'Democrat',
                                   reg_df['aversion_index_D'], reg_df['aversion_index_R'])
    reg_df['party_binary'] = (reg_df['party_category'] == 'Republican').astype(int)
    reg_df['restriction'] = 8 - reg_df['free_speech_support_index']

    reg_cols = ['moral', 'othering', 'aversion', 'party_binary', 'restriction']
    reg_clean = reg_df[reg_cols].dropna()

    # Within-party models
    dem_reg = reg_clean[reg_clean['party_binary'] == 0]
    rep_reg = reg_clean[reg_clean['party_binary'] == 1]

    # M3-Dem
    X_d = sm.add_constant(dem_reg[['moral', 'othering', 'aversion']])
    m3d = sm.OLS(dem_reg['restriction'], X_d).fit()

    # M3-Rep
    X_r = sm.add_constant(rep_reg[['moral', 'othering', 'aversion']])
    m3r = sm.OLS(rep_reg['restriction'], X_r).fit()

    # Pooled (controlling for party)
    X_all = sm.add_constant(reg_clean[['moral', 'othering', 'aversion', 'party_binary']])
    m1 = sm.OLS(reg_clean['restriction'], X_all).fit()

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)

    components = ['moral', 'othering', 'aversion']
    comp_labels = ['Moral Identity\n(sees party as\nmoral cause)',
                   'Othering\n(sees out-party\nas alien)',
                   'Social Aversion\n(avoids out-party\nmembers)']

    for i, (comp, label) in enumerate(zip(components, comp_labels)):
        ax = axes[i]

        # Republicans only
        coef_r = m3r.params[comp]
        ci_r = m3r.conf_int().loc[comp]
        sig_r = m3r.pvalues[comp] < 0.05

        # Democrats only
        coef_d = m3d.params[comp]
        ci_d = m3d.conf_int().loc[comp]
        sig_d = m3d.pvalues[comp] < 0.05

        # Pooled
        coef_all = m1.params[comp]
        ci_all = m1.conf_int().loc[comp]
        sig_all = m1.pvalues[comp] < 0.05

        y_pos = [2, 1, 0]
        y_labels = ['Republicans\nonly', 'Democrats\nonly', 'All students\n(controlling\nfor party)']

        for j, (coef, ci, sig) in enumerate([(coef_r, ci_r, sig_r),
                                              (coef_d, ci_d, sig_d),
                                              (coef_all, ci_all, sig_all)]):
            color = RED if j == 0 else (BLUE if j == 1 else GRAY)
            marker = 'D' if sig else 'o'
            ax.errorbar(coef, y_pos[j], xerr=[[coef - ci[0]], [ci[1] - coef]],
                       fmt=marker, color=color, markersize=8, capsize=5, capthick=1.5,
                       linewidth=2)
            # Add coefficient text
            p_val = m3r.pvalues[comp] if j == 0 else (m3d.pvalues[comp] if j == 1 else m1.pvalues[comp])
            sig_text = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
            ax.text(coef + 0.02, y_pos[j] + 0.15, f'{coef:+.2f}{sig_text}',
                    fontsize=9, ha='left', color=color)

        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        ax.set_xlabel('Effect on speech restriction')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.4, 0.8)

    fig.suptitle('Does Partisan Hostility Predict Support for Speech Restrictions?',
                 fontsize=15, fontweight='bold', y=1.02)

    # Add legend
    axes[2].plot([], [], 'D', color='gray', markersize=6, label='Statistically significant (p < .05)')
    axes[2].plot([], [], 'o', color='gray', markersize=6, label='Not significant (p >= .05)')
    axes[2].legend(loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig8_regression_coefs.png')
    plt.close()
    print("  Saved.")

    # Print regression table for thesis
    print("\n" + "="*60)
    print("REGRESSION TABLE FOR THESIS")
    print("="*60)
    print("\nM1 (Pooled):")
    for var in ['moral', 'othering', 'aversion', 'party_binary']:
        print(f"  {var}: b={m1.params[var]:.3f}, SE={m1.bse[var]:.3f}, p={m1.pvalues[var]:.4f}")
    print(f"  R-sq={m1.rsquared:.4f}")

    print("\nM3-Dem:")
    for var in ['moral', 'othering', 'aversion']:
        print(f"  {var}: b={m3d.params[var]:.3f}, SE={m3d.bse[var]:.3f}, p={m3d.pvalues[var]:.4f}")
    print(f"  R-sq={m3d.rsquared:.4f}")

    print("\nM3-Rep:")
    for var in ['moral', 'othering', 'aversion']:
        print(f"  {var}: b={m3r.params[var]:.3f}, SE={m3r.bse[var]:.3f}, p={m3r.pvalues[var]:.4f}")
    print(f"  R-sq={m3r.rsquared:.4f}")

except ImportError:
    print("  statsmodels not available, skipping regression figure")

# ============================================================================
# NEW: Component Bar Chart for Table 3 (per Comment 15)
# ============================================================================
print("\nBonus: Component Bar Chart (for Table 3, 6 values)")

fig, ax = plt.subplots(figsize=(10, 6))

components = ['Moral\nIdentity', 'Othering', 'Social\nAversion']
dem_means = [dem_moral.mean(), dem_other.mean(), dem_aversion.mean()]
rep_means = [rep_moral.mean(), rep_other.mean(), rep_aversion.mean()]
dem_sds = [dem_moral.std(ddof=1), dem_other.std(ddof=1), dem_aversion.std(ddof=1)]
rep_sds = [rep_moral.std(ddof=1), rep_other.std(ddof=1), rep_aversion.std(ddof=1)]

x = np.arange(len(components))
width = 0.35

b1 = ax.bar(x - width/2, dem_means, width, color=BLUE, edgecolor='white',
            label='Democrats')
b2 = ax.bar(x + width/2, rep_means, width, color=RED, edgecolor='white',
            label='Republicans')

# Value + SD labels
for bars, means, sds in [(b1, dem_means, dem_sds), (b2, rep_means, rep_sds)]:
    for bar, m, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{m:.2f}\n(SD={sd:.2f})', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Average Score (1 = Low, 5 = High)')
ax.set_title('Affective Polarization by Component and Party',
             fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.set_ylim(1.0, 4.5)
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig_Table3_components.png')
plt.close()
print("  Saved.")

print("\n" + "="*60)
print("ALL FIGURES GENERATED IN:", OUTPUT_DIR)
print("="*60)
