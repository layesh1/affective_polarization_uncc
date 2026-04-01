"""
Verification Script: Re-run all regressions and correlations
to verify every number in the thesis document.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('Cleaned_Data/polarization_cleaned_CORRECTED.csv')
print(f"Loaded {len(df)} rows")

rep = df[df['party_category'] == 'Republican'].copy()
dem = df[df['party_category'] == 'Democrat'].copy()
print(f"Republicans: n={len(rep)}, Democrats: n={len(dem)}")

# ============================================================================
# 1. VERIFY ALL DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("1. DESCRIPTIVE STATISTICS VERIFICATION")
print("="*80)

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1-1)*g1.std(ddof=1)**2 + (n2-1)*g2.std(ddof=1)**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled_std

# Overall Affective Polarization
dem_ap = dem['affective_polarization_D'].dropna()
rep_ap = rep['affective_polarization_R'].dropna()
t, p = stats.ttest_ind(dem_ap, rep_ap)
d = cohens_d(dem_ap, rep_ap)
print(f"\nOverall Affective Polarization:")
print(f"  Dem: M={dem_ap.mean():.3f}, SD={dem_ap.std(ddof=1):.3f}, n={len(dem_ap)}")
print(f"  Rep: M={rep_ap.mean():.3f}, SD={rep_ap.std(ddof=1):.3f}, n={len(rep_ap)}")
print(f"  t={t:.2f}, p={p:.6f}, d={d:.4f}")

# Moral Identity
dem_moral = dem['moral_index_D'].dropna()
rep_moral = rep['moral_index_R'].dropna()
t, p = stats.ttest_ind(dem_moral, rep_moral)
d = cohens_d(dem_moral, rep_moral)
print(f"\nMoral Identity:")
print(f"  Dem: M={dem_moral.mean():.3f}, SD={dem_moral.std(ddof=1):.3f}, n={len(dem_moral)}")
print(f"  Rep: M={rep_moral.mean():.3f}, SD={rep_moral.std(ddof=1):.3f}, n={len(rep_moral)}")
print(f"  t={t:.2f}, p={p:.6f}, d={d:.4f}")

# Othering
dem_other = dem['othering_index_D'].dropna()
rep_other = rep['othering_index_R'].dropna()
t, p = stats.ttest_ind(dem_other, rep_other)
d = cohens_d(dem_other, rep_other)
print(f"\nOthering:")
print(f"  Dem: M={dem_other.mean():.3f}, SD={dem_other.std(ddof=1):.3f}, n={len(dem_other)}")
print(f"  Rep: M={rep_other.mean():.3f}, SD={rep_other.std(ddof=1):.3f}, n={len(rep_other)}")
print(f"  t={t:.2f}, p={p:.6f}, d={d:.4f}")

# Social Aversion
dem_aversion = dem['aversion_index_D'].dropna()
rep_aversion = rep['aversion_index_R'].dropna()
t, p = stats.ttest_ind(dem_aversion, rep_aversion)
d = cohens_d(dem_aversion, rep_aversion)
print(f"\nSocial Aversion:")
print(f"  Dem: M={dem_aversion.mean():.3f}, SD={dem_aversion.std(ddof=1):.3f}, n={len(dem_aversion)}")
print(f"  Rep: M={rep_aversion.mean():.3f}, SD={rep_aversion.std(ddof=1):.3f}, n={len(rep_aversion)}")
print(f"  t={t:.2f}, p={p:.6f}, d={d:.4f}")

# Free Speech
dem_fs = dem['free_speech_support_index'].dropna()
rep_fs = rep['free_speech_support_index'].dropna()
t, p = stats.ttest_ind(dem_fs, rep_fs)
d = cohens_d(dem_fs, rep_fs)
print(f"\nFree Speech Support Index (higher = more support for free speech):")
print(f"  Dem: M={dem_fs.mean():.3f}, SD={dem_fs.std(ddof=1):.3f}, n={len(dem_fs)}")
print(f"  Rep: M={rep_fs.mean():.3f}, SD={rep_fs.std(ddof=1):.3f}, n={len(rep_fs)}")
print(f"  t={t:.2f}, p={p:.6f}, d={d:.4f}")

# ============================================================================
# 2. FREE SPEECH: FACULTY vs STUDENT vs COMBINED
# ============================================================================
print("\n" + "="*80)
print("2. FREE SPEECH BATTERIES (Faculty vs Student vs Combined)")
print("="*80)

# Need to compute faculty-only and student-only indices
# Faculty: Q92, Q95-Q102 (9 items)
# Student: Q103-Q106 (4 items)
faculty_items = [f'Q{q}_scaled' for q in [92, 95, 96, 97, 98, 99, 100, 101, 102]]
student_items = [f'Q{q}_scaled' for q in [103, 104, 105, 106]]
all_fs_items = faculty_items + student_items

# Check which columns exist
existing_faculty = [c for c in faculty_items if c in df.columns]
existing_student = [c for c in student_items if c in df.columns]
existing_all = [c for c in all_fs_items if c in df.columns]

print(f"Faculty items found: {len(existing_faculty)}/{len(faculty_items)}")
print(f"Student items found: {len(existing_student)}/{len(student_items)}")

df['faculty_speech_index'] = df[existing_faculty].mean(axis=1)
df['student_speech_index'] = df[existing_student].mean(axis=1)
df['combined_speech_index'] = df[existing_all].mean(axis=1)

# Re-filter after adding columns
rep2 = df[df['party_category'] == 'Republican']
dem2 = df[df['party_category'] == 'Democrat']

for name, col in [('Faculty Speech', 'faculty_speech_index'),
                   ('Student Speech', 'student_speech_index'),
                   ('Combined (all 13)', 'combined_speech_index'),
                   ('Original FS Support', 'free_speech_support_index')]:
    d_vals = dem2[col].dropna()
    r_vals = rep2[col].dropna()
    t, p = stats.ttest_ind(d_vals, r_vals)
    d = cohens_d(d_vals, r_vals)
    print(f"\n{name}:")
    print(f"  Dem: M={d_vals.mean():.3f}, SD={d_vals.std(ddof=1):.3f}")
    print(f"  Rep: M={r_vals.mean():.3f}, SD={r_vals.std(ddof=1):.3f}")
    print(f"  t={t:.2f}, p={p:.6f}, d={d:.4f}")

# NOTE: The thesis uses a RESTRICTION index (lower = more free speech support)
# Need to check if thesis reports restriction (7 - support) or support values
print("\nNOTE: If thesis reports RESTRICTION scores (higher = more restrictive):")
print("  The free_speech_support_index is coded so HIGH = more support for free speech")
print("  The thesis Table 4 reports lower scores for Democrats = more pro-free-speech")
print("  This means the thesis is reporting RESTRICTION scores")
print("  Restriction = 8 - support (for 7-point scale)")

# Compute restriction index
df['faculty_restriction'] = 8 - df['faculty_speech_index']
df['student_restriction'] = 8 - df['student_speech_index']
df['combined_restriction'] = 8 - df['combined_speech_index']

rep3 = df[df['party_category'] == 'Republican']
dem3 = df[df['party_category'] == 'Democrat']

print("\n--- RESTRICTION SCORES (higher = more restrictive) ---")
for name, col in [('Faculty Restriction', 'faculty_restriction'),
                   ('Student Restriction', 'student_restriction'),
                   ('Combined Restriction', 'combined_restriction')]:
    d_vals = dem3[col].dropna()
    r_vals = rep3[col].dropna()
    t, p = stats.ttest_ind(r_vals, d_vals)  # Rep - Dem direction
    d = cohens_d(r_vals, d_vals)
    print(f"\n{name}:")
    print(f"  Dem: M={d_vals.mean():.3f}, SD={d_vals.std(ddof=1):.3f}")
    print(f"  Rep: M={r_vals.mean():.3f}, SD={r_vals.std(ddof=1):.3f}")
    print(f"  t={t:.2f}, p={p:.6f}, d={d:.4f}")

# ============================================================================
# 3. BIVARIATE CORRELATIONS: Aversion vs Speech
# ============================================================================
print("\n" + "="*80)
print("3. BIVARIATE CORRELATIONS: Aversion vs Free Speech Restriction")
print("="*80)

# For Democrats
dem_av = dem3['aversion_index_D'].dropna()
dem_rest = dem3.loc[dem_av.index, 'combined_restriction'].dropna()
common_idx = dem_av.index.intersection(dem_rest.index)
r_dem, p_dem = stats.pearsonr(dem3.loc[common_idx, 'aversion_index_D'],
                               dem3.loc[common_idx, 'combined_restriction'])
print(f"\nDemocrats: aversion vs speech restriction")
print(f"  r = {r_dem:.4f}, p = {p_dem:.6f}, n = {len(common_idx)}")

# For Republicans
rep_av = rep3['aversion_index_R'].dropna()
rep_rest = rep3.loc[rep_av.index, 'combined_restriction'].dropna()
common_idx_r = rep_av.index.intersection(rep_rest.index)
r_rep, p_rep = stats.pearsonr(rep3.loc[common_idx_r, 'aversion_index_R'],
                               rep3.loc[common_idx_r, 'combined_restriction'])
print(f"\nRepublicans: aversion vs speech restriction")
print(f"  r = {r_rep:.4f}, p = {p_rep:.6f}, n = {len(common_idx_r)}")

# All three components vs speech restriction
print("\n--- All components vs speech restriction ---")
for party_name, party_df, moral_col, other_col, aversion_col in [
    ('Democrats', dem3, 'moral_index_D', 'othering_index_D', 'aversion_index_D'),
    ('Republicans', rep3, 'moral_index_R', 'othering_index_R', 'aversion_index_R')]:
    print(f"\n{party_name}:")
    for comp_name, comp_col in [('Moral Identity', moral_col), ('Othering', other_col), ('Aversion', aversion_col)]:
        valid = party_df[[comp_col, 'combined_restriction']].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid[comp_col], valid['combined_restriction'])
            print(f"  {comp_name}: r = {r:.4f}, p = {p:.6f}")

# ============================================================================
# 4. OLS REGRESSION
# ============================================================================
print("\n" + "="*80)
print("4. OLS REGRESSION MODELS")
print("="*80)

try:
    import statsmodels.api as sm

    # Prepare regression data - need unified polarization columns
    # For each person, use their party-appropriate polarization scores
    reg_df = df[df['party_category'].isin(['Democrat', 'Republican'])].copy()

    # Create unified columns
    reg_df['moral'] = np.where(reg_df['party_category'] == 'Democrat',
                                reg_df['moral_index_D'], reg_df['moral_index_R'])
    reg_df['othering'] = np.where(reg_df['party_category'] == 'Democrat',
                                   reg_df['othering_index_D'], reg_df['othering_index_R'])
    reg_df['aversion'] = np.where(reg_df['party_category'] == 'Democrat',
                                   reg_df['aversion_index_D'], reg_df['aversion_index_R'])
    reg_df['party_binary'] = (reg_df['party_category'] == 'Republican').astype(int)
    reg_df['restriction'] = reg_df['combined_restriction']

    # Drop missing
    reg_cols = ['moral', 'othering', 'aversion', 'party_binary', 'restriction']
    reg_clean = reg_df[reg_cols].dropna()
    print(f"\nRegression sample: n={len(reg_clean)}")
    print(f"  Democrats: {(reg_clean['party_binary']==0).sum()}")
    print(f"  Republicans: {(reg_clean['party_binary']==1).sum()}")

    # M1: Pooled model
    print("\n--- M1: Pooled Model ---")
    X1 = sm.add_constant(reg_clean[['moral', 'othering', 'aversion', 'party_binary']])
    y = reg_clean['restriction']
    m1 = sm.OLS(y, X1).fit()
    print(m1.summary2().tables[1].to_string())
    print(f"R-squared: {m1.rsquared:.4f}")

    # M2: Interaction model
    print("\n--- M2: Interaction Model ---")
    reg_clean['moral_x_party'] = reg_clean['moral'] * reg_clean['party_binary']
    reg_clean['othering_x_party'] = reg_clean['othering'] * reg_clean['party_binary']
    reg_clean['aversion_x_party'] = reg_clean['aversion'] * reg_clean['party_binary']

    X2 = sm.add_constant(reg_clean[['moral', 'othering', 'aversion', 'party_binary',
                                     'moral_x_party', 'othering_x_party', 'aversion_x_party']])
    m2 = sm.OLS(y, X2).fit()
    print(m2.summary2().tables[1].to_string())
    print(f"R-squared: {m2.rsquared:.4f}")

    # M3-Dem: Democrats only
    print("\n--- M3-Dem: Democrats Only ---")
    dem_reg = reg_clean[reg_clean['party_binary'] == 0]
    X3d = sm.add_constant(dem_reg[['moral', 'othering', 'aversion']])
    m3d = sm.OLS(dem_reg['restriction'], X3d).fit()
    print(m3d.summary2().tables[1].to_string())
    print(f"R-squared: {m3d.rsquared:.4f}")

    # M3-Rep: Republicans only
    print("\n--- M3-Rep: Republicans Only ---")
    rep_reg = reg_clean[reg_clean['party_binary'] == 1]
    X3r = sm.add_constant(rep_reg[['moral', 'othering', 'aversion']])
    m3r = sm.OLS(rep_reg['restriction'], X3r).fit()
    print(m3r.summary2().tables[1].to_string())
    print(f"R-squared: {m3r.rsquared:.4f}")

except ImportError:
    print("statsmodels not installed - cannot run OLS regression")
    print("Install with: pip3 install statsmodels")

# ============================================================================
# 5. CRONBACH'S ALPHA
# ============================================================================
print("\n" + "="*80)
print("5. CRONBACH'S ALPHA")
print("="*80)

def cronbach_alpha(items_df):
    items_df = items_df.dropna()
    n = items_df.shape[1]
    if n < 2 or len(items_df) < 3:
        return np.nan
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    return (n / (n - 1)) * (1 - item_vars.sum() / total_var)

# Aversion
print(f"\nAversion - Democrats (Q138, Q139, Q140_scaled):")
alpha_dem_av = cronbach_alpha(dem[['Q138_scaled', 'Q139_scaled', 'Q140_scaled']])
print(f"  alpha = {alpha_dem_av:.3f}")

print(f"\nAversion - Republicans (Q135, Q136, Q137_scaled):")
alpha_rep_av = cronbach_alpha(rep[['Q135_scaled', 'Q136_scaled', 'Q137_scaled']])
print(f"  alpha = {alpha_rep_av:.3f}")

# Moral Identity
print(f"\nMoral Identity - Democrats:")
alpha_dem_moral = cronbach_alpha(dem[['moral1D', 'moral2D', 'moral3D']])
print(f"  alpha = {alpha_dem_moral:.3f}")

print(f"\nMoral Identity - Republicans:")
alpha_rep_moral = cronbach_alpha(rep[['moral1R', 'moral2R', 'moral3R']])
print(f"  alpha = {alpha_rep_moral:.3f}")

# Othering
print(f"\nOthering - Democrats:")
alpha_dem_other = cronbach_alpha(dem[['other1D', 'other2D', 'other3D']])
print(f"  alpha = {alpha_dem_other:.3f}")

print(f"\nOthering - Republicans:")
alpha_rep_other = cronbach_alpha(rep[['other1R', 'other2R', 'other3R']])
print(f"  alpha = {alpha_rep_other:.3f}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
