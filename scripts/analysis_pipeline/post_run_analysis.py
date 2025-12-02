#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of YouTube Shorts Performance Data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/tagging/entrepreneurship/_infra/queries/master_query_results.csv')

# Filter out clips with no qualitative hook type (indicative of missing qual analysis)
hook_mask = df['hook_type'].notna() & df['hook_type'].astype(str).str.strip().ne("")
if hook_mask.sum() != len(df):
    removed = len(df) - hook_mask.sum()
    print(f"Filtering out {removed} clip(s) with blank hook_type before analysis.")
df = df[hook_mask]

# Normalize metrics per channel (VPI/LPI/CPI)
channel_means = df.groupby('channel_id').agg({
    'view_count': 'mean',
    'like_count': 'mean',
    'comment_count': 'mean'
}).rename(columns={
    'view_count': 'mean_views',
    'like_count': 'mean_likes',
    'comment_count': 'mean_comments'
})

df = df.merge(channel_means, left_on='channel_id', right_index=True, how='left')

df['VPI'] = (df['view_count'] / df['mean_views']).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
df['LPI'] = (df['like_count'] / df['mean_likes']).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
df['CPI'] = (df['comment_count'] / df['mean_comments']).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)

print("="*80)
print("CHANNEL NORMALIZATION")
print("="*80)
print(f"\nDataset: {len(df)} clips from {df['channel_id'].nunique()} channels")
print("\nChannel means (views/likes/comments):")
for channel, row in channel_means.iterrows():
    print(f"  {channel}: views={row['mean_views']:.1f}, likes={row['mean_likes']:.1f}, comments={row['mean_comments']:.2f}")
print("\n" + "-"*80)

print("="*80)
print("YOUTUBE SHORTS PERFORMANCE ANALYSIS")
print("="*80)
print(f"\nDataset: {len(df)} videos from {df['channel_id'].nunique()} channel(s)")
print(f"Date range: {df['days_since_publish'].min()} to {df['days_since_publish'].max()} days old")

# ============================================================================
# PART 1: EXECUTIVE SUMMARY (will be filled in at end)
# ============================================================================

# ============================================================================
# PART 2: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: CORRELATION ANALYSIS")
print("="*80)

# Define feature groups
quant_features = [
    'duration_s', 'word_count', 'wpm', 'hook_word_count', 'hook_wpm',
    'num_sentences', 'question_start', 'reading_level', 'filler_count',
    'filler_density', 'first_person_ratio', 'second_person_ratio'
]

binary_features = ['has_examples', 'has_payoff', 'has_numbers', 'insider_language']

categorical_features = ['hook_type', 'hook_emotion', 'topic_primary', 'technical_depth']

performance_metrics = ['VPI', 'like_count', 'comment_count', 'engagement_rate']

def interpret_correlation(r):
    """Interpret correlation strength"""
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std

print("\n--- Quantitative Features vs Performance Metrics ---\n")

correlation_results = []

for feature in quant_features:
    for metric in performance_metrics:
        r, p = pearsonr(df[feature], df[metric])
        correlation_results.append({
            'Feature': feature,
            'Metric': metric,
            'Pearson_r': round(r, 4),
            'p_value': round(p, 4),
            'Significant': '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')),
            'Interpretation': interpret_correlation(r)
        })

corr_df = pd.DataFrame(correlation_results)

# Pivot for VPI
print("CORRELATIONS WITH VPI (views normalized by channel):")
print("-" * 60)
view_corr = corr_df[corr_df['Metric'] == 'VPI'].sort_values('Pearson_r', key=abs, ascending=False)
for _, row in view_corr.iterrows():
    print(f"  {row['Feature']:25s} r={row['Pearson_r']:+.4f}  p={row['p_value']:.4f} {row['Significant']:3s} ({row['Interpretation']})")

print("\nCORRELATIONS WITH ENGAGEMENT_RATE:")
print("-" * 60)
eng_corr = corr_df[corr_df['Metric'] == 'engagement_rate'].sort_values('Pearson_r', key=abs, ascending=False)
for _, row in eng_corr.iterrows():
    print(f"  {row['Feature']:25s} r={row['Pearson_r']:+.4f}  p={row['p_value']:.4f} {row['Significant']:3s} ({row['Interpretation']})")

# Top 5 predictors
print("\n" + "-"*60)
print("TOP 5 PREDICTORS FOR VPI (by |r|):")
for i, (_, row) in enumerate(view_corr.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']}: r={row['Pearson_r']:+.4f}")

print("\nTOP 5 PREDICTORS FOR ENGAGEMENT_RATE (by |r|):")
for i, (_, row) in enumerate(eng_corr.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']}: r={row['Pearson_r']:+.4f}")

# ============================================================================
# PART 3: BINARY FEATURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: BINARY FEATURE ANALYSIS (T-Tests)")
print("="*80)

for feature in binary_features:
    print(f"\n--- {feature.upper()} ---")
    
    group_0 = df[df[feature] == 0]
    group_1 = df[df[feature] == 1]
    
    print(f"  Sample sizes: No={len(group_0)}, Yes={len(group_1)}")
    
    for metric in ['VPI', 'engagement_rate']:
        mean_0 = group_0[metric].mean()
        mean_1 = group_1[metric].mean()
        
        # T-test
        t_stat, p_val = ttest_ind(group_1[metric], group_0[metric])
        
        # Cohen's d
        d = cohens_d(group_1[metric], group_0[metric])
        
        # Percentage lift
        if mean_0 > 0:
            lift = ((mean_1 - mean_0) / mean_0) * 100
        else:
            lift = 0
        
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        
        print(f"\n  {metric}:")
        print(f"    No:  mean={mean_0:,.3f}" if metric == 'VPI' else f"    No:  mean={mean_0:.4f}")
        print(f"    Yes: mean={mean_1:,.3f}" if metric == 'VPI' else f"    Yes: mean={mean_1:.4f}")
        print(f"    Lift: {lift:+.1f}%")
        print(f"    t={t_stat:.3f}, p={p_val:.4f} ({sig})")
        print(f"    Cohen's d={d:.3f} ({'small' if abs(d) < 0.5 else ('medium' if abs(d) < 0.8 else 'large')} effect)")

# ============================================================================
# PART 4: CATEGORICAL FEATURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: CATEGORICAL FEATURE ANALYSIS (ANOVA)")
print("="*80)

for feature in categorical_features:
    print(f"\n--- {feature.upper()} ---")
    
    categories = df[feature].unique()
    print(f"  Categories: {list(categories)}")
    
    # Group statistics
    group_stats = df.groupby(feature).agg({
        'VPI': ['count', 'mean', 'median', 'std'],
        'engagement_rate': ['mean', 'median', 'std']
    }).round(2)
    
    print(f"\n  VPI by {feature}:")
    for cat in categories:
        cat_data = df[df[feature] == cat]
        n = len(cat_data)
        mean_v = cat_data['VPI'].mean()
        med_v = cat_data['VPI'].median()
        std_v = cat_data['VPI'].std()
        print(f"    {cat:20s}: n={n:2d}, mean={mean_v:>10,.3f}, median={med_v:>10,.3f}, std={std_v:>10,.3f}")
    
    # ANOVA for VPI
    groups = [df[df[feature] == cat]['VPI'].values for cat in categories]
    if len(groups) > 1 and all(len(g) > 0 for g in groups):
        f_stat, p_val = f_oneway(*groups)
        # Eta-squared
        grand_mean = df['VPI'].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((df['VPI'] - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        print(f"\n  ANOVA (VPI): F={f_stat:.3f}, p={p_val:.4f} ({sig})")
        print(f"  Eta-squared: {eta_sq:.4f} ({'small' if eta_sq < 0.06 else ('medium' if eta_sq < 0.14 else 'large')} effect)")
    
    print(f"\n  Engagement Rate by {feature}:")
    for cat in categories:
        cat_data = df[df[feature] == cat]
        n = len(cat_data)
        mean_e = cat_data['engagement_rate'].mean()
        med_e = cat_data['engagement_rate'].median()
        print(f"    {cat:20s}: n={n:2d}, mean={mean_e:.4f}, median={med_e:.4f}")

# ============================================================================
# PART 5: TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: TEMPORAL PATTERN ANALYSIS")
print("="*80)

print("\n--- Day of Week Analysis ---")
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_stats = df.groupby('day_published_number').agg({
    'VPI': ['count', 'mean', 'median', 'std'],
    'engagement_rate': ['mean']
}).round(2)

print("\n  VPI by Day of Week:")
for day_num in sorted(df['day_published_number'].dropna().unique()):
    day_int = int(day_num)
    if day_int < 0 or day_int >= 7:
        day_name = f"Day {day_num}"
    else:
        day_name = day_names[day_int]
    day_data = df[df['day_published_number'] == day_num]
    n = len(day_data)
    mean_v = day_data['VPI'].mean()
    print(f"    {day_name:12s}: n={n:2d}, mean={mean_v:>10,.3f}")

# ANOVA for day of week
day_groups = [df[df['day_published_number'] == d]['VPI'].values for d in df['day_published_number'].unique()]
if len(day_groups) > 1:
    f_stat, p_val = f_oneway(*day_groups)
    print(f"\n  ANOVA (day effect): F={f_stat:.3f}, p={p_val:.4f}")

print("\n--- Hour of Day Analysis ---")
hour_stats = df.groupby('hour_published_number')['VPI'].agg(['count', 'mean']).round(3)
print("\n  VPI by Hour:")
for hour in sorted(df['hour_published_number'].dropna().unique()):
    hour_int = int(hour)
    hour_label = f"{hour_int:2d}"
    hour_data = df[df['hour_published_number'] == hour]
    n = len(hour_data)
    mean_v = hour_data['VPI'].mean()
    print(f"    Hour {hour_label}: n={n:2d}, mean={mean_v:>10,.3f}")

print("\n--- Recency Bias ---")
r, p = pearsonr(df['days_since_publish'], df['VPI'])
print(f"  Correlation (days_since_publish vs VPI): r={r:.4f}, p={p:.4f}")
print(f"  Interpretation: {interpret_correlation(r)}")

# ============================================================================
# PART 6: INTERACTION EFFECTS
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: INTERACTION EFFECTS")
print("="*80)

print("\n--- Hook Type × Hook Emotion ---")
interaction = df.groupby(['hook_type', 'hook_emotion'])['VPI'].agg(['count', 'mean']).round(3)
interaction = interaction[interaction['count'] >= 2]  # Filter small groups
interaction = interaction.sort_values('mean', ascending=False)
print("\n  Top Combinations (n >= 2):")
for idx, row in interaction.head(10).iterrows():
    print(f"    {idx[0]:20s} + {idx[1]:15s}: n={int(row['count']):2d}, mean={row['mean']:>10,.3f}")

print("\n--- Has Numbers × Hook Type ---")
for hook in df['hook_type'].unique():
    hook_data = df[df['hook_type'] == hook]
    no_num = hook_data[hook_data['has_numbers'] == 0]['VPI'].mean()
    yes_num = hook_data[hook_data['has_numbers'] == 1]['VPI'].mean()
    n_no = len(hook_data[hook_data['has_numbers'] == 0])
    n_yes = len(hook_data[hook_data['has_numbers'] == 1])
    if n_no > 0 and n_yes > 0:
        lift = ((yes_num - no_num) / no_num * 100) if no_num > 0 else 0
        print(f"  {hook:20s}: No nums (n={n_no})={no_num:>10,.3f}, With nums (n={n_yes})={yes_num:>10,.3f}, lift={lift:+.1f}%")

# ============================================================================
# PART 7: OUTLIER ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: OUTLIER ANALYSIS")
print("="*80)

print("\n--- Top 5 Performers (by VPI) ---")
top5 = df.nlargest(5, 'VPI')[['video_id', 'VPI', 'engagement_rate', 'hook_type', 'hook_emotion', 'has_payoff', 'has_numbers', 'duration_s', 'view_count']]
for i, (_, row) in enumerate(top5.iterrows(), 1):
    print(f"  {i}. VPI={row['VPI']:>6.3f}, raw views={row['view_count']:>10,}, eng={row['engagement_rate']:.4f}, hook={row['hook_type']}, emotion={row['hook_emotion']}, payoff={row['has_payoff']}, nums={row['has_numbers']}, dur={row['duration_s']:.0f}s")

print("\n  Common traits in top 5:")
print(f"    - has_payoff: {top5['has_payoff'].mean()*100:.0f}% have it")
print(f"    - has_numbers: {top5['has_numbers'].mean()*100:.0f}% have it")
print(f"    - hook_type distribution: {dict(top5['hook_type'].value_counts())}")
print(f"    - hook_emotion distribution: {dict(top5['hook_emotion'].value_counts())}")
print(f"    - avg duration: {top5['duration_s'].mean():.0f}s")

print("\n--- Bottom 5 Performers (by VPI) ---")
bot5 = df.nsmallest(5, 'VPI')[['video_id', 'VPI', 'engagement_rate', 'hook_type', 'hook_emotion', 'has_payoff', 'has_numbers', 'duration_s', 'view_count']]
for i, (_, row) in enumerate(bot5.iterrows(), 1):
    print(f"  {i}. VPI={row['VPI']:>6.3f}, raw views={row['view_count']:>10,}, eng={row['engagement_rate']:.4f}, hook={row['hook_type']}, emotion={row['hook_emotion']}, payoff={row['has_payoff']}, nums={row['has_numbers']}, dur={row['duration_s']:.0f}s")

print("\n  Common traits in bottom 5:")
print(f"    - has_payoff: {bot5['has_payoff'].mean()*100:.0f}% have it")
print(f"    - has_numbers: {bot5['has_numbers'].mean()*100:.0f}% have it")
print(f"    - hook_type distribution: {dict(bot5['hook_type'].value_counts())}")
print(f"    - avg duration: {bot5['duration_s'].mean():.0f}s")

# High engagement, low VPI (above median engagement, below median VPI)
med_eng = df['engagement_rate'].median()
med_vpi = df['VPI'].median()

high_eng_low_vpi = df[(df['engagement_rate'] > med_eng) & (df['VPI'] < med_vpi)]
print(f"\n--- High Engagement + Low VPI (n={len(high_eng_low_vpi)}) ---")
if len(high_eng_low_vpi) > 0:
    print(f"  hook_type distribution: {dict(high_eng_low_vpi['hook_type'].value_counts())}")
    print(f"  has_payoff: {high_eng_low_vpi['has_payoff'].mean()*100:.0f}%")

# ============================================================================
# PART 8: MULTICOLLINEARITY CHECK
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: MULTICOLLINEARITY CHECK")
print("="*80)

print("\n--- High Correlations Between Features (|r| > 0.7) ---")
corr_matrix = df[quant_features].corr()
high_corr = []
for i, feat1 in enumerate(quant_features):
    for j, feat2 in enumerate(quant_features):
        if i < j:
            r = corr_matrix.loc[feat1, feat2]
            if abs(r) > 0.7:
                high_corr.append((feat1, feat2, r))

if high_corr:
    for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {f1:25s} <-> {f2:25s}: r={r:.4f}")
else:
    print("  No feature pairs with |r| > 0.7 found")

print("\n--- Moderate Correlations (0.5 < |r| < 0.7) ---")
mod_corr = []
for i, feat1 in enumerate(quant_features):
    for j, feat2 in enumerate(quant_features):
        if i < j:
            r = corr_matrix.loc[feat1, feat2]
            if 0.5 < abs(r) <= 0.7:
                mod_corr.append((feat1, feat2, r))

for f1, f2, r in sorted(mod_corr, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {f1:25s} <-> {f2:25s}: r={r:.4f}")

# ============================================================================
# PART 9: NON-LINEAR RELATIONSHIPS (BINNED ANALYSIS)
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: NON-LINEAR RELATIONSHIPS (QUARTILE ANALYSIS)")
print("="*80)

for feature in ['duration_s', 'wpm', 'hook_wpm', 'reading_level', 'filler_density']:
    print(f"\n--- {feature} Quartile Analysis ---")
    # Build quantile bins defensively in case of duplicate edges (flat distributions)
    quantiles = np.quantile(df[feature].dropna(), np.linspace(0, 1, 5))
    quantile_edges = np.unique(quantiles)
    n_bins = len(quantile_edges) - 1
    if n_bins < 2:
        print("  Not enough unique values to compute quartiles; skipping.")
        continue
    labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'][:n_bins]
    df[f'{feature}_quartile'] = pd.cut(
        df[feature],
        bins=quantile_edges,
        labels=labels,
        include_lowest=True,
        duplicates='drop'
    )
    
    quartile_stats = df.groupby(f'{feature}_quartile')['VPI'].agg(['count', 'mean', 'median']).round(3)
    for q in quartile_stats.index:
        row = quartile_stats.loc[q]
        print(f"  {q:12s}: n={int(row['count']):2d}, mean={row['mean']:>10,.3f}, median={row['median']:>10,.3f}")
    
    # Check for non-linear pattern (only when full 4-bin breakdown exists)
    means = [df[df[f'{feature}_quartile'] == q]['VPI'].mean() for q in labels if q in df[f'{feature}_quartile'].values]
    if len(means) == 4:
        if means[1] > means[0] and means[1] > means[3]:
            print("  → Possible inverted-U pattern (peaks at Q2)")
        elif means[2] > means[1] and means[2] > means[3]:
            print("  → Possible inverted-U pattern (peaks at Q3)")
        elif means[0] > means[1] > means[2] > means[3]:
            print("  → Linear negative relationship")
        elif means[0] < means[1] < means[2] < means[3]:
            print("  → Linear positive relationship")
        else:
            print("  → No clear linear/non-linear pattern")

# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("EXECUTIVE SUMMARY")
print("="*80)

# Calculate key findings (using VPI)
payoff_yes = df[df['has_payoff'] == 1]['VPI'].mean()
payoff_no = df[df['has_payoff'] == 0]['VPI'].mean()
payoff_lift = ((payoff_yes - payoff_no) / payoff_no * 100) if payoff_no > 0 else 0

examples_yes = df[df['has_examples'] == 1]['VPI'].mean()
examples_no = df[df['has_examples'] == 0]['VPI'].mean()
examples_lift = ((examples_yes - examples_no) / examples_no * 100) if examples_no > 0 else 0

numbers_yes = df[df['has_numbers'] == 1]['VPI'].mean()
numbers_no = df[df['has_numbers'] == 0]['VPI'].mean()
numbers_lift = ((numbers_yes - numbers_no) / numbers_no * 100) if numbers_no > 0 else 0

# Bold claims vs story tease
bold_claim_views = df[df['hook_type'] == 'bold_claim']['VPI'].mean()
story_tease_views = df[df['hook_type'] == 'story_tease']['VPI'].mean()
bold_vs_story = ((bold_claim_views - story_tease_views) / story_tease_views * 100) if story_tease_views > 0 else 0

print(f"""
KEY FINDINGS:

1. BINARY FEATURES IMPACT:
   • has_payoff: {payoff_lift:+.1f}% lift (videos with payoff {'perform better' if payoff_lift > 0 else 'perform worse'})
   • has_examples: {examples_lift:+.1f}% lift (examples {'help' if examples_lift > 0 else 'hurt'} performance)
   • has_numbers: {numbers_lift:+.1f}% lift (numbers {'boost' if numbers_lift > 0 else 'reduce'} normalized views)

2. HOOK TYPE PERFORMANCE (VPI):
   • Bold claims avg: {bold_claim_views:.3f} VPI
   • Story tease avg: {story_tease_views:.3f} VPI
   • Bold claims {'+' if bold_vs_story > 0 else ''}{bold_vs_story:.1f}% vs story tease

3. TOP CORRELATIONS WITH VPI:
""")

view_corr_sorted = corr_df[corr_df['Metric'] == 'VPI'].sort_values('Pearson_r', key=abs, ascending=False)
for i, (_, row) in enumerate(view_corr_sorted.head(3).iterrows(), 1):
    print(f"   {i}. {row['Feature']}: r={row['Pearson_r']:+.4f} ({row['Interpretation']})")

print(f"""
4. TOP PERFORMERS PROFILE:
   • Avg duration: {top5['duration_s'].mean():.0f}s (vs dataset avg {df['duration_s'].mean():.0f}s)
   • 100% have payoff: {top5['has_payoff'].mean() == 1}
   • Dominant hook types: {', '.join(top5['hook_type'].mode().tolist())}

5. SAMPLE SIZE CAVEAT:
   • n={len(df)} videos from single channel
   • Some categories have <5 samples
   • Results should be validated on larger dataset

ACTIONABLE RECOMMENDATIONS:
""")

# Generate recommendations based on data
recommendations = []
if payoff_lift > 10:
    recommendations.append(f"✓ Always include a payoff/resolution ({payoff_lift:+.0f}% lift)")
if examples_lift < -10:
    recommendations.append(f"✓ Avoid examples in hooks ({examples_lift:.0f}% when used)")
elif examples_lift > 10:
    recommendations.append(f"✓ Include examples ({examples_lift:+.0f}% lift)")
if numbers_lift > 10:
    recommendations.append(f"✓ Use numbers/stats in content ({numbers_lift:+.0f}% lift)")
if bold_vs_story > 20:
    recommendations.append(f"✓ Prefer bold claims over story teases ({bold_vs_story:+.0f}% better)")
elif bold_vs_story < -20:
    recommendations.append(f"✓ Prefer story teases over bold claims ({-bold_vs_story:+.0f}% better)")

for rec in recommendations:
    print(f"   {rec}")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
