"""
==============================================================================
Step 3: Bias Detection & Statistical Analysis
==============================================================================
Deep statistical analysis of sentiment scores to prove bias exists.
Includes t-tests, effect sizes, intersectional analysis.
==============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. LOAD SCORED DATA
# =============================================================================

def load_scored_data(path=None):
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "02_Data",
            "sentiment_scores_all_systems.csv"
        )
    df = pd.read_csv(path)
    print(f"Loaded scored dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# 2. COMPREHENSIVE BIAS CALCULATION
# =============================================================================

def calculate_bias_by_demographic(df, score_col, baseline_group="White_Male"):
    """Calculate bias for each demographic group relative to baseline."""
    group_stats = df.groupby("Demographic_Group")[score_col].agg(
        ["mean", "std", "count"]
    )
    baseline_mean = group_stats.loc[baseline_group, "mean"]

    results = []
    for group in group_stats.index:
        bias = group_stats.loc[group, "mean"] - baseline_mean
        pct_bias = (bias / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
        results.append({
            "Demographic_Group": group,
            "Mean_Score": group_stats.loc[group, "mean"],
            "Std_Dev": group_stats.loc[group, "std"],
            "Count": int(group_stats.loc[group, "count"]),
            "Bias_vs_Baseline": bias,
            "Pct_Bias": pct_bias,
        })
    return pd.DataFrame(results)


def calculate_bias_by_race(df, score_col, baseline="White"):
    """Calculate bias by race."""
    group_stats = df.groupby("Race")[score_col].agg(["mean", "std", "count"])
    baseline_mean = group_stats.loc[baseline, "mean"]

    results = []
    for race in group_stats.index:
        bias = group_stats.loc[race, "mean"] - baseline_mean
        results.append({
            "Race": race,
            "Mean_Score": group_stats.loc[race, "mean"],
            "Bias_vs_White": bias,
        })
    return pd.DataFrame(results)


def calculate_bias_by_category(df, score_col):
    """Calculate bias broken down by emotion category."""
    results = []
    for category in df["Template_Category"].unique():
        cat_df = df[df["Template_Category"] == category]
        group_means = cat_df.groupby("Race")[score_col].mean()
        white_mean = group_means.get("White", 0)
        for race in group_means.index:
            bias = group_means[race] - white_mean
            results.append({
                "Category": category,
                "Race": race,
                "Mean_Score": group_means[race],
                "Bias_vs_White": bias,
            })
    return pd.DataFrame(results)


# =============================================================================
# 3. STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

def run_statistical_tests(df, score_col):
    """
    Run t-tests comparing each demographic group to the baseline (White_Male).
    Also computes Cohen's d effect size.
    """
    baseline_scores = df[df["Demographic_Group"] == "White_Male"][score_col]

    results = []
    for group in df["Demographic_Group"].unique():
        if group == "White_Male":
            continue

        group_scores = df[df["Demographic_Group"] == group][score_col]

        # Independent t-test
        t_stat, p_value = stats.ttest_ind(baseline_scores, group_scores)

        # Cohen's d (effect size)
        pooled_std = np.sqrt(
            (baseline_scores.std() ** 2 + group_scores.std() ** 2) / 2
        )
        cohens_d = (baseline_scores.mean() - group_scores.mean()) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p_value = stats.mannwhitneyu(
            baseline_scores, group_scores, alternative="two-sided"
        )

        results.append({
            "Demographic_Group": group,
            "Baseline_Mean": baseline_scores.mean(),
            "Group_Mean": group_scores.mean(),
            "Difference": baseline_scores.mean() - group_scores.mean(),
            "T_Statistic": t_stat,
            "P_Value": p_value,
            "Significant_005": p_value < 0.05,
            "Significant_001": p_value < 0.001,
            "Cohens_d": cohens_d,
            "Effect_Size_Label": (
                "Large" if abs(cohens_d) >= 0.8 else
                "Medium" if abs(cohens_d) >= 0.5 else
                "Small" if abs(cohens_d) >= 0.2 else
                "Negligible"
            ),
            "MannWhitney_U": u_stat,
            "MannWhitney_P": u_p_value,
        })

    return pd.DataFrame(results)


def run_anova_by_race(df, score_col):
    """One-way ANOVA: tests if ANY race group differs significantly."""
    groups = [group[score_col].values for _, group in df.groupby("Race")]
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value


def run_two_way_analysis(df, score_col):
    """Analyze interaction between race and gender."""
    results = []
    for race in df["Race"].unique():
        for gender in df["Gender"].unique():
            subset = df[(df["Race"] == race) & (df["Gender"] == gender)]
            results.append({
                "Race": race,
                "Gender": gender,
                "Mean_Score": subset[score_col].mean(),
                "Std_Dev": subset[score_col].std(),
                "Count": len(subset),
            })
    return pd.DataFrame(results)


# =============================================================================
# 4. INTERSECTIONAL ANALYSIS
# =============================================================================

def intersectional_analysis(df, score_col):
    """Analyze how race and gender INTERACT to create compounding bias."""
    print("\n  INTERSECTIONAL ANALYSIS")
    print("  " + "-" * 50)

    pivot = df.pivot_table(
        values=score_col,
        index="Race",
        columns="Gender",
        aggfunc="mean"
    )
    print(f"\n  Mean {score_col} (Race × Gender):")
    print(pivot.to_string())

    # Interaction effects
    grand_mean = df[score_col].mean()
    race_effects = df.groupby("Race")[score_col].mean() - grand_mean
    gender_effects = df.groupby("Gender")[score_col].mean() - grand_mean

    print(f"\n  Main Effects:")
    print(f"    Race effects (deviation from grand mean {grand_mean:.4f}):")
    for race, effect in race_effects.items():
        print(f"      {race:10s}: {effect:+.4f}")
    print(f"    Gender effects:")
    for gender, effect in gender_effects.items():
        print(f"      {gender:10s}: {effect:+.4f}")

    return pivot


# =============================================================================
# 5. PATTERN ANALYSIS BY EMOTION CATEGORY
# =============================================================================

def emotion_bias_analysis(df, score_col):
    """Analyze if bias varies by type of emotion expressed."""
    print(f"\n  BIAS BY EMOTION CATEGORY ({score_col}):")
    print("  " + "-" * 50)

    for category in sorted(df["Template_Category"].unique()):
        cat_df = df[df["Template_Category"] == category]
        white_mean = cat_df[cat_df["Race"] == "White"][score_col].mean()
        black_mean = cat_df[cat_df["Race"] == "Black"][score_col].mean()
        bias = black_mean - white_mean
        print(f"    {category:15s}: White={white_mean:+.4f}, "
              f"Black={black_mean:+.4f}, Bias={bias:+.4f}"
              f"{'  ⚠' if abs(bias) > 0.03 else ''}")


# =============================================================================
# 6. FULL ANALYSIS REPORT
# =============================================================================

def run_full_analysis(df):
    """Run complete bias detection analysis."""
    systems = [
        ("VADER", "VADER_compound"),
        ("TextBlob", "TextBlob_polarity"),
        ("BERT", "BERT_score"),
    ]

    all_bias_results = {}
    all_stat_results = {}

    for system_name, score_col in systems:
        print("\n" + "=" * 60)
        print(f"  ANALYSIS FOR: {system_name}")
        print("=" * 60)

        # Bias by demographic
        bias_df = calculate_bias_by_demographic(df, score_col)
        all_bias_results[system_name] = bias_df
        print(f"\n  Bias by Demographic Group (vs White_Male):")
        for _, row in bias_df.sort_values("Bias_vs_Baseline").iterrows():
            flag = "⚠ BIASED" if abs(row["Bias_vs_Baseline"]) > 0.03 else ""
            print(f"    {row['Demographic_Group']:20s}: "
                  f"mean={row['Mean_Score']:+.4f}, "
                  f"bias={row['Bias_vs_Baseline']:+.4f} "
                  f"({row['Pct_Bias']:+.1f}%) {flag}")

        # Statistical tests
        stat_df = run_statistical_tests(df, score_col)
        all_stat_results[system_name] = stat_df
        print(f"\n  Statistical Significance (t-test vs White_Male):")
        for _, row in stat_df.iterrows():
            sig = "***" if row["Significant_001"] else ("*" if row["Significant_005"] else "ns")
            print(f"    {row['Demographic_Group']:20s}: "
                  f"t={row['T_Statistic']:+.3f}, p={row['P_Value']:.4f} {sig}, "
                  f"d={row['Cohens_d']:+.3f} ({row['Effect_Size_Label']})")

        # ANOVA
        f_stat, p_val = run_anova_by_race(df, score_col)
        print(f"\n  One-way ANOVA (Race): F={f_stat:.3f}, p={p_val:.6f}")
        print(f"  {'Significant difference across races!' if p_val < 0.05 else 'No significant difference.'}")

        # Intersectional
        intersectional_analysis(df, score_col)

        # Emotion bias
        emotion_bias_analysis(df, score_col)

    return all_bias_results, all_stat_results


# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

def save_analysis_results(all_bias, all_stats, output_dir):
    """Save all analysis results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for system, bias_df in all_bias.items():
        bias_df.to_csv(
            os.path.join(output_dir, f"bias_by_demographic_{system}.csv"),
            index=False
        )
    for system, stat_df in all_stats.items():
        stat_df.to_csv(
            os.path.join(output_dir, f"statistical_tests_{system}.csv"),
            index=False
        )
    print(f"\nAnalysis results saved to: {output_dir}")


# =============================================================================
# 8. MAIN
# =============================================================================

if __name__ == "__main__":
    df = load_scored_data()
    all_bias, all_stats = run_full_analysis(df)

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "04_Results"
    )
    save_analysis_results(all_bias, all_stats, output_dir)
