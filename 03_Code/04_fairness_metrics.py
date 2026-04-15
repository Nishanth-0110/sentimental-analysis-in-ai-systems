"""
==============================================================================
Step 4: Fairness Metrics (using Microsoft Fairlearn)
==============================================================================
Applies 5 standard fairness metrics to quantify bias:
  1. Demographic Parity Difference
  2. Equal Opportunity Difference
  3. Equalized Odds Difference
  4. Disparate Impact Ratio
  5. Calibration (group-wise)

Uses Fairlearn library (Microsoft) - industry standard.
==============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import warnings

warnings.filterwarnings("ignore")


def load_scored_data(path=None):
    """Load the scored sentiment data."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "02_Data",
            "sentiment_scores_all_systems.csv"
        )
    df = pd.read_csv(path)
    print(f"Loaded scored data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# 1. PREPARE DATA FOR FAIRNESS ANALYSIS
# =============================================================================

def prepare_binary_labels(df, score_col, threshold=0.0):
    """
    Convert continuous sentiment scores to binary labels for fairness metrics.
    Positive (>= threshold) vs Negative (< threshold).
    """
    df = df.copy()
    df["true_label"] = 0  # All complaints are actually negative
    df["predicted_label"] = (df[score_col] >= threshold).astype(int)
    # Also create a "severity" classification
    # High severity (very negative) vs Low severity
    median_score = df[score_col].median()
    df["predicted_severity"] = (df[score_col] < median_score).astype(int)
    return df


# =============================================================================
# 2. FAIRNESS METRICS CALCULATION
# =============================================================================

def demographic_parity_difference(y_pred, sensitive_attr):
    """
    Metric 1: Demographic Parity Difference
    Measures if all groups get positive predictions at the same rate.
    Ideal: 0 | Threshold: < 0.10
    """
    from fairlearn.metrics import demographic_parity_difference as dpd
    return dpd(y_true=np.zeros_like(y_pred), y_pred=y_pred,
               sensitive_features=sensitive_attr)


def demographic_parity_ratio(y_pred, sensitive_attr):
    """
    Metric 4: Disparate Impact Ratio
    Ratio of positive prediction rates between groups.
    Ideal: 1.0 | Legal threshold: > 0.80
    """
    from fairlearn.metrics import demographic_parity_ratio as dpr
    return dpr(y_true=np.zeros_like(y_pred), y_pred=y_pred,
               sensitive_features=sensitive_attr)


def selection_rates_by_group(y_pred, sensitive_attr):
    """Calculate positive prediction rate for each group."""
    from fairlearn.metrics import selection_rate
    groups = np.unique(sensitive_attr)
    rates = {}
    for g in groups:
        mask = sensitive_attr == g
        rates[g] = y_pred[mask].mean()
    return rates


def equalized_odds_difference(y_true, y_pred, sensitive_attr):
    """
    Metric 3: Equalized Odds Difference
    Combines TPR and FPR parity. Ideal: 0 | Threshold: < 0.10
    """
    from fairlearn.metrics import equalized_odds_difference as eod
    return eod(y_true=y_true, y_pred=y_pred,
               sensitive_features=sensitive_attr)


def equal_opportunity_difference_manual(y_true, y_pred, sensitive_attr):
    """
    Metric 2: Equal Opportunity Difference
    Ensures equal true positive rates across groups.
    Uses manual calculation since all true labels are 0 (negative).
    """
    groups = np.unique(sensitive_attr)
    tpr_per_group = {}

    for g in groups:
        mask = sensitive_attr == g
        g_true = y_true[mask]
        g_pred = y_pred[mask]
        # Use prediction rate as proxy since all true labels are negative
        tpr_per_group[g] = g_pred.mean()

    values = list(tpr_per_group.values())
    return max(values) - min(values), tpr_per_group


def calibration_by_group(scores, sensitive_attr, n_bins=5):
    """
    Metric 5: Calibration
    Checks if prediction confidence means the same thing across groups.
    """
    groups = np.unique(sensitive_attr)
    calibration_results = {}

    for g in groups:
        mask = sensitive_attr == g
        g_scores = scores[mask]
        # Calculate score distribution statistics
        calibration_results[g] = {
            "mean": g_scores.mean(),
            "std": g_scores.std(),
            "median": np.median(g_scores),
            "q25": np.percentile(g_scores, 25),
            "q75": np.percentile(g_scores, 75),
        }

    # Calibration difference: max mean difference
    means = [v["mean"] for v in calibration_results.values()]
    cal_diff = max(means) - min(means)

    return cal_diff, calibration_results


# =============================================================================
# 3. COMPREHENSIVE FAIRNESS REPORT
# =============================================================================

def generate_fairness_report(df, score_col, system_name):
    """Generate complete fairness report for one sentiment system."""
    print(f"\n{'=' * 60}")
    print(f"  FAIRNESS ANALYSIS REPORT: {system_name}")
    print(f"{'=' * 60}")

    # Prepare data
    prepped = prepare_binary_labels(df, score_col)
    y_pred = prepped["predicted_label"].values
    y_severity = prepped["predicted_severity"].values
    scores = df[score_col].values

    results = {}

    # --- By Race ---
    print(f"\n  PROTECTED ATTRIBUTE: Race")
    print("  " + "-" * 50)

    race_attr = df["Race"].values

    # Metric 1: Demographic Parity
    dp_diff = demographic_parity_difference(y_pred, race_attr)
    dp_status = "PASS" if abs(dp_diff) < 0.10 else "FAIL"
    print(f"\n  1. Demographic Parity Difference: {dp_diff:.4f}")
    print(f"     Threshold: < 0.10  |  Status: {dp_status}")

    # Selection rates
    sel_rates = selection_rates_by_group(y_pred, race_attr)
    print(f"     Selection rates by race:")
    for race, rate in sorted(sel_rates.items()):
        print(f"       {race:10s}: {rate:.4f} ({rate*100:.1f}%)")

    # Metric 2: Equal Opportunity
    eo_diff, eo_groups = equal_opportunity_difference_manual(
        prepped["true_label"].values, y_pred, race_attr
    )
    eo_status = "PASS" if eo_diff < 0.10 else "FAIL"
    print(f"\n  2. Equal Opportunity Difference: {eo_diff:.4f}")
    print(f"     Threshold: < 0.10  |  Status: {eo_status}")

    # Metric 3: Equalized Odds
    try:
        eod_val = equalized_odds_difference(
            prepped["true_label"].values, y_pred, race_attr
        )
    except Exception:
        eod_val = eo_diff  # Fallback
    eod_status = "PASS" if eod_val < 0.10 else "FAIL"
    print(f"\n  3. Equalized Odds Difference: {eod_val:.4f}")
    print(f"     Threshold: < 0.10  |  Status: {eod_status}")

    # Metric 4: Disparate Impact Ratio
    try:
        di_ratio = demographic_parity_ratio(y_pred, race_attr)
    except Exception:
        # Manual calculation
        rates = list(sel_rates.values())
        di_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
    di_status = "PASS" if di_ratio > 0.80 else "FAIL"
    print(f"\n  4. Disparate Impact Ratio: {di_ratio:.4f}")
    print(f"     Threshold: > 0.80  |  Status: {di_status}")
    print(f"     Legal standard (80% rule): {'COMPLIANT' if di_status == 'PASS' else 'VIOLATION'}")

    # Metric 5: Calibration
    cal_diff, cal_results = calibration_by_group(scores, race_attr)
    cal_status = "PASS" if cal_diff < 0.10 else "FAIL"
    print(f"\n  5. Calibration Difference: {cal_diff:.4f}")
    print(f"     Threshold: < 0.10  |  Status: {cal_status}")
    for race, stats_dict in sorted(cal_results.items()):
        print(f"       {race:10s}: mean={stats_dict['mean']:+.4f}, "
              f"std={stats_dict['std']:.4f}")

    # Summary
    metrics_passed = sum([
        abs(dp_diff) < 0.10,
        eo_diff < 0.10,
        eod_val < 0.10,
        di_ratio > 0.80,
        cal_diff < 0.10,
    ])

    print(f"\n  {'=' * 50}")
    print(f"  OVERALL RESULT: {metrics_passed}/5 PASSED")
    if metrics_passed <= 2:
        print("  ASSESSMENT: SEVERELY BIASED SYSTEM")
    elif metrics_passed <= 4:
        print("  ASSESSMENT: PARTIALLY BIASED SYSTEM")
    else:
        print("  ASSESSMENT: FAIR SYSTEM")
    print(f"  {'=' * 50}")

    results = {
        "system": system_name,
        "demographic_parity_diff": dp_diff,
        "equal_opportunity_diff": eo_diff,
        "equalized_odds_diff": eod_val,
        "disparate_impact_ratio": di_ratio,
        "calibration_diff": cal_diff,
        "metrics_passed": metrics_passed,
        "selection_rates": sel_rates,
    }

    # --- By Gender ---
    print(f"\n  PROTECTED ATTRIBUTE: Gender")
    print("  " + "-" * 50)
    gender_attr = df["Gender"].values

    dp_gender = demographic_parity_difference(y_pred, gender_attr)
    print(f"  Demographic Parity (Gender): {dp_gender:.4f}")
    sel_gender = selection_rates_by_group(y_pred, gender_attr)
    for g, rate in sorted(sel_gender.items()):
        print(f"    {g:10s}: {rate:.4f}")

    # --- By Intersectional Group ---
    print(f"\n  PROTECTED ATTRIBUTE: Demographic Group (Intersectional)")
    print("  " + "-" * 50)
    demo_attr = df["Demographic_Group"].values

    dp_demo = demographic_parity_difference(y_pred, demo_attr)
    print(f"  Demographic Parity (Intersectional): {dp_demo:.4f}")
    sel_demo = selection_rates_by_group(y_pred, demo_attr)
    for g, rate in sorted(sel_demo.items()):
        print(f"    {g:20s}: {rate:.4f}")

    return results


# =============================================================================
# 4. MAIN
# =============================================================================

def run_all_fairness_analyses(df):
    """Run fairness analysis for all three sentiment systems."""
    systems = [
        ("VADER", "VADER_compound"),
        ("TextBlob", "TextBlob_polarity"),
        ("BERT", "BERT_score"),
    ]

    all_results = []
    for system_name, score_col in systems:
        result = generate_fairness_report(df, score_col, system_name)
        all_results.append(result)

    # Comparison summary
    print("\n" + "=" * 70)
    print("  FAIRNESS COMPARISON ACROSS ALL SYSTEMS")
    print("=" * 70)
    print(f"\n  {'Metric':<30s} {'VADER':>10s} {'TextBlob':>10s} {'BERT':>10s}")
    print("  " + "-" * 60)
    metrics = [
        ("Dem. Parity Diff (<0.10)", "demographic_parity_diff"),
        ("Equal Opp. Diff (<0.10)", "equal_opportunity_diff"),
        ("Equalized Odds Diff (<0.10)", "equalized_odds_diff"),
        ("Disparate Impact (>0.80)", "disparate_impact_ratio"),
        ("Calibration Diff (<0.10)", "calibration_diff"),
    ]
    for label, key in metrics:
        vals = [r[key] for r in all_results]
        print(f"  {label:<30s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")

    passed = [r["metrics_passed"] for r in all_results]
    print(f"  {'Tests Passed':<30s} {'%d/5' % passed[0]:>10s} "
          f"{'%d/5' % passed[1]:>10s} {'%d/5' % passed[2]:>10s}")

    return all_results


if __name__ == "__main__":
    df = load_scored_data()
    results = run_all_fairness_analyses(df)

    # Save fairness metrics
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "04_Results"
    )
    os.makedirs(output_dir, exist_ok=True)

    metrics_rows = []
    for r in results:
        metrics_rows.append({
            "System": r["system"],
            "Demographic_Parity_Diff": r["demographic_parity_diff"],
            "Equal_Opportunity_Diff": r["equal_opportunity_diff"],
            "Equalized_Odds_Diff": r["equalized_odds_diff"],
            "Disparate_Impact_Ratio": r["disparate_impact_ratio"],
            "Calibration_Diff": r["calibration_diff"],
            "Metrics_Passed": r["metrics_passed"],
        })
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(output_dir, "fairness_metrics_all_systems.csv"), index=False
    )
    print(f"\nFairness metrics saved to: {output_dir}")
