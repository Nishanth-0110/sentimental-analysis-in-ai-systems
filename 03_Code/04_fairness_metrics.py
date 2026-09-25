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

    Two binary views are produced:
    - ``predicted_label``: sign of the sentiment score (>= threshold). Used for
      Demographic Parity and Disparate Impact, which only need predictions.
    - ``true_label`` / ``predicted_severity``: severity split at the VADER
      median (a name-agnostic reference) vs the system's own median split.
      These are used for Equal Opportunity / Equalized Odds, which require a
      non-degenerate ground-truth label. Using an all-zero ground truth would
      make TPR/FPR undefined; the VADER median split is the same proxy label
      used by the mitigation (05), privacy (07) and demo (07_Demo) code.
    """
    df = df.copy()
    df["predicted_label"] = (df[score_col] >= threshold).astype(int)
    median_score = df[score_col].median()
    df["predicted_severity"] = (df[score_col] >= median_score).astype(int)
    # Proxy ground truth: VADER is rule-based and name-agnostic, so a
    # VADER-median split is a fair reference severity label.
    vader_median = df["VADER_compound"].median() if "VADER_compound" in df.columns else median_score
    df["true_label"] = (df["VADER_compound"].to_numpy() >= vader_median).astype(int) \
        if "VADER_compound" in df.columns else (df[score_col].to_numpy() >= median_score).astype(int)
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
    y_true = prepped["true_label"].values          # VADER-median proxy labels
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

    # Metric 2: Equal Opportunity (TPR parity on the severity task:
    # proxy truth = VADER median split, prediction = system median split)
    try:
        from fairlearn.metrics import MetricFrame, true_positive_rate
        eo_diff = float(MetricFrame(
            metrics=true_positive_rate,
            y_true=y_true,
            y_pred=y_severity,
            sensitive_features=race_attr,
        ).difference())
    except Exception:
        eo_diff = float("nan")
    eo_status = "PASS" if eo_diff < 0.10 else "FAIL"
    print(f"\n  2. Equal Opportunity Difference (TPR parity, severity task): {eo_diff:.4f}")
    print(f"     Threshold: < 0.10  |  Status: {eo_status}")

    # Metric 3: Equalized Odds (TPR + FPR parity on the severity task)
    try:
        eod_val = float(equalized_odds_difference(
            y_true, y_severity, race_attr
        ))
    except Exception:
        eod_val = eo_diff  # Fallback
    eod_status = "PASS" if eod_val < 0.10 else "FAIL"
    print(f"\n  3. Equalized Odds Difference (severity task): {eod_val:.4f}")
    print(f"     Threshold: < 0.10  |  Status: {eod_status}")

    # Metric 4: Disparate Impact Ratio
    try:
        di_ratio = demographic_parity_ratio(y_pred, race_attr)
    except Exception:
        # Manual calculation
        rates = list(sel_rates.values())
        di_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
    # Degenerate case: if (almost) no group receives a positive prediction,
    # the ratio is driven by rounding noise and is not a meaningful signal.
    di_degenerate = max(sel_rates.values()) <= 0.01 if sel_rates else True
    di_status = "PASS" if di_ratio > 0.80 else ("N/A" if di_degenerate else "FAIL")
    print(f"\n  4. Disparate Impact Ratio: {di_ratio:.4f}")
    if di_degenerate:
        print("     Threshold: > 0.80  |  Status: N/A (degenerate: ~0 positive selections in all groups)")
    else:
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
    checks = [
        abs(dp_diff) < 0.10,
        eo_diff < 0.10,
        eod_val < 0.10,
        cal_diff < 0.10,
    ]
    if not di_degenerate:
        checks.append(di_ratio > 0.80)
    metrics_passed = sum(checks)
    metrics_total = len(checks)

    print(f"\n  {'=' * 50}")
    print(f"  OVERALL RESULT: {metrics_passed}/{metrics_total} PASSED"
          + (" (DI excluded: degenerate)" if di_degenerate else ""))
    if metrics_passed == metrics_total:
        print("  ASSESSMENT: FAIR SYSTEM")
    elif metrics_passed / metrics_total <= 0.4:
        print("  ASSESSMENT: SEVERELY BIASED SYSTEM")
    else:
        print("  ASSESSMENT: PARTIALLY BIASED SYSTEM")
    print(f"  {'=' * 50}")

    results = {
        "system": system_name,
        "demographic_parity_diff": dp_diff,
        "equal_opportunity_diff": eo_diff,
        "equalized_odds_diff": eod_val,
        "disparate_impact_ratio": di_ratio,
        "disparate_impact_degenerate": di_degenerate,
        "calibration_diff": cal_diff,
        "metrics_passed": metrics_passed,
        "metrics_total": metrics_total,
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
    """Run fairness analysis for all available sentiment systems."""
    systems = [
        ("VADER", "VADER_compound"),
        ("TextBlob", "TextBlob_polarity"),
        ("BERT", "BERT_score"),
        ("RoBERTa", "RoBERTa_score"),
    ]

    all_results = []
    for system_name, score_col in systems:
        if score_col not in df.columns:
            print(f"\n  Skipping {system_name}: column '{score_col}' not in dataset.")
            continue
        result = generate_fairness_report(df, score_col, system_name)
        all_results.append(result)

    # Comparison summary
    print("\n" + "=" * 70)
    print("  FAIRNESS COMPARISON ACROSS ALL SYSTEMS")
    print("=" * 70)
    header = f"  {'Metric':<30s}" + "".join(f"{r['system']:>10s}" for r in all_results)
    print(f"\n{header}")
    print("  " + "-" * (30 + 10 * len(all_results)))
    metrics = [
        ("Dem. Parity Diff (<0.10)", "demographic_parity_diff"),
        ("Equal Opp. Diff (<0.10)", "equal_opportunity_diff"),
        ("Equalized Odds Diff (<0.10)", "equalized_odds_diff"),
        ("Disparate Impact (>0.80)", "disparate_impact_ratio"),
        ("Calibration Diff (<0.10)", "calibration_diff"),
    ]
    for label, key in metrics:
        row = f"  {label:<30s}" + "".join(f"{r[key]:>10.4f}" for r in all_results)
        print(row)

    passed = [f"{r['metrics_passed']}/{r['metrics_total']}" for r in all_results]
    print(f"  {'Tests Passed':<30s}" + "".join(f"{p:>10s}" for p in passed))

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
            "DI_Degenerate": r.get("disparate_impact_degenerate", False),
            "Calibration_Diff": r["calibration_diff"],
            "Metrics_Passed": r["metrics_passed"],
        })
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(output_dir, "fairness_metrics_all_systems.csv"), index=False
    )
    print(f"\nFairness metrics saved to: {output_dir}")
