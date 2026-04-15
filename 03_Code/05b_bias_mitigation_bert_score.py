"""
==============================================================================
Step 5B: Bias Mitigation DIRECTLY on BERT outputs (recommended)
==============================================================================
Why this file exists:
- The original mitigation step (05_bias_mitigation.py) trains a TF-IDF + Logistic
  Regression proxy model with VADER-derived labels, then applies mitigation.
  That is NOT the same as mitigating BERT's name-based bias.

What this script does:
- Uses the already-computed `BERT_score` as the model signal.
- Defines a proxy ground-truth label using VADER (rule-based, name-agnostic):
    y_true = 1 if VADER_compound >= median else 0
- Trains a simple 1-D classifier f(BERT_score) -> y_true.
  This makes post-processing equivalent to applying group-wise thresholds on
  BERT scores.
- Compares:
    1) Baseline 1-D model on BERT_score
    2) Reweighing (pre-processing) on the 1-D model
    3) ExponentiatedGradient (in-processing) on the 1-D model
    4) ThresholdOptimizer (post-processing) on the 1-D model

Outputs:
- 04_Results/mitigation_comparison_bert_score_constraints.csv
- 04_Results/mitigation_comparison_bert_score_constraints_with_deltas.csv
==============================================================================
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def _fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> dict:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

    dp_diff = demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    )

    # equalized_odds_difference requires y_true; if it errors, fall back to dp
    try:
        eo_diff = equalized_odds_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
        )
    except Exception:
        eo_diff = dp_diff

    # Disparate impact ratio based on selection rates per group
    groups = np.unique(sensitive)
    rates = {}
    for g in groups:
        mask = sensitive == g
        rates[g] = float(np.mean(y_pred[mask])) if np.any(mask) else 0.0
    rate_values = [v for v in rates.values() if v > 0]
    di_ratio = min(rate_values) / max(rate_values) if rate_values and max(rate_values) > 0 else 0.0

    return {
        "dem_parity_diff": float(abs(dp_diff)),
        "equalized_odds_diff": float(eo_diff),
        "disparate_impact": float(di_ratio),
        "selection_rates": rates,
    }


def _fit_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def _fit_reweighing(X_train: np.ndarray, y_train: np.ndarray, sensitive_train: np.ndarray) -> LogisticRegression:
    # Same reweighing idea as in 05_bias_mitigation.py, but for 1-D features.
    groups = np.unique(sensitive_train)
    n_total = len(y_train)
    sample_weights = np.ones(n_total, dtype=float)

    for label in [0, 1]:
        for g in groups:
            mask = (sensitive_train == g) & (y_train == label)
            n_group_label = int(mask.sum())
            n_group = int((sensitive_train == g).sum())
            n_label = int((y_train == label).sum())
            if n_group_label > 0:
                expected = (n_group * n_label) / n_total
                weight = expected / n_group_label
                sample_weights[mask] = weight

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def _fit_exponentiated_gradient(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    constraint_name: str = "demographic_parity",
):
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient

    # Project decision: keep only Demographic Parity (DP) as the mitigation constraint.
    # (We still compute Equalized Odds *metrics* for reporting, but we don't optimize for EO.)
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    if constraint_name != "demographic_parity":
        raise ValueError("Only Demographic Parity constraint is supported in this run.")
    constraints = DemographicParity()
    mitigator = ExponentiatedGradient(
        estimator=base_model,
        constraints=constraints,
        max_iter=50,
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator


def _fit_threshold_optimizer(
    prefit_estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    constraint_name: str = "demographic_parity",
):
    from fairlearn.postprocessing import ThresholdOptimizer

    post = ThresholdOptimizer(
        estimator=prefit_estimator,
        constraints=constraint_name,
        objective="accuracy_score",
        prefit=True,
    )
    post.fit(X_train, y_train, sensitive_features=sensitive_train)
    return post


def run():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "02_Data", "sentiment_scores_all_systems.csv")
    out_path = os.path.join(repo_root, "04_Results", "mitigation_comparison_bert_score_constraints.csv")
    out_path_deltas = os.path.join(repo_root, "04_Results", "mitigation_comparison_bert_score_constraints_with_deltas.csv")

    df = pd.read_csv(data_path)

    # Proxy ground-truth label: VADER (name-agnostic) median split
    vader_median = float(df["VADER_compound"].median())
    y_true = (df["VADER_compound"].to_numpy() >= vader_median).astype(int)

    # Feature: BERT score only (1-D)
    X = df[["BERT_score"]].to_numpy().astype(float)

    # Sensitive attribute (race) for fairness auditing
    sensitive = df["Race"].to_numpy()

    # Split
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y_true, sensitive, test_size=0.30, random_state=42, stratify=y_true
    )

    results = []

    # Baseline
    baseline = _fit_baseline(X_train, y_train)
    y_pred = baseline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    fm = _fairness_metrics(y_test, y_pred, sens_test)
    results.append({
        "method": "Baseline (BERT_score -> LR)",
        "accuracy": acc,
        **{k: fm[k] for k in ("dem_parity_diff", "equalized_odds_diff", "disparate_impact")},
    })

    # Reweighing
    reweigh = _fit_reweighing(X_train, y_train, sens_train)
    y_pred = reweigh.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    fm = _fairness_metrics(y_test, y_pred, sens_test)
    results.append({
        "method": "Reweighing (BERT_score -> LR)",
        "accuracy": acc,
        **{k: fm[k] for k in ("dem_parity_diff", "equalized_odds_diff", "disparate_impact")},
    })

    # ExponentiatedGradient with Demographic Parity
    eg_dp = _fit_exponentiated_gradient(X_train, y_train, sens_train, constraint_name="demographic_parity")
    y_pred = eg_dp.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    fm = _fairness_metrics(y_test, y_pred, sens_test)
    results.append({
        "method": "ExponentiatedGradient (Demographic Parity)",
        "accuracy": acc,
        **{k: fm[k] for k in ("dem_parity_diff", "equalized_odds_diff", "disparate_impact")},
    })

    # ThresholdOptimizer (post-processing) with Demographic Parity
    post_dp = _fit_threshold_optimizer(
        baseline, X_train, y_train, sens_train, constraint_name="demographic_parity"
    )
    y_pred = post_dp.predict(X_test, sensitive_features=sens_test)
    acc = float(accuracy_score(y_test, y_pred))
    fm = _fairness_metrics(y_test, y_pred, sens_test)
    results.append({
        "method": "ThresholdOptimizer (Demographic Parity)",
        "accuracy": acc,
        **{k: fm[k] for k in ("dem_parity_diff", "equalized_odds_diff", "disparate_impact")},
    })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)

    # Add baseline deltas so improvements are obvious at a glance.
    baseline_row = out_df.iloc[0]
    baseline_acc = float(baseline_row["accuracy"])
    baseline_dp = float(baseline_row["dem_parity_diff"])
    baseline_eo = float(baseline_row["equalized_odds_diff"])
    baseline_di = float(baseline_row["disparate_impact"])

    out_df_d = out_df.copy()
    out_df_d["delta_accuracy"] = out_df_d["accuracy"].astype(float) - baseline_acc
    out_df_d["delta_dem_parity_diff"] = out_df_d["dem_parity_diff"].astype(float) - baseline_dp
    out_df_d["delta_equalized_odds_diff"] = out_df_d["equalized_odds_diff"].astype(float) - baseline_eo
    out_df_d["delta_disparate_impact"] = out_df_d["disparate_impact"].astype(float) - baseline_di
    out_df_d["improves_dem_parity"] = out_df_d["delta_dem_parity_diff"] < 0
    out_df_d["improves_eq_odds"] = out_df_d["delta_equalized_odds_diff"] < 0
    out_df_d["improves_disp_impact"] = out_df_d["delta_disparate_impact"] > 0
    out_df_d.to_csv(out_path_deltas, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_deltas}")
    print(out_df)


if __name__ == "__main__":
    run()
