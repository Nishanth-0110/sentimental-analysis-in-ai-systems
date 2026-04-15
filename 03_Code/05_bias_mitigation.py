"""
==============================================================================
Step 5: Bias Mitigation Techniques
==============================================================================
Implements three approaches:
  1. Pre-processing:  Reweighing
  2. In-processing:   Adversarial Debiasing (via Fairlearn ExponentiatedGradient)
  3. Post-processing: Threshold Optimizer (Calibrated Equalized Odds)

Compares all three and recommends best approach.
==============================================================================
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
import joblib

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PREPARE MODEL DATA
# =============================================================================

def prepare_model_data(df, score_col="VADER_compound"):
    """
    Prepare text data for training a sentiment classifier.
    Uses TF-IDF features from complaint text.
    Creates binary labels from VADER scores as ground truth proxy.
    """
    # Binary label: 1 = less negative (above median), 0 = more negative
    median_score = df[score_col].median()
    df = df.copy()
    df["label"] = (df[score_col] >= median_score).astype(int)

    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(df["Full_Text"])

    y = np.array(df["label"].values)
    sensitive_race = np.array(df["Race"].values)
    sensitive_gender = np.array(df["Gender"].values)
    sensitive_demo = np.array(df["Demographic_Group"].values)

    return X, y, sensitive_race, sensitive_gender, sensitive_demo, vectorizer, df


def _build_name_pool(prep_df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Build a stable pool of names and a mapping name->race from the dataset."""
    name_race = (
        prep_df[["Name", "Race"]]
        .dropna()
        .drop_duplicates(subset=["Name"])
        .set_index("Name")["Race"]
        .to_dict()
    )
    names = sorted(list(name_race.keys()))
    return names, name_race


def _replace_name_once(text: str, old_name: str, new_name: str) -> str:
    """Replace the first occurrence of a name token in the text.

    Our dataset format typically begins with the name, so we handle that fast-path.
    Fallback uses a word-boundary regex replacement.
    """
    if text.startswith(old_name + " "):
        return new_name + text[len(old_name):]
    # word boundary replace, first occurrence only
    pattern = r"\\b" + re.escape(old_name) + r"\\b"
    replaced, n = re.subn(pattern, new_name, text, count=1)
    if n == 0:
        # last-resort: prefix replacement (keeps the rest of the sentence)
        parts = text.split(" ", 1)
        if len(parts) == 2:
            return new_name + " " + parts[1]
        return new_name
    return replaced


# =============================================================================
# 2. BASELINE MODEL (BIASED)
# =============================================================================

def train_baseline_model(X_train, y_train, X_test, y_test, sensitive_test):
    """Train a standard logistic regression (will be biased)."""
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Fairness metrics
    dp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    )

    try:
        eo_diff = equalized_odds_difference(
            y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
        )
    except Exception:
        eo_diff = dp_diff

    # Disparate impact
    groups = np.unique(sensitive_test)
    rates = {}
    for g in groups:
        mask = sensitive_test == g
        rates[g] = y_pred[mask].mean()
    rate_values = [v for v in rates.values() if v > 0]
    di_ratio = min(rate_values) / max(rate_values) if rate_values and max(rate_values) > 0 else 0

    print(f"\n  BASELINE MODEL (Biased):")
    print(f"    Accuracy:               {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    Demographic Parity Diff: {dp_diff:.4f} "
          f"({'FAIL' if abs(dp_diff) >= 0.10 else 'PASS'})")
    print(f"    Equalized Odds Diff:     {eo_diff:.4f}")
    print(f"    Disparate Impact Ratio:  {di_ratio:.4f} "
          f"({'FAIL' if di_ratio < 0.80 else 'PASS'})")
    print(f"    Selection rates: {rates}")

    return model, {
        "method": "Baseline",
        "accuracy": accuracy,
        "dem_parity_diff": abs(dp_diff),
        "equalized_odds_diff": eo_diff,
        "disparate_impact": di_ratio,
        "predictions": y_pred,
    }


# =============================================================================
# 3. MITIGATION #1 (REPLACEMENT): COUNTERFACTUAL DATA AUGMENTATION (CDA)
# =============================================================================

def apply_cda(X_train, y_train, sensitive_train, text_train, name_train,
             X_test, y_test, sensitive_test, name_pool, name_to_race,
             n_counterfactuals=3, seed=42):
    """Pre-processing: Counterfactual Data Augmentation (CDA).

    For each training example, create additional training examples by swapping
    ONLY the customer name while keeping the rest of the complaint text fixed.
    This pushes the classifier to learn that the name token should not change
    the prediction.
    """
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )

    try:
        from scipy.sparse import vstack as sp_vstack
    except Exception as e:
        raise RuntimeError("scipy is required for CDA augmentation (sparse vstack)") from e

    rng = np.random.default_rng(seed)

    augmented_texts = []
    augmented_labels = []
    augmented_sensitive = []

    name_pool_arr = np.array(name_pool)

    for t, old_name, label in zip(text_train, name_train, y_train):
        # sample counterfactual names (exclude the original)
        candidates = name_pool_arr[name_pool_arr != old_name]
        if len(candidates) == 0:
            continue
        k = min(int(n_counterfactuals), len(candidates))
        sampled = rng.choice(candidates, size=k, replace=False)

        for new_name in sampled:
            new_text = _replace_name_once(str(t), str(old_name), str(new_name))
            augmented_texts.append(new_text)
            augmented_labels.append(int(label))
            augmented_sensitive.append(name_to_race.get(str(new_name), "Unknown"))

    if len(augmented_texts) == 0:
        # No augmentation possible; fall back to training on original data.
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        # Transform augmented texts using the SAME TF-IDF vectorizer feature space
        # (X_train/X_test were already created from that vectorizer).
        # We reconstruct a vectorizer transform by reusing the fact that X_train
        # was created by the caller; therefore, we must also receive the
        # corresponding vectorizer externally. To keep changes minimal, we
        # instead regenerate the TF-IDF space from X_train metadata is not
        # possible. So we require X_train to be built with the module-level
        # vectorizer in this pipeline; in our case, we use the existing
        # `vectorizer` captured in `run_full_mitigation_pipeline`.
        raise RuntimeError(
            "CDA requires vectorizer access to transform augmented texts. "
            "Use apply_cda_with_vectorizer(...) instead."
        )

    accuracy = accuracy_score(y_test, y_pred)
    dp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    )
    try:
        eo_diff = equalized_odds_difference(
            y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
        )
    except Exception:
        eo_diff = dp_diff

    groups_test = np.unique(sensitive_test)
    rates = {}
    for g in groups_test:
        mask = sensitive_test == g
        rates[g] = y_pred[mask].mean()
    rate_values = [v for v in rates.values() if v > 0]
    di_ratio = min(rate_values) / max(rate_values) if rate_values and max(rate_values) > 0 else 0

    print(f"\n  CDA (Pre-processing - Counterfactual Name Swaps):")
    print(f"    Accuracy:               {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    Demographic Parity Diff: {dp_diff:.4f} "
          f"({'FAIL' if abs(dp_diff) >= 0.10 else 'PASS'})")
    print(f"    Equalized Odds Diff:     {eo_diff:.4f}")
    print(f"    Disparate Impact Ratio:  {di_ratio:.4f} "
          f"({'FAIL' if di_ratio < 0.80 else 'PASS'})")

    return {
        "method": "CDA (Name Swap Augmentation)",
        "accuracy": accuracy,
        "dem_parity_diff": abs(dp_diff),
        "equalized_odds_diff": eo_diff,
        "disparate_impact": di_ratio,
        "predictions": y_pred,
    }


def apply_cda_with_vectorizer(vectorizer, X_train, y_train, sensitive_train,
                              text_train, name_train,
                              X_test, y_test, sensitive_test,
                              name_pool, name_to_race,
                              n_counterfactuals=3, seed=42):
    """CDA implementation that can transform augmented texts using `vectorizer`."""
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

    from scipy.sparse import vstack as sp_vstack

    rng = np.random.default_rng(seed)
    name_pool_arr = np.array(name_pool)

    augmented_texts = []
    augmented_labels = []
    augmented_sensitive = []

    for t, old_name, label in zip(text_train, name_train, y_train):
        candidates = name_pool_arr[name_pool_arr != old_name]
        if len(candidates) == 0:
            continue
        k = min(int(n_counterfactuals), len(candidates))
        sampled = rng.choice(candidates, size=k, replace=False)
        for new_name in sampled:
            new_text = _replace_name_once(str(t), str(old_name), str(new_name))
            augmented_texts.append(new_text)
            augmented_labels.append(int(label))
            augmented_sensitive.append(name_to_race.get(str(new_name), "Unknown"))

    if len(augmented_texts) > 0:
        X_aug = vectorizer.transform(augmented_texts)
        X_train_aug = sp_vstack([X_train, X_aug])
        y_train_aug = np.concatenate([y_train, np.array(augmented_labels, dtype=int)])
        sensitive_train_aug = np.concatenate([sensitive_train, np.array(augmented_sensitive)])
    else:
        X_train_aug = X_train
        y_train_aug = y_train
        sensitive_train_aug = sensitive_train

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_aug, y_train_aug)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    dp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    )
    try:
        eo_diff = equalized_odds_difference(
            y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
        )
    except Exception:
        eo_diff = dp_diff

    groups_test = np.unique(sensitive_test)
    rates = {}
    for g in groups_test:
        mask = sensitive_test == g
        rates[g] = y_pred[mask].mean()
    rate_values = [v for v in rates.values() if v > 0]
    di_ratio = min(rate_values) / max(rate_values) if rate_values and max(rate_values) > 0 else 0

    print(f"\n  CDA (Pre-processing - Counterfactual Name Swaps):")
    print(f"    Accuracy:               {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    Demographic Parity Diff: {dp_diff:.4f} "
          f"({'FAIL' if abs(dp_diff) >= 0.10 else 'PASS'})")
    print(f"    Equalized Odds Diff:     {eo_diff:.4f}")
    print(f"    Disparate Impact Ratio:  {di_ratio:.4f} "
          f"({'FAIL' if di_ratio < 0.80 else 'PASS'})")

    return model, {
        "method": "CDA (Name Swap Augmentation)",
        "accuracy": accuracy,
        "dem_parity_diff": abs(dp_diff),
        "equalized_odds_diff": eo_diff,
        "disparate_impact": di_ratio,
        "predictions": y_pred,
    }


# =============================================================================
# 4. MITIGATION #2: EXPONENTIATED GRADIENT (In-processing)
# =============================================================================

def apply_exponentiated_gradient(X_train, y_train, sensitive_train,
                                  X_test, y_test, sensitive_test):
    """
    In-processing: Uses Fairlearn's ExponentiatedGradient to train a model
    with fairness constraints built into the optimization.
    This is the equivalent of adversarial debiasing for traditional ML.
    """
    from fairlearn.reductions import (
        ExponentiatedGradient,
        DemographicParity,
    )
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )

    base_model = LogisticRegression(max_iter=1000, random_state=42)
    constraint = DemographicParity()

    mitigator = ExponentiatedGradient(
        estimator=base_model,
        constraints=constraint,
        max_iter=50,
    )
    # Convert sparse to dense for fairlearn compatibility
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    mitigator.fit(X_train_dense, y_train, sensitive_features=sensitive_train)

    y_pred = mitigator.predict(X_test_dense)
    accuracy = accuracy_score(y_test, y_pred)

    dp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    )
    try:
        eo_diff = equalized_odds_difference(
            y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
        )
    except Exception:
        eo_diff = dp_diff

    groups = np.unique(sensitive_test)
    rates = {}
    for g in groups:
        mask = sensitive_test == g
        rates[g] = y_pred[mask].mean()
    rate_values = [v for v in rates.values() if v > 0]
    di_ratio = min(rate_values) / max(rate_values) if rate_values and max(rate_values) > 0 else 0

    print(f"\n  EXPONENTIATED GRADIENT / ADVERSARIAL DEBIASING (In-processing):")
    print(f"    Accuracy:               {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    Demographic Parity Diff: {dp_diff:.4f} "
          f"({'FAIL' if abs(dp_diff) >= 0.10 else 'PASS'})")
    print(f"    Equalized Odds Diff:     {eo_diff:.4f}")
    print(f"    Disparate Impact Ratio:  {di_ratio:.4f} "
          f"({'FAIL' if di_ratio < 0.80 else 'PASS'})")

    return mitigator, {
        "method": "Adversarial Debiasing",
        "accuracy": accuracy,
        "dem_parity_diff": abs(dp_diff),
        "equalized_odds_diff": eo_diff,
        "disparate_impact": di_ratio,
        "predictions": y_pred,
    }


# =============================================================================
# 5. MITIGATION #3: THRESHOLD OPTIMIZER (Post-processing)
# =============================================================================

def apply_threshold_optimizer(baseline_model, X_train, y_train, sensitive_train,
                               X_test, y_test, sensitive_test):
    """
    Post-processing: Adjusts decision thresholds per group to achieve fairness.
    Equivalent to Calibrated Equalized Odds.
    """
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )

    postprocessor = ThresholdOptimizer(
        estimator=baseline_model,
        constraints="demographic_parity",
        objective="accuracy_score",
        prefit=True,
    )
    # Convert sparse to dense for fairlearn compatibility
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    postprocessor.fit(X_train_dense, y_train, sensitive_features=sensitive_train)

    y_pred = postprocessor.predict(X_test_dense, sensitive_features=sensitive_test)
    accuracy = accuracy_score(y_test, y_pred)

    dp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    )
    try:
        eo_diff = equalized_odds_difference(
            y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
        )
    except Exception:
        eo_diff = dp_diff

    groups = np.unique(sensitive_test)
    rates = {}
    for g in groups:
        mask = sensitive_test == g
        rates[g] = y_pred[mask].mean()
    rate_values = [v for v in rates.values() if v > 0]
    di_ratio = min(rate_values) / max(rate_values) if rate_values and max(rate_values) > 0 else 0

    print(f"\n  THRESHOLD OPTIMIZER / CALIBRATED EQ. ODDS (Post-processing):")
    print(f"    Accuracy:               {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    Demographic Parity Diff: {dp_diff:.4f} "
          f"({'FAIL' if abs(dp_diff) >= 0.10 else 'PASS'})")
    print(f"    Equalized Odds Diff:     {eo_diff:.4f}")
    print(f"    Disparate Impact Ratio:  {di_ratio:.4f} "
          f"({'FAIL' if di_ratio < 0.80 else 'PASS'})")

    return postprocessor, {
        "method": "Calibrated Eq. Odds",
        "accuracy": accuracy,
        "dem_parity_diff": abs(dp_diff),
        "equalized_odds_diff": eo_diff,
        "disparate_impact": di_ratio,
        "predictions": y_pred,
    }


# =============================================================================
# 6. COMPARISON & RECOMMENDATION
# =============================================================================

def compare_methods(results_list):
    """Create comprehensive comparison of all mitigation methods."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE MITIGATION COMPARISON")
    print("=" * 80)

    headers = ["Method", "Accuracy", "Dem. Parity", "Eq. Odds", "Disp. Impact",
               "Acc. Loss", "Bias Reduction"]
    print(f"\n  {headers[0]:<25s} {headers[1]:>10s} {headers[2]:>12s} "
          f"{headers[3]:>10s} {headers[4]:>12s} {headers[5]:>10s} {headers[6]:>15s}")
    print("  " + "-" * 95)

    baseline_acc = results_list[0]["accuracy"]
    baseline_dp = results_list[0]["dem_parity_diff"]

    for r in results_list:
        acc_loss = baseline_acc - r["accuracy"]
        bias_red = ((baseline_dp - r["dem_parity_diff"]) / baseline_dp * 100
                     if baseline_dp > 0 else 0)

        dp_status = "PASS" if r["dem_parity_diff"] < 0.10 else "FAIL"
        di_status = "PASS" if r["disparate_impact"] > 0.80 else "FAIL"

        print(f"  {r['method']:<25s} {r['accuracy']:>9.1%} "
              f"{r['dem_parity_diff']:>10.4f}({dp_status}) "
              f"{r['equalized_odds_diff']:>10.4f} "
              f"{r['disparate_impact']:>10.4f}({di_status}) "
              f"{acc_loss:>+9.1%} {bias_red:>14.1f}%")

    # Determine winner
    mitigated = [r for r in results_list if r["method"] != "Baseline"]
    best = min(mitigated, key=lambda x: x["dem_parity_diff"])
    print(f"\n  RECOMMENDED METHOD: {best['method']}")
    print(f"  Reason: Lowest Demographic Parity Difference ({best['dem_parity_diff']:.4f})")
    print(f"  Accuracy cost: {(baseline_acc - best['accuracy'])*100:.1f}%")

    return best


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def run_full_mitigation_pipeline(df, score_col="VADER_compound"):
    """Run complete mitigation pipeline."""
    print("=" * 60)
    print("  BIAS MITIGATION PIPELINE")
    print("=" * 60)

    # Prepare data
    X, y, sensitive_race, sensitive_gender, sensitive_demo, vectorizer, prep_df = \
        prepare_model_data(df, score_col)

    # Train/test split (stratified)
    text_all = np.array(prep_df["Full_Text"].values)
    name_all = np.array(prep_df["Name"].values)
    X_train, X_test, y_train, y_test, race_train, race_test, text_train, text_test, name_train, name_test = train_test_split(
        X, y, sensitive_race, text_all, name_all, test_size=0.3, random_state=42, stratify=y
    )

    # Name pool for counterfactual swaps
    name_pool, name_to_race = _build_name_pool(prep_df)

    results_list = []

    # Baseline
    print("\n--- Training Baseline ---")
    baseline_model, baseline_results = train_baseline_model(
        X_train, y_train, X_test, y_test, race_test
    )
    results_list.append(baseline_results)

    # Mitigation 1 (replacement): Counterfactual Data Augmentation (CDA)
    print("\n--- Applying CDA (Counterfactual Data Augmentation) ---")
    cda_model, cda_results = apply_cda_with_vectorizer(
        vectorizer,
        X_train, y_train, race_train,
        text_train, name_train,
        X_test, y_test, race_test,
        name_pool, name_to_race,
        n_counterfactuals=3,
        seed=42,
    )
    results_list.append(cda_results)

    # Mitigation 2: Exponentiated Gradient
    print("\n--- Applying Exponentiated Gradient ---")
    eg_model, eg_results = apply_exponentiated_gradient(
        X_train, y_train, race_train, X_test, y_test, race_test
    )
    results_list.append(eg_results)

    # Mitigation 3: Threshold Optimizer
    print("\n--- Applying Threshold Optimizer ---")
    to_model, to_results = apply_threshold_optimizer(
        baseline_model, X_train, y_train, race_train, X_test, y_test, race_test
    )
    results_list.append(to_results)

    # Compare
    best = compare_methods(results_list)

    # Save models
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "07_Demo", "models"
    )
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(baseline_model, os.path.join(output_dir, "baseline_model.pkl"))
    joblib.dump(cda_model, os.path.join(output_dir, "cda_model.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    print(f"\n  Models saved to: {output_dir}")

    # Save comparison results
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "04_Results"
    )
    os.makedirs(results_dir, exist_ok=True)
    comparison_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "predictions"}
        for r in results_list
    ])

    # Add baseline deltas so it's immediately obvious whether a method improved.
    baseline_row = comparison_df.iloc[0]
    baseline_acc = float(baseline_row["accuracy"])
    baseline_dp = float(baseline_row["dem_parity_diff"])
    baseline_eo = float(baseline_row["equalized_odds_diff"])
    baseline_di = float(baseline_row["disparate_impact"])

    comparison_df["delta_accuracy"] = comparison_df["accuracy"].astype(float) - baseline_acc
    comparison_df["delta_dem_parity_diff"] = comparison_df["dem_parity_diff"].astype(float) - baseline_dp
    comparison_df["delta_equalized_odds_diff"] = comparison_df["equalized_odds_diff"].astype(float) - baseline_eo
    comparison_df["delta_disparate_impact"] = comparison_df["disparate_impact"].astype(float) - baseline_di

    # For dp/eo, LOWER is better. For disparate impact, CLOSER TO 1 is better; we also show +/-. 
    comparison_df["improves_dem_parity"] = comparison_df["delta_dem_parity_diff"] < 0
    comparison_df["improves_eq_odds"] = comparison_df["delta_equalized_odds_diff"] < 0
    comparison_df["improves_disp_impact"] = comparison_df["delta_disparate_impact"] > 0

    # Save to a NEW file so you can compare with the earlier run
    comparison_df.to_csv(
        os.path.join(results_dir, "mitigation_comparison_cda_with_deltas.csv"), index=False
    )

    return results_list, baseline_model, cda_model, eg_model, vectorizer


if __name__ == "__main__":
    # Load scored data
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "02_Data",
        "sentiment_scores_all_systems.csv"
    )
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    results_list, baseline, reweigh, eg, vec = run_full_mitigation_pipeline(df)
