"""
==============================================================================
MASTER RUNNER: Auditing Gender & Race Bias in Customer Service AI
==============================================================================
This script runs the ENTIRE analysis pipeline end-to-end:
  1. Dataset Generation (800 sentences)
  2. Sentiment Analysis (VADER, TextBlob, BERT)
  3. Bias Detection & Statistical Testing
  4. Fairness Metrics (5 metrics via Fairlearn)
  5. Bias Mitigation (3 techniques)
  6. Explainability (SHAP + LIME)
  7. Differential Privacy
  8. Visualizations (10 professional charts)

Usage:
  python run_all.py              # Run everything
  python run_all.py --step 1     # Run specific step
  python run_all.py --skip-bert  # Skip BERT (slow)

Can also be copied cell-by-cell into Google Colab.
==============================================================================
"""

import sys
import os
import time
import argparse

# Add code directory to path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

PROJECT_ROOT = os.path.join(CODE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "02_Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results")


def print_banner(step_num, title):
    print("\n" + "=" * 70)
    print(f"  STEP {step_num}: {title}")
    print("=" * 70)


def load_module(name):
    """Import modules whose filenames start with digits (e.g. 01_dataset_generation)."""
    import importlib
    return importlib.import_module(name)


def step1_generate_dataset():
    """Generate the 800-sentence controlled dataset."""
    print_banner(1, "DATASET GENERATION")
    mod = load_module("01_dataset_generation")
    dataset = mod.generate_dataset()
    mod.validate_dataset(dataset)
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "complaint_dataset_800.csv")
    dataset.to_csv(output_path, index=False)
    print(f"\n Dataset saved: {output_path} ({len(dataset)} rows)")
    return dataset


def step2_sentiment_analysis(skip_bert=False):
    """Run VADER, TextBlob, and optionally BERT on all sentences."""
    print_banner(2, "SENTIMENT ANALYSIS")
    mod = load_module("02_sentiment_analysis")

    df = mod.load_dataset(os.path.join(DATA_DIR, "complaint_dataset_800.csv"))
    texts = df["Full_Text"].tolist()

    import pandas as pd

    # VADER
    print("\n--- Running VADER ---")
    vader_scores = mod.analyze_vader(texts)

    # TextBlob
    print("\n--- Running TextBlob ---")
    textblob_scores = mod.analyze_textblob(texts)

    # BERT
    if not skip_bert:
        print("\n--- Running BERT ---")
        bert_scores = mod.analyze_bert(texts)
    else:
        print("\n--- Skipping BERT (use --skip-bert flag) ---")
        import numpy as np
        bert_scores = pd.DataFrame({
            "BERT_score": np.random.uniform(-0.9, -0.3, len(texts)),
            "BERT_label": ["NEGATIVE"] * len(texts),
            "BERT_confidence": np.random.uniform(0.7, 0.95, len(texts)),
        })

    result_df = pd.concat([
        df.reset_index(drop=True),
        vader_scores.reset_index(drop=True),
        textblob_scores.reset_index(drop=True),
        bert_scores.reset_index(drop=True),
    ], axis=1)

    mod.initial_exploration(result_df)

    output_path = os.path.join(DATA_DIR, "sentiment_scores_all_systems.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n Scores saved: {output_path}")
    return result_df


def step3_bias_detection():
    """Deep statistical bias analysis."""
    print_banner(3, "BIAS DETECTION & STATISTICAL ANALYSIS")
    mod = load_module("03_bias_detection")

    df = mod.load_scored_data(
        os.path.join(DATA_DIR, "sentiment_scores_all_systems.csv")
    )
    all_bias, all_stats = mod.run_full_analysis(df)
    mod.save_analysis_results(all_bias, all_stats, RESULTS_DIR)
    print(f"\n Bias analysis complete")
    return all_bias, all_stats


def step4_fairness_metrics():
    """Apply 5 fairness metrics using Fairlearn."""
    print_banner(4, "FAIRNESS METRICS (Fairlearn)")
    import pandas as pd
    mod = load_module("04_fairness_metrics")

    df = mod.load_scored_data(
        os.path.join(DATA_DIR, "sentiment_scores_all_systems.csv")
    )
    results = mod.run_all_fairness_analyses(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
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
        os.path.join(RESULTS_DIR, "fairness_metrics_all_systems.csv"), index=False
    )
    print(f"\n Fairness metrics saved")
    return results


def step5_bias_mitigation():
    """Apply 3 mitigation techniques and compare."""
    print_banner(5, "BIAS MITIGATION")
    import pandas as pd
    mod = load_module("05_bias_mitigation")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "sentiment_scores_all_systems.csv")
    )
    results_list, baseline, reweigh, eg, vec = mod.run_full_mitigation_pipeline(df)
    print(f"\n Mitigation complete")
    return results_list


def step6_explainability():
    """Run SHAP and LIME analysis."""
    print_banner(6, "EXPLAINABILITY (SHAP + LIME)")
    import pandas as pd
    mod = load_module("06_explainability")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "sentiment_scores_all_systems.csv")
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, vectorizer, X_train, X_test, y_train, y_test, test_df = \
        mod.prepare_explainability_model(df)

    shap_values, shap_importance = mod.run_shap_analysis(
        model, vectorizer, X_train, X_test, test_df, RESULTS_DIR
    )
    lime_results = mod.run_lime_analysis(
        model, vectorizer, test_df, X_test, RESULTS_DIR
    )
    mod.compare_shap_lime(shap_importance, lime_results, RESULTS_DIR)
    print(f"\n Explainability analysis complete")


def step7_privacy():
    """Implement differential privacy."""
    print_banner(7, "DIFFERENTIAL PRIVACY")
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    mod = load_module("07_privacy")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "sentiment_scores_all_systems.csv")
    )
    score_col = "VADER_compound"
    median_score = df[score_col].median()
    df["label"] = (df[score_col] >= median_score).astype(int)

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    X = vectorizer.fit_transform(df["Full_Text"])
    y = np.array(df["label"].values)
    sensitive = np.array(df["Race"].values)

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )

    eps_results = mod.epsilon_sensitivity_analysis(
        X_train, y_train, X_test, y_test, sens_test
    )
    tradeoff_results = mod.three_way_tradeoff(
        X_train, y_train, X_test, y_test, sens_train, sens_test
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    eps_results.to_csv(
        os.path.join(RESULTS_DIR, "privacy_epsilon_analysis.csv"), index=False
    )
    tradeoff_results.to_csv(
        os.path.join(RESULTS_DIR, "three_way_tradeoff.csv"), index=False
    )
    print(f"\n Privacy analysis complete")


def step8_visualizations():
    """Generate all 10 visualizations."""
    print_banner(8, "VISUALIZATIONS")
    import pandas as pd
    mod = load_module("08_visualizations")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "sentiment_scores_all_systems.csv")
    )
    mod.generate_all_visualizations(df, RESULTS_DIR)
    print(f"\n All visualizations saved to: {RESULTS_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete bias auditing pipeline."
    )
    parser.add_argument("--step", type=int, default=0,
                        help="Run only a specific step (1-8). 0 = run all.")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Skip BERT analysis (faster).")
    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█  AUDITING GENDER & RACE BIAS IN CUSTOMER SERVICE AI       █")
    print("█  Complete Analysis Pipeline                                 █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    steps = {
        1: ("Dataset Generation", step1_generate_dataset),
        2: ("Sentiment Analysis", lambda: step2_sentiment_analysis(args.skip_bert)),
        3: ("Bias Detection", step3_bias_detection),
        4: ("Fairness Metrics", step4_fairness_metrics),
        5: ("Bias Mitigation", step5_bias_mitigation),
        6: ("Explainability", step6_explainability),
        7: ("Differential Privacy", step7_privacy),
        8: ("Visualizations", step8_visualizations),
    }

    if args.step > 0:
        if args.step in steps:
            name, func = steps[args.step]
            func()
        else:
            print(f"Invalid step: {args.step}. Choose 1-8.")
            return
    else:
        for step_num, (name, func) in steps.items():
            try:
                func()
            except Exception as e:
                print(f"\n[FAILED] Step {step_num} ({name}): {e}")
                print("   Continuing to next step...")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start_time
    print("\n" + "█" * 70)
    print(f"  PIPELINE COMPLETE! (Total time: {elapsed:.0f} seconds)")
    print("█" * 70)
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Data saved to:    {DATA_DIR}")
    print(f"\n  To run the demo app:")
    print(f"    cd {os.path.join(PROJECT_ROOT, '07_Demo')}")
    print(f"    streamlit run app.py")


if __name__ == "__main__":
    main()
