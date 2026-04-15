"""
==============================================================================
Step 6: Explainability Analysis (SHAP + LIME)
==============================================================================
Uses SHAP and LIME to explain:
  - Which words drive sentiment predictions
  - How the same word is weighted differently for different names
  - Why bias exists at the feature level
  - Customer-facing explanations
==============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PREPARE MODEL FOR EXPLAINABILITY
# =============================================================================

def prepare_explainability_model(df, score_col="VADER_compound"):
    """Train a model specifically for explainability analysis."""
    df = df.copy()
    median_score = df[score_col].median()
    df["label"] = (df[score_col] >= median_score).astype(int)

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    X = vectorizer.fit_transform(df["Full_Text"])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, np.array(df["label"].values), np.array(df.index.values),
        test_size=0.3, random_state=42, stratify=df["label"]
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    test_df = df.loc[idx_test].copy()
    test_df["predicted"] = model.predict(X_test)

    return model, vectorizer, X_train, X_test, y_train, y_test, test_df


# =============================================================================
# 2. SHAP ANALYSIS
# =============================================================================

def run_shap_analysis(model, vectorizer, X_train, X_test, test_df, output_dir):
    """
    SHAP (SHapley Additive exPlanations):
    - Shows which features (words) influence predictions
    - Based on game theory
    - Global + local explanations
    """
    import shap

    print("\n" + "=" * 60)
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Create SHAP explainer
    explainer = shap.LinearExplainer(model, X_train, feature_names=vectorizer.get_feature_names_out())

    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)

    feature_names = vectorizer.get_feature_names_out()

    # --- Global Feature Importance ---
    print("\n  TOP 20 MOST IMPORTANT FEATURES (GLOBAL):")
    print("  " + "-" * 50)
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_shap": mean_shap
    }).sort_values("mean_shap", ascending=False)

    for _, row in feature_importance.head(20).iterrows():
        bar = "█" * int(row["mean_shap"] * 100)
        print(f"    {row['feature']:15s}: {row['mean_shap']:.4f} {bar}")

    # Save feature importance
    feature_importance.to_csv(
        os.path.join(output_dir, "shap_feature_importance.csv"), index=False
    )

    # --- SHAP Summary Plot ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Summary Plot: Feature Importance for Sentiment Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("\n  Saved: shap_summary_plot.png")

    # --- Compare SHAP values across demographics ---
    print("\n  SHAP VALUES BY DEMOGRAPHIC GROUP:")
    print("  " + "-" * 50)

    # Calculate mean SHAP for each demographic
    demo_shap = {}
    for demo in test_df["Demographic_Group"].unique():
        mask = test_df["Demographic_Group"].values == demo
        demo_indices = np.where(mask[:len(shap_values)])[0]
        if len(demo_indices) > 0:
            demo_shap[demo] = shap_values[demo_indices].mean(axis=0)

    # Find words that have different SHAP values by demographic
    print("\n  WORDS WITH LARGEST DEMOGRAPHIC VARIATION:")
    emotion_words = ["angry", "furious", "frustrated", "disappointed",
                     "upset", "annoyed", "mad", "outraged", "irritated",
                     "unhappy", "demands", "insists", "expects", "threatens"]

    for word in emotion_words:
        if word in feature_names:
            idx = list(feature_names).index(word)
            values_by_demo = {}
            for demo, vals in demo_shap.items():
                values_by_demo[demo] = vals[idx]

            if values_by_demo:
                max_val = max(values_by_demo.values())
                min_val = min(values_by_demo.values())
                variation = max_val - min_val

                if abs(variation) > 0.001:
                    print(f"\n    Word: '{word}' (variation: {variation:.4f})")
                    for demo in sorted(values_by_demo.keys()):
                        v = values_by_demo[demo]
                        print(f"      {demo:20s}: {v:+.4f}")

    # --- Individual Explanations (Force Plots) ---
    print("\n  INDIVIDUAL EXPLANATION EXAMPLES:")
    print("  " + "-" * 50)

    # Pick examples from different demographics with same template
    template_1_indices = test_df[test_df["Template_Number"] == 1].index
    if len(template_1_indices) >= 4:
        for demo in ["White_Male", "Black_Male", "Indian_Male", "Chinese_Male"]:
            demo_idx = test_df[
                (test_df["Template_Number"] == 1) &
                (test_df["Demographic_Group"] == demo)
            ]
            if len(demo_idx) > 0:
                row = demo_idx.iloc[0]
                local_idx = list(test_df.index).index(row.name)
                if local_idx < len(shap_values):
                    sv = shap_values[local_idx]
                    top_features = np.argsort(np.abs(sv))[-5:][::-1]
                    print(f"\n    {row['Full_Text']}")
                    print(f"    Prediction: {'Positive' if row.get('predicted', 0) else 'Negative'}")
                    print(f"    Top contributing words:")
                    for fi in top_features:
                        if abs(sv[fi]) > 0.001:
                            direction = "NEGATIVE" if sv[fi] < 0 else "POSITIVE"
                            print(f"      '{feature_names[fi]}': {sv[fi]:+.4f} → {direction}")

    # --- SHAP Bar Plot ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar_plot.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("\n  Saved: shap_bar_plot.png")

    return shap_values, feature_importance


# =============================================================================
# 3. LIME ANALYSIS
# =============================================================================

def run_lime_analysis(model, vectorizer, test_df, X_test, output_dir):
    """
    LIME (Local Interpretable Model-agnostic Explanations):
    - Explains individual predictions in human-readable terms
    - Creates "What if?" counterfactual scenarios
    - Designed for customer-facing explanations
    """
    from lime.lime_text import LimeTextExplainer

    print("\n" + "=" * 60)
    print("  LIME EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Create LIME explainer for text
    class_names = ["Negative", "Positive"]
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    # Create prediction function for LIME
    def predict_fn(texts):
        transformed = vectorizer.transform(texts)
        return model.predict_proba(transformed)

    # --- Generate LIME explanations for key examples ---
    print("\n  LIME EXPLANATIONS FOR KEY EXAMPLES:")
    print("  " + "-" * 50)

    # Select comparison pairs (same template, different demographics)
    comparison_pairs = [
        ("White_Male", "Black_Male", "angry"),
        ("White_Female", "Black_Female", "frustrated"),
        ("White_Male", "Indian_Male", "disappointed"),
        ("White_Male", "Chinese_Male", "demanding"),
    ]

    lime_results = []

    for demo_a, demo_b, category in comparison_pairs:
        # Get examples from the same template category
        cat_df = test_df[test_df["Template_Category"] == category]

        example_a = cat_df[cat_df["Demographic_Group"] == demo_a]
        example_b = cat_df[cat_df["Demographic_Group"] == demo_b]

        if len(example_a) > 0 and len(example_b) > 0:
            text_a = example_a.iloc[0]["Full_Text"]
            text_b = example_b.iloc[0]["Full_Text"]

            print(f"\n    === Comparison: {demo_a} vs {demo_b} ({category}) ===")

            # Explain text A
            exp_a = explainer.explain_instance(
                text_a, predict_fn, num_features=10, num_samples=500
            )
            print(f"\n    Text: \"{text_a}\"")
            proba_a = predict_fn([text_a])[0]
            print(f"    Prediction: Negative={proba_a[0]:.3f}, Positive={proba_a[1]:.3f}")
            print(f"    Top features (LIME):")
            for word, weight in exp_a.as_list()[:6]:
                direction = "→ Negative" if weight < 0 else "→ Positive"
                print(f"      '{word}': {weight:+.4f} {direction}")

            # Explain text B
            exp_b = explainer.explain_instance(
                text_b, predict_fn, num_features=10, num_samples=500
            )
            print(f"\n    Text: \"{text_b}\"")
            proba_b = predict_fn([text_b])[0]
            print(f"    Prediction: Negative={proba_b[0]:.3f}, Positive={proba_b[1]:.3f}")
            print(f"    Top features (LIME):")
            for word, weight in exp_b.as_list()[:6]:
                direction = "→ Negative" if weight < 0 else "→ Positive"
                print(f"      '{word}': {weight:+.4f} {direction}")

            # Difference analysis
            diff = proba_a[0] - proba_b[0]
            print(f"\n    ⚠ BIAS: {demo_b} scored {abs(diff)*100:.1f}% "
                  f"{'more' if diff < 0 else 'less'} negative for same complaint!")

            lime_results.append({
                "demo_a": demo_a, "demo_b": demo_b,
                "category": category,
                "text_a": text_a, "text_b": text_b,
                "neg_prob_a": proba_a[0], "neg_prob_b": proba_b[0],
                "difference": diff,
            })

            # Save LIME HTML explanation
            try:
                html_a = exp_a.as_html()
                with open(os.path.join(output_dir,
                          f"lime_{demo_a}_{category}.html"), "w",
                          encoding="utf-8") as f:
                    f.write(html_a)
                html_b = exp_b.as_html()
                with open(os.path.join(output_dir,
                          f"lime_{demo_b}_{category}.html"), "w",
                          encoding="utf-8") as f:
                    f.write(html_b)
            except Exception:
                pass

    # --- Counterfactual Analysis ---
    print("\n\n  COUNTERFACTUAL ANALYSIS ('What If?'):")
    print("  " + "-" * 50)

    base_complaints = [
        "is angry about the delayed delivery",
        "is frustrated with the long wait time",
        "demands immediate refund for the defective product",
    ]
    names_to_test = {
        "Brad": "White_Male",
        "Jamal": "Black_Male",
        "Lakisha": "Black_Female",
        "Emily": "White_Female",
        "Amit": "Indian_Male",
        "Wei": "Chinese_Male",
    }

    for complaint in base_complaints:
        print(f"\n    Complaint: '{{name}} {complaint}'")
        scores = {}
        for name, demo in names_to_test.items():
            full_text = f"{name} {complaint}"
            proba = predict_fn([full_text])[0]
            scores[name] = proba[0]  # Negative probability
            print(f"      {name:10s} ({demo:15s}): "
                  f"Negative={proba[0]:.3f}, Positive={proba[1]:.3f}")

        max_name = max(scores, key=scores.get)
        min_name = min(scores, key=scores.get)
        diff = scores[max_name] - scores[min_name]
        print(f"      ⚠ Max difference: {diff*100:.1f}% "
              f"({max_name} vs {min_name})")

    # Save LIME results
    if lime_results:
        pd.DataFrame(lime_results).to_csv(
            os.path.join(output_dir, "lime_comparison_results.csv"), index=False
        )

    return lime_results


# =============================================================================
# 4. SHAP vs LIME COMPARISON
# =============================================================================

def compare_shap_lime(shap_importance, lime_results, output_dir):
    """Compare insights from SHAP and LIME."""
    print("\n" + "=" * 60)
    print("  SHAP vs LIME COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame([
        {"Aspect": "Scope", "SHAP": "Global + Local", "LIME": "Local only"},
        {"Aspect": "Rigor", "SHAP": "Mathematically exact", "LIME": "Approximate"},
        {"Aspect": "Audience", "SHAP": "Technical/Research", "LIME": "End-users"},
        {"Aspect": "Speed", "SHAP": "Faster (linear)", "LIME": "Slower (sampling)"},
        {"Aspect": "Interpretability", "SHAP": "Feature-level", "LIME": "Word-level"},
        {"Aspect": "Best For", "SHAP": "Proving bias exists", "LIME": "Explaining to users"},
    ])
    print(f"\n{comparison.to_string(index=False)}")

    comparison.to_csv(
        os.path.join(output_dir, "shap_vs_lime_comparison.csv"), index=False
    )


# =============================================================================
# 5. MAIN
# =============================================================================

if __name__ == "__main__":
    # Load data
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "02_Data",
        "sentiment_scores_all_systems.csv"
    )
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "04_Results"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Prepare model
    model, vectorizer, X_train, X_test, y_train, y_test, test_df = \
        prepare_explainability_model(df)

    # SHAP
    shap_values, shap_importance = run_shap_analysis(
        model, vectorizer, X_train, X_test, test_df, output_dir
    )

    # LIME
    lime_results = run_lime_analysis(
        model, vectorizer, test_df, X_test, output_dir
    )

    # Comparison
    compare_shap_lime(shap_importance, lime_results, output_dir)

    print("\n\nExplainability analysis complete!")
