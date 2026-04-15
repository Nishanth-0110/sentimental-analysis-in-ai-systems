"""
==============================================================================
Streamlit Demo App: Sentiment Analysis Bias Detector
==============================================================================
Interactive web app that demonstrates bias in sentiment analysis.
Shows side-by-side comparison of biased vs fair AI systems.
Includes LIME explanations and pre-loaded examples.

Run with:  streamlit run app.py
==============================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="AI Bias Detector",
    page_icon="🔍",
    layout="wide",
)


# =============================================================================
# SENTIMENT ANALYSIS ENGINES
# =============================================================================

@st.cache_resource
def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_textblob():
    from textblob import TextBlob
    return TextBlob


@st.cache_resource
def load_bert_pipeline():
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
        )
    except Exception:
        return None


def get_vader_score(analyzer, text):
    scores = analyzer.polarity_scores(text)
    return scores["compound"]


def get_textblob_score(TextBlobClass, text):
    from textblob import TextBlob
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_bert_score(pipe, text):
    if pipe is None:
        return 0
    result = pipe(text, truncation=True, max_length=512)[0]
    return -result["score"] if result["label"] == "NEGATIVE" else result["score"]


@st.cache_resource
def load_lime_explainer():
    from lime.lime_text import LimeTextExplainer

    return LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])


def bert_predict_proba(pipe, texts):
    """Return [P(negative), P(positive)] for LIME."""
    text_list = [str(text) for text in texts]
    results = pipe(text_list, truncation=True, max_length=512, batch_size=min(16, max(1, len(text_list))))
    probs = []
    for result in results:
        score = float(result["score"])
        if result["label"] == "NEGATIVE":
            probs.append([score, 1.0 - score])
        else:
            probs.append([1.0 - score, score])
    return np.array(probs, dtype=float)


# =============================================================================
# DEMOGRAPHIC PARITY MITIGATION (ThresholdOptimizer on BERT outputs)
# =============================================================================


@st.cache_resource
def load_dp_mitigation_models():
    """Train baseline + Demographic Parity post-processor from the project dataset.

    This mirrors the project mitigation experiment: learn a simple 1-D decision rule
    on `BERT_score`, then apply Fairlearn ThresholdOptimizer with a Demographic
    Parity constraint.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from fairlearn.postprocessing import ThresholdOptimizer

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "02_Data", "sentiment_scores_all_systems.csv")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["VADER_compound", "BERT_score", "Race"]).copy()

    vader_median = float(df["VADER_compound"].median())
    y_true = (df["VADER_compound"].to_numpy() >= vader_median).astype(int)
    X = df[["BERT_score"]].to_numpy().astype(float)
    sensitive = df["Race"].astype(str).to_numpy()

    X_train, _, y_train, _, sens_train, _ = train_test_split(
        X,
        y_true,
        sensitive,
        test_size=0.30,
        random_state=42,
        stratify=y_true,
    )

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)

    post = ThresholdOptimizer(
        estimator=baseline,
        constraints="demographic_parity",
        objective="accuracy_score",
        prefit=True,
    )
    post.fit(X_train, y_train, sensitive_features=sens_train)

    known_groups = set(pd.Series(sensitive).unique().tolist())
    return baseline, post, known_groups


@st.cache_data
def load_mitigation_summary():
    """Load the latest dataset-level mitigation comparison (baseline vs DP mitigation)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(
        repo_root,
        "04_Results",
        "mitigation_comparison_bert_score_constraints_with_deltas.csv",
    )

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if "method" not in df.columns:
        return None

    keep = df[
        df["method"].isin(
            [
                "Baseline (BERT_score -> LR)",
                "ThresholdOptimizer (Demographic Parity)",
            ]
        )
    ].copy()

    if keep.empty:
        return None

    baseline_row = keep.loc[keep["method"] == "Baseline (BERT_score -> LR)"]
    mitigated_row = keep.loc[keep["method"] == "ThresholdOptimizer (Demographic Parity)"]
    if baseline_row.empty or mitigated_row.empty:
        return None

    baseline_row = baseline_row.iloc[0]
    mitigated_row = mitigated_row.iloc[0]

    # Friendly labels
    keep["method"] = keep["method"].replace(
        {
            "Baseline (BERT_score -> LR)": "Baseline (BERT output decision)",
            "ThresholdOptimizer (Demographic Parity)": "After Mitigation: ThresholdOptimizer (DP)",
        }
    )

    cols = [
        c
        for c in [
            "method",
            "accuracy",
            "dem_parity_diff",
            "equalized_odds_diff",
            "disparate_impact",
            "delta_accuracy",
            "delta_dem_parity_diff",
            "delta_equalized_odds_diff",
            "delta_disparate_impact",
        ]
        if c in keep.columns
    ]
    keep = keep[cols]

    metric_specs = [
        ("accuracy", "Accuracy", "higher"),
        ("dem_parity_diff", "Demographic Parity Difference", "lower"),
        ("equalized_odds_diff", "Equalized Odds Difference", "lower"),
        ("disparate_impact", "Disparate Impact", "closer_to_one"),
    ]

    diff_rows = []
    for column, label, direction in metric_specs:
        if column not in df.columns:
            continue

        before = float(baseline_row[column])
        after = float(mitigated_row[column])
        change = after - before

        if direction == "higher":
            improved = after > before
        elif direction == "lower":
            improved = after < before
        else:
            improved = abs(1.0 - after) < abs(1.0 - before)

        diff_rows.append(
            {
                "Metric": label,
                "Before Mitigation": round(before, 3),
                "After Mitigation": round(after, 3),
                "Change": round(change, 3),
                "Impact": "Improved" if improved else ("No Change" if np.isclose(after, before) else "Worse"),
            }
        )

    diff_df = pd.DataFrame(diff_rows)
    return keep, diff_df


@st.cache_data
def load_project_dataset():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "02_Data", "sentiment_scores_all_systems.csv")
    df = pd.read_csv(data_path)
    return df


@st.cache_data
def get_dataset_example(name, full_text):
    """Return the stored dataset row for an exact preloaded example match."""
    df = load_project_dataset().copy()
    match = df[
        (df["Name"].astype(str) == str(name))
        & (df["Full_Text"].astype(str) == str(full_text))
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "name": str(row["Name"]),
        "sentence_id": int(row["Sentence_ID"]),
        "template_category": str(row["Template_Category"]),
        "emotion_intensity": str(row["Emotion_Intensity"]),
        "full_text": str(row["Full_Text"]),
        "race": str(row["Race"]),
        "gender": str(row["Gender"]),
        "bert_score": float(row["BERT_score"]),
        "bert_label": str(row["BERT_label"]),
        "bert_confidence": float(row["BERT_confidence"]),
    }


def extract_complaint_body(name, full_text):
    """Remove the leading dataset name so we can reapply different names cleanly."""
    prefix = f"{name} "
    if str(full_text).startswith(prefix):
        return str(full_text)[len(prefix):]
    return str(full_text)


@st.cache_data
def load_preloaded_examples():
    """Build the sidebar examples directly from exact dataset rows."""
    df = load_project_dataset().copy()
    sentence_ids = [142, 131, 226, 216, 682, 562, 437, 732]
    selected = df[df["Sentence_ID"].isin(sentence_ids)].copy()
    selected = selected.set_index("Sentence_ID").loc[sentence_ids].reset_index()

    examples = {}
    for row in selected.to_dict("records"):
        label = f"{row['Race']} {row['Gender']} - {row['Template_Category'].title()}"
        examples[label] = {
            "name": str(row["Name"]),
            "full_text": str(row["Full_Text"]),
            "sentence_id": int(row["Sentence_ID"]),
        }
    return examples


@st.cache_data
def load_precomputed_lime_html(race, gender, template_category):
    """Load a precomputed LIME HTML file when an exact demo match exists."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_map = {
        ("Black", "Male", "angry"): "lime_Black_Male_angry.html",
        ("White", "Male", "angry"): "lime_White_Male_angry.html",
        ("Black", "Female", "frustrated"): "lime_Black_Female_frustrated.html",
        ("White", "Female", "frustrated"): "lime_White_Female_frustrated.html",
        ("Indian", "Male", "disappointed"): "lime_Indian_Male_disappointed.html",
        ("Chinese", "Male", "demanding"): "lime_Chinese_Male_demanding.html",
        ("White", "Male", "demanding"): "lime_White_Male_demanding.html",
        ("White", "Male", "disappointed"): "lime_White_Male_disappointed.html",
    }

    filename = file_map.get((str(race), str(gender), str(template_category).lower()))
    if not filename:
        return None, None

    html_path = os.path.join(repo_root, "04_Results", filename)
    if not os.path.exists(html_path):
        return None, None

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(), filename


@st.cache_data
def build_showcase_template():
    """Pick a dataset template where DP mitigation changes outcomes most visibly.

    Uses precomputed BERT scores in the dataset (fast + deterministic).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from fairlearn.postprocessing import ThresholdOptimizer

    df = load_project_dataset().dropna(subset=["BERT_score", "VADER_compound", "Race", "Template_Number"]).copy()
    df["Race"] = df["Race"].astype(str)

    vader_median = float(df["VADER_compound"].median())
    y_true = (df["VADER_compound"].to_numpy() >= vader_median).astype(int)
    X = df[["BERT_score"]].to_numpy().astype(float)
    sensitive = df["Race"].to_numpy()

    X_train, _, y_train, _, sens_train, _ = train_test_split(
        X,
        y_true,
        sensitive,
        test_size=0.30,
        random_state=42,
        stratify=y_true,
    )

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)
    post = ThresholdOptimizer(
        estimator=baseline,
        constraints="demographic_parity",
        objective="accuracy_score",
        prefit=True,
    )
    post.fit(X_train, y_train, sensitive_features=sens_train)

    # Predict on full dataset
    df["baseline_pred"] = baseline.predict(X)
    df["mitigated_pred"] = post.predict(X, sensitive_features=sensitive)

    # For each template, compute race-wise positive rate gap (DP-like) before/after
    # (Positive = label 1)
    rows = []
    for template_num, g in df.groupby("Template_Number"):
        base_rates = g.groupby("Race")["baseline_pred"].mean()
        mit_rates = g.groupby("Race")["mitigated_pred"].mean()
        if len(base_rates) < 2:
            continue
        base_gap = float(base_rates.max() - base_rates.min())
        mit_gap = float(mit_rates.max() - mit_rates.min())
        improvement = base_gap - mit_gap
        rows.append((template_num, base_gap, mit_gap, improvement))

    if not rows:
        return None

    best_template, base_gap, mit_gap, improvement = sorted(rows, key=lambda x: (x[3], x[1]), reverse=True)[0]
    best = df[df["Template_Number"] == best_template].copy()

    # Pick one representative row per race for display
    display = (
        best.sort_values(["Race", "Gender", "Name"])
        .groupby("Race", as_index=False)
        .head(1)
        .copy()
    )

    display["Baseline"] = np.where(display["baseline_pred"] == 0, "Negative", "Not Negative")
    display["After DP Mitigation"] = np.where(display["mitigated_pred"] == 0, "Negative", "Not Negative")

    out = display[["Race", "Name", "Full_Text", "BERT_score", "Baseline", "After DP Mitigation"]].copy()
    out["BERT_score"] = out["BERT_score"].astype(float).round(3)

    summary = {
        "template_number": int(best_template),
        "baseline_gap": round(base_gap, 3),
        "mitigated_gap": round(mit_gap, 3),
        "improvement": round(improvement, 3),
    }
    return summary, out


# =============================================================================
# BIAS SIMULATION (demonstrates the concept)
# =============================================================================

# Name-to-demographic mapping
NAME_DEMOGRAPHICS = {
    # White
    "brad": ("White", "Male"), "connor": ("White", "Male"),
    "jake": ("White", "Male"), "wyatt": ("White", "Male"),
    "garrett": ("White", "Male"), "brad johnson": ("White", "Male"),
    "emily": ("White", "Female"), "molly": ("White", "Female"),
    "katie": ("White", "Female"), "megan": ("White", "Female"),
    "allison": ("White", "Female"), "emily wilson": ("White", "Female"),
    # Black
    "deshawn": ("Black", "Male"), "jamal": ("Black", "Male"),
    "darnell": ("Black", "Male"), "tyrone": ("Black", "Male"),
    "malik": ("Black", "Male"), "jamal williams": ("Black", "Male"),
    "lakisha": ("Black", "Female"), "latoya": ("Black", "Female"),
    "shaniqua": ("Black", "Female"), "tamika": ("Black", "Female"),
    "imani": ("Black", "Female"), "lakisha brown": ("Black", "Female"),
    # Indian
    "amit": ("Indian", "Male"), "raj": ("Indian", "Male"),
    "kumar": ("Indian", "Male"), "aditya": ("Indian", "Male"),
    "vikram": ("Indian", "Male"), "rajesh kumar": ("Indian", "Male"),
    "priya": ("Indian", "Female"), "ananya": ("Indian", "Female"),
    "deepika": ("Indian", "Female"), "kavya": ("Indian", "Female"),
    "neha": ("Indian", "Female"), "priya patel": ("Indian", "Female"),
    # Chinese
    "wei": ("Chinese", "Male"), "ming": ("Chinese", "Male"),
    "chen": ("Chinese", "Male"), "zhang": ("Chinese", "Male"),
    "liu": ("Chinese", "Male"), "wei chen": ("Chinese", "Male"),
    "ying": ("Chinese", "Female"), "mei": ("Chinese", "Female"),
    "xiu": ("Chinese", "Female"), "jing": ("Chinese", "Female"),
    "hui": ("Chinese", "Female"), "mei chen": ("Chinese", "Female"),
}

# Bias factors (simulated based on research findings)
BIAS_FACTORS = {
    ("Black", "Male"): -0.15,
    ("Black", "Female"): -0.12,
    ("Indian", "Male"): -0.07,
    ("Indian", "Female"): -0.04,
    ("Chinese", "Male"): -0.06,
    ("Chinese", "Female"): -0.03,
    ("White", "Male"): 0.00,
    ("White", "Female"): 0.02,
}


def detect_demographic(name):
    """Detect demographic from name."""
    lower = name.lower().strip()
    if lower in NAME_DEMOGRAPHICS:
        return NAME_DEMOGRAPHICS[lower]
    # Check first name
    first = lower.split()[0] if lower else ""
    if first in NAME_DEMOGRAPHICS:
        return NAME_DEMOGRAPHICS[first]
    return ("Unknown", "Unknown")


def simulate_biased_score(base_score, name):
    """Simulate what a biased system would return."""
    race, gender = detect_demographic(name)
    bias = BIAS_FACTORS.get((race, gender), 0)
    return np.clip(base_score + bias, -1, 1), race, gender, bias


def simulate_fair_score(base_score):
    """Fair system returns the same score regardless of name."""
    return base_score


# =============================================================================
# LIME EXPLANATION
# =============================================================================

@st.cache_data(show_spinner=False)
def generate_lime_explanation(text, num_features=6, num_samples=60):
    """Generate a real LIME explanation for the BERT sentiment model."""
    pipe = load_bert_pipeline()
    if pipe is None:
        return None, None

    explainer = load_lime_explainer()
    explanation = explainer.explain_instance(
        str(text),
        classifier_fn=lambda texts: bert_predict_proba(pipe, texts),
        num_features=num_features,
        num_samples=num_samples,
        top_labels=2,
    )

    available_labels = []
    try:
        available_labels = list(explanation.available_labels())
    except Exception:
        available_labels = []

    label_to_use = 0 if 0 in available_labels else (available_labels[0] if available_labels else None)
    if label_to_use is None:
        return explanation, []

    return explanation.as_list(label=label_to_use), explanation.as_html()


def get_counterfactual_audit_names(max_per_group=2):
    """Build a balanced set of names for counterfactual auditing."""
    rows = []
    for raw_name, (race, gender) in NAME_DEMOGRAPHICS.items():
        rows.append(
            {
                "Name": raw_name.title(),
                "Race": race,
                "Gender": gender,
                "WordCount": len(raw_name.split()),
            }
        )

    names_df = pd.DataFrame(rows)
    names_df = names_df.sort_values(
        ["Race", "Gender", "WordCount", "Name"],
        ascending=[True, True, False, True],
    )
    names_df = names_df.groupby(["Race", "Gender"], as_index=False).head(max_per_group)
    names_df["Demographic"] = names_df["Race"] + " " + names_df["Gender"]
    return names_df[["Name", "Race", "Gender", "Demographic"]].reset_index(drop=True)


def compute_counterfactual_table(bert_pipe, baseline_model, dp_post, known_groups, complaint_text):
    """Compute counterfactual (name-swap) BERT scores + baseline/mitigated decisions."""
    counterfactual_names = get_counterfactual_audit_names(max_per_group=2)

    rows = []
    for item in counterfactual_names.to_dict("records"):
        name = item["Name"]
        race = item["Race"]
        gender = item["Gender"]
        demo = item["Demographic"]
        cf_text = f"{name} {complaint_text}"
        cf_bert = float(get_bert_score(bert_pipe, cf_text))
        X_cf = np.array([[cf_bert]], dtype=float)
        base_cf = int(baseline_model.predict(X_cf)[0])

        if race in known_groups:
            mit_cf = int(dp_post.predict(X_cf, sensitive_features=np.array([race]))[0])
        else:
            mit_cf = base_cf

        baseline_label = "Negative" if base_cf == 0 else "Not Negative"
        mitigated_label = "Negative" if mit_cf == 0 else "Not Negative"
        changed = baseline_label != mitigated_label
        baseline_wait = "2-4 hours" if base_cf == 0 else "8+ hours"
        mitigated_wait = "2-4 hours" if mit_cf == 0 else "8+ hours"

        rows.append(
            {
                "Name": name,
                "Race": race,
                "Gender": gender,
                "Demographic": demo,
                "BERT Score": round(float(cf_bert), 3),
                "Baseline": baseline_label,
                "After DP Mitigation": mitigated_label,
                "Baseline Response Time": baseline_wait,
                "After Mitigation Response Time": mitigated_wait,
                "Mitigation Changed?": "Yes" if changed else "No",
                "Decision Shift": f"{baseline_label} -> {mitigated_label}" if changed else "No change",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["Race", "Gender", "Name"]).reset_index(drop=True)

    # Summary: how inconsistent are decisions across the same complaint?
    baseline_negative = int((df["Baseline"] == "Negative").sum())
    mitigated_negative = int((df["After DP Mitigation"] == "Negative").sum())
    baseline_unique = int(df["Baseline"].nunique())
    mitigated_unique = int(df["After DP Mitigation"].nunique())
    changed_count = int((df["Mitigation Changed?"] == "Yes").sum())

    by_demo = (
        df.groupby("Demographic")
        .agg(
            Names_Audited=("Name", "count"),
            Baseline_Negative=("Baseline", lambda s: int((s == "Negative").sum())),
            Mitigated_Negative=("After DP Mitigation", lambda s: int((s == "Negative").sum())),
        )
        .reset_index()
    )
    by_demo["Baseline Negative Rate"] = (by_demo["Baseline_Negative"] / by_demo["Names_Audited"]).round(3)
    by_demo["Mitigated Negative Rate"] = (by_demo["Mitigated_Negative"] / by_demo["Names_Audited"]).round(3)
    by_demo["Rate Change"] = (
        by_demo["Mitigated Negative Rate"] - by_demo["Baseline Negative Rate"]
    ).round(3)
    by_demo = by_demo[
        [
            "Demographic",
            "Names_Audited",
            "Baseline_Negative",
            "Mitigated_Negative",
            "Baseline Negative Rate",
            "Mitigated Negative Rate",
            "Rate Change",
        ]
    ]

    summary = {
        "names_audited": int(len(df)),
        "baseline_negative": baseline_negative,
        "mitigated_negative": mitigated_negative,
        "baseline_unique": baseline_unique,
        "mitigated_unique": mitigated_unique,
        "changed_count": changed_count,
        "baseline_fast_responses": baseline_negative,
        "mitigated_fast_responses": mitigated_negative,
    }

    return df, by_demo, summary


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1>🔍 Sentiment Analysis Bias Detector</h1>
        <p style='font-size: 1.2rem; color: gray;'>
            Test how AI sentiment analysis treats different names differently
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Load analyzers
    vader = load_vader()
    bert_pipe = load_bert_pipeline()

    # Sidebar: Pre-loaded examples
    st.sidebar.header("📋 Pre-loaded Examples")
    st.sidebar.markdown("Click to load an example:")

    examples = load_preloaded_examples()

    selected_example = st.sidebar.radio(
        "Choose example:", list(examples.keys()), index=0
    )

    # Input section
    col1, col2 = st.columns([1, 2])

    with col1:
        default_name = examples[selected_example]["name"]
        default_text = examples[selected_example]["full_text"]

        customer_name = st.text_input(
            "👤 Customer Name",
            value=default_name,
            disabled=True,
        )
        complaint_text = st.text_area(
            "💬 Complaint Text",
            value=default_text,
            height=120,
            disabled=True,
        )

    with col2:
        st.info("""
        **How this works:**
        1. Choose one of the pre-loaded dataset examples
        2. The app uses the stored BERT score from `sentiment_scores_all_systems.csv`
        3. It shows the **baseline decision** vs the **post-processed mitigation**
           (ThresholdOptimizer with Demographic Parity)
        4. The explanation highlights which complaint words push sentiment

        The inputs are locked so the demo always uses exact dataset rows.
        """)

    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        example_row = get_dataset_example(
            examples[selected_example]["name"],
            examples[selected_example]["full_text"],
        )
        if example_row is None:
            st.error("Couldn't find the selected pre-loaded example in sentiment_scores_all_systems.csv.")
            return

        full_text = example_row["full_text"]
        complaint_body = extract_complaint_body(customer_name, full_text)
        bert_score = float(example_row["bert_score"])
        race = example_row["race"]
        gender = example_row["gender"]

        if bert_pipe is None:
            st.error("BERT model couldn't be loaded. Please check your environment and try again.")
            return

        # Load mitigation models (trained from the project dataset)
        baseline_model, dp_post, known_groups = load_dp_mitigation_models()

        X_one = np.array([[bert_score]], dtype=float)
        baseline_pred = int(baseline_model.predict(X_one)[0])
        if race in known_groups:
            mitigated_pred = int(dp_post.predict(X_one, sensitive_features=np.array([race]))[0])
        else:
            mitigated_pred = baseline_pred

        # Detected demographic
        st.markdown("---")
        if race != "Unknown":
            st.markdown(f"**Detected Demographic Signal:** {race} {gender}")
        else:
            st.markdown("**Detected Demographic Signal:** Not recognized "
                       "(using neutral baseline)")

        # Results: Side by side
        st.markdown("## 📊 Results")

        summary_bundle = load_mitigation_summary()
        if summary_bundle is not None:
            summary, diff_df = summary_bundle
            st.markdown("### 📈 Dataset-Level Fairness (Baseline vs DP Mitigation)")
            st.caption(
                "These values come from the saved project audit CSV over the full dataset. "
                "They stay fixed even when you change a single person name in the demo input."
            )
            if not diff_df.empty:
                st.markdown("#### Dataset-Level Before vs After Mitigation")
                metric_cols = st.columns(len(diff_df))
                for col, row in zip(metric_cols, diff_df.to_dict("records")):
                    change_value = float(row["Change"])
                    col.metric(
                        row["Metric"],
                        f"{row['After Mitigation']:.3f}",
                        delta=f"{change_value:+.3f} vs baseline",
                    )
                    col.caption(
                        f"Before: {row['Before Mitigation']:.3f} | {row['Impact']}"
                    )

                with st.expander("Show fixed dataset audit tables"):
                    st.dataframe(diff_df, width="stretch", hide_index=True)
                    st.markdown("##### Raw audited rows from the CSV")
                    st.dataframe(summary, width="stretch", hide_index=True)
            st.markdown("---")

        # Counterfactual summary for THIS complaint (same text, different names)
        with st.spinner("Computing counterfactual name-swap behavior..."):
            cf_df, _cf_group_df, cf_summary = compute_counterfactual_table(
                bert_pipe, baseline_model, dp_post, known_groups, complaint_body
            )
        base_neg = cf_summary["baseline_negative"]
        mit_neg = cf_summary["mitigated_negative"]
        base_unique = cf_summary["baseline_unique"]
        mit_unique = cf_summary["mitigated_unique"]

        st.markdown("### 🔄 Counterfactual Spread (Same Complaint, Different Names)")
        st.caption(
            "This audits the same complaint across a balanced set of names. It shows whether changing only the name changes routing decisions, and whether DP mitigation reduces that variability."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Names Audited", str(cf_summary["names_audited"]))
        c2.metric(
            "Baseline: # Negative",
            str(cf_summary["baseline_negative"]),
            delta=f"{cf_summary['mitigated_negative'] - cf_summary['baseline_negative']:+d} after mitigation",
        )
        c3.metric("Decision Variability", f"{base_unique}→{mit_unique} unique decisions")

        c4.metric("Names With Decision Change", str(cf_summary["changed_count"]))

        if mit_unique < base_unique:
            st.success("Mitigation reduced decision variability across names for this complaint.")
        elif mit_unique == base_unique:
            st.info("For this complaint, mitigation did not reduce variability across the audited names.")
        else:
            st.warning("Mitigation increased variability for this audited set (rare; verify with dataset-level table above).")

        if cf_summary["changed_count"] > 0:
            st.caption("Rows marked `Yes` in `Mitigation Changed?` are the names where post-processing changed the final routing decision.")

        st.markdown("#### Response Time Impact")
        st.caption(
            "In this demo, complaints routed as `Negative` are treated as faster-priority cases."
        )
        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Baseline: Faster Responses",
            str(cf_summary["baseline_fast_responses"]),
        )
        r2.metric(
            "After Mitigation: Faster Responses",
            str(cf_summary["mitigated_fast_responses"]),
            delta=f"{cf_summary['mitigated_fast_responses'] - cf_summary['baseline_fast_responses']:+d}",
        )
        r3.metric(
            "Longer-Wait Cases",
            str(cf_summary["names_audited"] - cf_summary["mitigated_fast_responses"]),
        )

        st.markdown("#### By Name")
        st.dataframe(cf_df, width="stretch", hide_index=True)
        st.markdown("---")

        # If the user example doesn't show a change, show a guaranteed dataset-based example
        if mit_unique == base_unique:
            st.markdown("### ✅ Showcase Example (Guaranteed Visible Mitigation Effect)")
            st.caption(
                "This example is automatically selected from the project dataset to show the largest reduction in race-wise outcome gap "
                "after Demographic Parity mitigation."
            )
            showcase = build_showcase_template()
            if showcase is None:
                st.warning("Couldn't find a showcase template. (This is unexpected.)")
            else:
                showcase_summary, showcase_df = showcase
                s1, s2, s3 = st.columns(3)
                s1.metric("Template #", str(showcase_summary["template_number"]))
                s2.metric("Race-wise outcome gap", f"{showcase_summary['baseline_gap']} → {showcase_summary['mitigated_gap']}")
                s3.metric("Gap reduction", f"-{showcase_summary['improvement']}")
                st.dataframe(showcase_df, width="stretch", hide_index=True)
                st.markdown("---")


        if False:
            st.markdown("""
            <div style='background-color: #FFCDD2; padding: 1.5rem; 
                        border-radius: 10px; border: 2px solid #F44336;'>
                <h3 style='color: #C62828; text-align: center;'>
                    ⚠️ BASELINE AI (Before Mitigation)
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # Severity/urgency derived from baseline negative probability
            severity_biased = (
                "Very Negative 🔴" if baseline_proba_neg >= 0.85 else
                "Negative 🟠" if baseline_proba_neg >= 0.70 else
                "Slightly Negative 🟡" if baseline_proba_neg >= 0.50 else
                "Neutral/Positive 🟢"
            )
            urgency_biased = (
                "CRITICAL" if baseline_proba_neg >= 0.85 else
                "HIGH" if baseline_proba_neg >= 0.70 else
                "MEDIUM" if baseline_proba_neg >= 0.50 else
                "LOW"
            )
            wait_biased = (
                "1-2 hours" if baseline_proba_neg >= 0.85 else
                "2-4 hours" if baseline_proba_neg >= 0.70 else
                "4-8 hours" if baseline_proba_neg >= 0.50 else
                "8+ hours"
            )

            st.metric("BERT Score", f"{bert_score:.3f}")
            st.write(f"**Baseline Decision:** {'Negative' if baseline_pred == 0 else 'Not Negative'}")
            st.write(f"**Severity:** {severity_biased}")
            st.write(f"**Urgency Level:** {urgency_biased}")
            st.write(f"**Est. Response Time:** {wait_biased}")
            st.error("No fairness constraint applied")

        if False:
            st.markdown("""
            <div style='background-color: #C8E6C9; padding: 1.5rem; 
                        border-radius: 10px; border: 2px solid #4CAF50;'>
                <h3 style='color: #2E7D32; text-align: center;'>
                    ✅ MITIGATED AI (After Mitigation)
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # After mitigation: decision may change due to group-conditional thresholding
            severity_fair = "Negative 🟠" if mitigated_pred == 0 else "Neutral/Positive 🟢"
            urgency_fair = "HIGH" if mitigated_pred == 0 else "LOW"
            wait_fair = "2-4 hours" if mitigated_pred == 0 else "8+ hours"

            st.metric("BERT Score", f"{bert_score:.3f}")
            st.write(f"**Mitigated Decision:** {'Negative' if mitigated_pred == 0 else 'Not Negative'}")
            st.write(f"**Severity:** {severity_fair}")
            st.write(f"**Urgency Level:** {urgency_fair}")
            st.write(f"**Est. Response Time:** {wait_fair}")
            st.success("Mitigation applied: ThresholdOptimizer (Demographic Parity) ✓")

        # Mitigation alert
        if race != "Unknown":
            st.markdown("---")
            changed = (baseline_pred != mitigated_pred)
            st.markdown(f"""
            
            """, unsafe_allow_html=True)

        # Explanation
        st.markdown("## 🔬 LIME Explanation")
        st.caption(
            "This demo uses precomputed LIME explanations from the project results so the UI stays responsive. "
            "Mitigation changes the final decision rule, not the word-level explanation."
        )
        lime_html, lime_filename = load_precomputed_lime_html(
            example_row["race"],
            example_row["gender"],
            example_row["template_category"],
        )

        if lime_html is None:
            st.info(
                "A precomputed LIME explanation is not available for this exact demo example. "
                "Use one of the angry/frustrated examples to see the full LIME view."
            )
        else:
            st.success(f"Loaded precomputed explanation: `{lime_filename}`")
            styled_lime_html = f"""
            <html>
                <head>
                    <style>
                        body {{
                            margin: 0;
                            padding: 20px;
                            background: linear-gradient(180deg, #fffaf2 0%, #fffdf8 100%) !important;
                            color: #1f2937 !important;
                            font-family: Georgia, "Times New Roman", serif;
                        }}
                        html, body {{
                            color: #1f2937 !important;
                        }}
                        * {{
                            color: #1f2937 !important;
                        }}
                        svg, div, p, span, text, h1, h2, h3, h4, h5, h6, td, th {{
                            color: #1f2937 !important;
                        }}
                        .lime-wrap {{
                            background: rgba(255, 255, 255, 0.96);
                            border: 1px solid #e5d9c5;
                            border-radius: 18px;
                            box-shadow: 0 12px 30px rgba(120, 98, 62, 0.10);
                            padding: 20px 24px;
                        }}
                        .lime-wrap svg {{
                            background: transparent !important;
                        }}
                        .lime-wrap text {{
                            fill: #1f2937 !important;
                        }}
                        .lime-wrap rect {{
                            stroke: rgba(96, 72, 36, 0.35) !important;
                        }}
                        a {{
                            color: #8b4513 !important;
                        }}
                        table {{
                            border-collapse: collapse !important;
                        }}
                        td, th {{
                            border-color: #eadfcd !important;
                        }}
                    </style>
                </head>
                <body>
                    <div class="lime-wrap">
                        {lime_html}
                    </div>
                </body>
            </html>
            """
            components.html(styled_lime_html, height=900, scrolling=True)
         
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
        <p><b>Auditing Gender & Race Bias in Customer Service AI</b></p>
        <p>Responsible AI Course Project | Demonstrating bias in sentiment analysis systems</p>
        <p>This demo shows baseline BERT behavior and Demographic Parity post-processing mitigation.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
