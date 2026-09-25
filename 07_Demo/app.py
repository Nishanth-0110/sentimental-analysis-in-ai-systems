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
import altair as alt
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="AI Bias Detector",
    page_icon="🔍",
    layout="wide",
)

MITIGATION_RANDOM_STATE = 42


def decision_label(pred):
    """Human-readable label for the routing classifier output.

    pred == 0 means the router flagged the complaint as strongly negative,
    which in this business scenario means it is escalated (urgent, human
    agent). pred == 1 means it is deprioritized to the standard queue.
    """
    return "Escalated (urgent)" if int(pred) == 0 else "Standard queue"


# =============================================================================
# SENTIMENT ANALYSIS ENGINES
# =============================================================================

@st.cache_resource
def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_bert_pipeline():
    """Load a lightweight HuggingFace sentiment pipeline.

    Returns None if transformers/torch aren't available (demo can still run).
    """
    try:
        from transformers import pipeline

        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
        )
    except Exception:
        return None


def bert_predict_proba(pipe, texts):
    """Return 2-class probabilities [P(NEGATIVE), P(POSITIVE)] for each text."""
    texts = list(texts)
    if pipe is None:
        return np.tile(np.array([[0.5, 0.5]], dtype=float), (len(texts), 1))

    outputs = pipe(texts, truncation=True)
    probs = []
    for out in outputs:
        label = str(out.get("label", ""))
        score = float(out.get("score", 0.5))
        score = float(np.clip(score, 0.0, 1.0))
        if label.upper() == "NEGATIVE":
            p_neg = score
            p_pos = 1.0 - score
        else:
            p_pos = score
            p_neg = 1.0 - score
        probs.append([p_neg, p_pos])
    return np.array(probs, dtype=float)


class _BaselinePostProcessor:
    """Fallback post-processor with ThresholdOptimizer-like predict signature."""

    def __init__(self, baseline_model):
        self._baseline_model = baseline_model

    def predict(self, X, sensitive_features=None, random_state=None):
        return self._baseline_model.predict(X)


@st.cache_resource
def load_dp_mitigation_models():
    """Train (and cache) baseline + DP mitigators used by the routing demo.

    Baseline model: LogisticRegression on 1-D feature [BERT_score].
    Target label: y_true = 1 if VADER_compound >= dataset median else 0.
    Sensitive attribute: Race.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    df = load_project_dataset().dropna(subset=["BERT_score", "VADER_compound", "Race"]).copy()
    df["Race"] = df["Race"].astype(str)

    vader_median = float(df["VADER_compound"].median())
    y = (df["VADER_compound"].to_numpy(dtype=float) >= vader_median).astype(int)
    X = df[["BERT_score"]].to_numpy().astype(float)
    sensitive = df["Race"].to_numpy()

    X_train, _X_test, y_train, _y_test, sens_train, _sens_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=0.30,
        random_state=MITIGATION_RANDOM_STATE,
        stratify=y,
    )

    baseline = LogisticRegression(max_iter=1000, random_state=MITIGATION_RANDOM_STATE)
    baseline.fit(X_train, y_train)

    post = None
    eg = None
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        from fairlearn.reductions import DemographicParity, ExponentiatedGradient

        post = ThresholdOptimizer(
            estimator=baseline,
            constraints="demographic_parity",
            objective="accuracy_score",
            prefit=True,
        )
        post.fit(X_train, y_train, sensitive_features=sens_train)

        eg = ExponentiatedGradient(
            estimator=LogisticRegression(max_iter=1000, random_state=MITIGATION_RANDOM_STATE),
            constraints=DemographicParity(),
            max_iter=50,
        )
        eg.fit(X_train, y_train, sensitive_features=sens_train)
    except Exception:
        post = _BaselinePostProcessor(baseline)
        eg = None

    known_groups = set(pd.Series(sensitive).unique().tolist())
    return baseline, post, eg, known_groups


def _predict_mitigated_label(
    mitigation_method: str,
    *,
    X_one: np.ndarray,
    race: str,
    baseline_model,
    dp_post,
    dp_eg,
    known_groups: set,
):
    """Return (pred_int, method_display_name) for the chosen mitigation method."""
    if mitigation_method == "In-processing: Exponentiated Gradient (DP)":
        if dp_eg is None:
            return int(baseline_model.predict(X_one)[0]), "Exponentiated Gradient (unavailable; fallback baseline)"
        return int(dp_eg.predict(X_one, random_state=MITIGATION_RANDOM_STATE)[0]), "Exponentiated Gradient (DP)"

    # Default: post-processing ThresholdOptimizer
    if race in known_groups:
        return (
            int(
                dp_post.predict(
                    X_one,
                    sensitive_features=np.array([race]),
                    random_state=MITIGATION_RANDOM_STATE,
                )[0]
            ),
            "ThresholdOptimizer (DP)",
        )
    return int(baseline_model.predict(X_one)[0]), "ThresholdOptimizer (DP; fallback baseline)"


def _disparate_impact_from_predictions(y_pred: np.ndarray, sensitive_features: np.ndarray) -> float:
    """Compute disparate impact as min(selection_rate)/max(selection_rate) across groups.

    This is equivalent to the common 80% rule ratio when selection_rate is defined
    as the fraction of positive predictions (label 1) per group.
    """
    df = pd.DataFrame({"y": np.asarray(y_pred, dtype=float), "s": np.asarray(sensitive_features)})
    rates = df.groupby("s")["y"].mean()
    if rates.empty:
        return float("nan")
    max_rate = float(rates.max())
    min_rate = float(rates.min())
    if np.isclose(max_rate, 0.0):
        return float("nan")
    return float(min_rate / max_rate)


@st.cache_data
def compute_dataset_level_fairness_summary(mitigation_method: str):
    """Compute dataset-level fairness metrics for the selected mitigation method.

    Uses the same dataset + target definition as the routing demo:
    - y_true: VADER_compound >= dataset median
    - feature: BERT_score
    - sensitive attribute: Race
    """
    from sklearn.metrics import accuracy_score
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )

    df = load_project_dataset().dropna(subset=["BERT_score", "VADER_compound", "Race"]).copy()
    df["Race"] = df["Race"].astype(str)

    vader_median = float(df["VADER_compound"].median())
    y_true = (df["VADER_compound"].to_numpy(dtype=float) >= vader_median).astype(int)
    X = df[["BERT_score"]].to_numpy().astype(float)
    sensitive = df["Race"].to_numpy()

    baseline_model, dp_post, dp_eg, known_groups = load_dp_mitigation_models()
    y_base = baseline_model.predict(X)

    if mitigation_method == "In-processing: Exponentiated Gradient (DP)":
        if dp_eg is None:
            y_mit = y_base
            method_label = "After Mitigation: Exponentiated Gradient (unavailable; baseline shown)"
        else:
            y_mit = dp_eg.predict(X, random_state=MITIGATION_RANDOM_STATE)
            method_label = "After Mitigation: Exponentiated Gradient (DP)"
    else:
        # Post-processing ThresholdOptimizer
        y_mit = dp_post.predict(
            X,
            sensitive_features=sensitive,
            random_state=MITIGATION_RANDOM_STATE,
        )
        method_label = "After Mitigation: ThresholdOptimizer (DP)"

    baseline_row = {
        "method": "Baseline (BERT output decision)",
        "accuracy": float(accuracy_score(y_true, y_base)),
        "dem_parity_diff": float(demographic_parity_difference(y_true, y_base, sensitive_features=sensitive)),
        "equalized_odds_diff": float(equalized_odds_difference(y_true, y_base, sensitive_features=sensitive)),
        "disparate_impact": float(_disparate_impact_from_predictions(y_base, sensitive)),
    }
    mitigated_row = {
        "method": method_label,
        "accuracy": float(accuracy_score(y_true, y_mit)),
        "dem_parity_diff": float(demographic_parity_difference(y_true, y_mit, sensitive_features=sensitive)),
        "equalized_odds_diff": float(equalized_odds_difference(y_true, y_mit, sensitive_features=sensitive)),
        "disparate_impact": float(_disparate_impact_from_predictions(y_mit, sensitive)),
    }

    summary = pd.DataFrame([baseline_row, mitigated_row])

    metric_specs = [
        ("accuracy", "Accuracy", "higher"),
        ("dem_parity_diff", "Demographic Parity Difference", "lower"),
        ("equalized_odds_diff", "Equalized Odds Difference", "lower"),
        ("disparate_impact", "Disparate Impact", "closer_to_one"),
    ]

    diff_rows = []
    for column, label, direction in metric_specs:
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
    return summary, diff_df


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
def load_preloaded_examples(mitigation_method: str):
    """Build the sidebar examples directly from exact dataset rows.

    Sentence_IDs are chosen so that the mitigation method actually flips the
    routing decision for several of them (verified against the current
    02_Data/sentiment_scores_all_systems.csv). The last three IDs are control
    examples whose decisions do NOT change.
    """
    df = load_project_dataset().copy()
    if str(mitigation_method) == "In-processing: Exponentiated Gradient (DP)":
        sentence_ids = [
            174,  # White Male: Standard queue -> Escalated (flips)
            199,  # Chinese Female: Standard queue -> Escalated (flips)
            736,  # White Female: Standard queue -> Escalated (flips)
            162,  # Indian Male: no change under EG (flips under TO)
            181,  # Black Male: no change under EG (flips under TO)
            186,  # Black Female: no change under EG (flips under TO)
            17,   # control: no change
            131,  # control: no change
            732,  # control: no change
        ]
    else:
        sentence_ids = [
            162,  # Indian Male: Standard queue -> Escalated (flips)
            166,  # Indian Female: Standard queue -> Escalated (flips)
            174,  # White Male: Standard queue -> Escalated (flips)
            181,  # Black Male: Standard queue -> Escalated (flips)
            186,  # Black Female: Standard queue -> Escalated (flips)
            193,  # Chinese Male: Standard queue -> Escalated (flips)
            198,  # Chinese Female: Standard queue -> Escalated (flips)
            736,  # White Female: Standard queue -> Escalated (flips)
            744,  # Black Male: Standard queue -> Escalated (flips)
            17,   # control: no change
            131,  # control: no change
            732,  # control: no change
        ]
    selected = df[df["Sentence_ID"].isin(sentence_ids)].copy()
    selected = selected.set_index("Sentence_ID").loc[sentence_ids].reset_index()

    examples = {}
    for row in selected.to_dict("records"):
        label = f"{row['Race']} {row['Gender']} - {row['Template_Category'].title()}"
        # Disambiguate duplicate group/category labels (e.g., two different
        # White-Male "angry" examples) so dict keys stay unique.
        if label in examples:
            label = f"{label} ({row['Name']})"
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
# =============================================================================
# DEMOGRAPHIC DETECTION (name -> race/gender signal for custom input)
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


def detect_demographic(name):
    """Detect demographic from a customer name (used for custom input)."""
    lower = str(name).lower().strip()
    if lower in NAME_DEMOGRAPHICS:
        return NAME_DEMOGRAPHICS[lower]
    # Check first name
    first = lower.split()[0] if lower else ""
    if first in NAME_DEMOGRAPHICS:
        return NAME_DEMOGRAPHICS[first]
    return ("Unknown", "Unknown")


# =============================================================================
# PRIVACY DEMO (Differential Privacy)
# =============================================================================

@st.cache_data
def load_privacy_artifact_csv(filename: str):
    """Load a saved privacy analysis CSV from 04_Results (if present)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "04_Results", filename)
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _prepare_privacy_split(max_features: int = 200, random_state: int = 42):
    """Prepare a small text classification task used only for DP demos.

    Label: whether VADER_compound is above the dataset median.
    Sensitive attribute: Race.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    df = load_project_dataset().dropna(subset=["Full_Text", "VADER_compound", "Race"]).copy()
    median_score = float(df["VADER_compound"].median())

    y = (df["VADER_compound"].to_numpy(dtype=float) >= median_score).astype(int)
    sensitive = df["Race"].astype(str).to_numpy()

    vectorizer = TfidfVectorizer(max_features=int(max_features), stop_words="english")
    X = vectorizer.fit_transform(df["Full_Text"].astype(str))

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=0.30,
        random_state=int(random_state),
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, sens_train, sens_test


def _train_lr_output_perturbation(
    X_train,
    y_train,
    X_test,
    y_test,
    sensitive_test,
    epsilon: float,
    seed: int,
):
    """Logistic regression with coefficient noise (output perturbation)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from fairlearn.metrics import demographic_parity_difference

    model = LogisticRegression(max_iter=1000, random_state=int(seed), C=1.0)
    model.fit(X_train, y_train)

    if np.isfinite(float(epsilon)):
        rng = np.random.default_rng(int(seed))
        sensitivity = 2.0 / (len(y_train) * model.C)
        scale = sensitivity / float(epsilon)
        model.coef_ = model.coef_ + rng.laplace(0.0, scale, size=model.coef_.shape)
        model.intercept_ = model.intercept_ + rng.laplace(0.0, scale, size=model.intercept_.shape)

    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None
    acc = float(accuracy_score(y_test, y_pred))
    dp_diff = float(
        abs(
            demographic_parity_difference(
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sensitive_test,
            )
        )
    )
    return acc, dp_diff, y_prob


def run_output_perturbation_sweep(
    epsilons,
    n_seeds: int = 5,
    max_features: int = 200,
    random_state: int = 42,
):
    X_train, X_test, y_train, y_test, _sens_train, sens_test = _prepare_privacy_split(
        max_features=max_features,
        random_state=random_state,
    )

    # Baseline (no privacy) probability scores for drift metrics
    from sklearn.linear_model import LogisticRegression
    base = LogisticRegression(max_iter=1000, random_state=int(random_state), C=1.0)
    base.fit(X_train, y_train)
    base_prob = base.predict_proba(X_test)[:, 1]

    def _score_gap_by_group(probs, sensitive):
        df_tmp = pd.DataFrame({"p": probs, "g": sensitive})
        means = df_tmp.groupby("g")["p"].mean()
        if means.empty:
            return 0.0
        return float(means.max() - means.min())

    rows = []
    for eps in epsilons:
        accs = []
        dps = []
        drifts = []
        score_gaps = []
        for seed in range(int(n_seeds)):
            acc, dp, y_prob = _train_lr_output_perturbation(
                X_train,
                y_train,
                X_test,
                y_test,
                sens_test,
                epsilon=float(eps),
                seed=seed,
            )
            accs.append(acc)
            dps.append(dp)
            if y_prob is not None:
                drifts.append(float(np.mean(np.abs(y_prob - base_prob))))
                score_gaps.append(_score_gap_by_group(y_prob, sens_test))

        rows.append(
            {
                "epsilon": float(eps),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "dem_parity_diff_mean": float(np.mean(dps)),
                "dem_parity_diff_std": float(np.std(dps)),
                "prob_drift_mean": float(np.mean(drifts)) if drifts else 0.0,
                "prob_drift_std": float(np.std(drifts)) if drifts else 0.0,
                "score_gap_mean": float(np.mean(score_gaps)) if score_gaps else 0.0,
                "score_gap_std": float(np.std(score_gaps)) if score_gaps else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    out["epsilon_label"] = out["epsilon"].apply(lambda x: "∞ (no privacy)" if not np.isfinite(x) else f"ε={x:g}")
    return out.sort_values("epsilon", key=lambda s: s.replace({np.inf: 1e18}))


# =============================================================================
# EXPLAINABILITY (SHAP)
# =============================================================================


@st.cache_data
def load_shap_feature_importance(top_k: int = 30):
    """Load global (dataset-level) SHAP feature importance computed offline."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "04_Results", "shap_feature_importance.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "feature" not in df.columns or "mean_shap" not in df.columns:
        return None
    df = df.copy()
    df["mean_shap"] = pd.to_numeric(df["mean_shap"], errors="coerce")
    df = df.dropna(subset=["mean_shap"]).sort_values("mean_shap", ascending=False)
    return df.head(int(top_k)).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def generate_shap_text_explanation(text: str, max_evals: int = 200):
    """Generate a SHAP text explanation for the BERT sentiment model.

    Returns:
        (html, token_df, error)

    Notes:
        - We embed `shap.getjs()` to make Streamlit rendering reliable.
        - We also return a token importance table as a robust fallback if HTML rendering is blocked.
    """
    try:
        import shap
    except Exception as exc:
        return None, None, f"SHAP unavailable: {exc}"

    pipe = load_bert_pipeline()
    if pipe is None:
        return None, None, "BERT pipeline unavailable. Install `transformers` + `torch` to enable SHAP."

    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return None, None, "Tokenizer unavailable on the BERT pipeline."

    masker = shap.maskers.Text(tokenizer)

    def _predict(texts):
        return bert_predict_proba(pipe, texts)

    explainer = shap.Explainer(
        _predict,
        masker,
        output_names=["NEGATIVE", "POSITIVE"],
        algorithm="partition",
    )

    shap_values = explainer([str(text)], max_evals=int(max_evals))

    token_df = None
    try:
        sv0 = shap_values[0]
        raw_tokens = getattr(sv0, "data", None)
        if raw_tokens is None:
            tokens = []
        elif isinstance(raw_tokens, (list, tuple, np.ndarray)):
            tokens = [str(t) for t in list(raw_tokens)]
        else:
            # Fallback: if SHAP returns a single string, approximate tokens by whitespace.
            tokens = str(raw_tokens).split()

        values = np.array(getattr(sv0, "values", []), dtype=float)
        if tokens and values.size:
            if values.ndim == 2 and values.shape[1] == 2:
                neg = values[:, 0]
                pos = values[:, 1]
            else:
                neg = values.reshape(-1)
                pos = np.zeros_like(neg)

            # Align token/value lengths if needed.
            n = int(min(len(tokens), len(neg)))
            tokens = tokens[:n]
            neg = neg[:n]
            pos = pos[:n]

            token_df = pd.DataFrame(
                {
                    "token": tokens,
                    "shap_NEGATIVE": neg,
                    "shap_POSITIVE": pos,
                    "abs_total": np.abs(neg) + np.abs(pos),
                }
            ).sort_values("abs_total", ascending=False)
    except Exception:
        token_df = None

    try:
        html_obj = shap.plots.text(shap_values[0])
        if html_obj is None:
            plot_html = None
        else:
            plot_html = getattr(html_obj, "data", None)
            if plot_html is None:
                plot_html = str(html_obj)

        if plot_html is not None and str(plot_html).strip().lower() == "none":
            plot_html = None

        if plot_html is None:
            return None, token_df, None

        js = shap.getjs()
        html = f"""
        <html>
            <head>{js}</head>
            <body style="margin:0; padding:0;">{plot_html}</body>
        </html>
        """
    except Exception as exc:
        return None, token_df, f"Failed to render SHAP HTML: {exc}"

    return html, token_df, None


# =============================================================================
# NAME-SWAP COMPARISON (same complaint, 40 different names)
# =============================================================================

NAME_SWAP_SYSTEMS = [
    ("RoBERTa (Twitter)", "RoBERTa_score"),
    ("DistilBERT", "BERT_score"),
    ("VADER", "VADER_compound"),
    ("TextBlob", "TextBlob_polarity"),
]


def name_swap_section():
    """Visualize the core research question: does changing only the customer's
    name change the sentiment score? Every variant shares identical complaint
    wording, so score differences are attributable to the name alone."""
    df = load_project_dataset()
    available = [(n, c) for n, c in NAME_SWAP_SYSTEMS if c in df.columns]
    if not available:
        return

    st.markdown("---")
    st.markdown("## 🔄 Same Complaint, Different Names")
    st.caption(
        "Each bar is the same complaint text with a different customer name "
        "(Bertrand & Mullainathan–style controlled comparison). Any score "
        "difference is caused by the name alone."
    )

    templates = (
        df.drop_duplicates("Template_Number")
          .sort_values("Template_Number")[
              ["Template_Number", "Template_Category", "Name", "Full_Text"]
          ]
          .set_index("Template_Number")
    )

    def _template_label(tmpl_num):
        row = templates.loc[tmpl_num]
        body = extract_complaint_body(row["Name"], row["Full_Text"])
        return f"{str(row['Template_Category']).title()} — '{body}'"

    c1, c2 = st.columns([2, 1])
    with c1:
        tmpl_num = st.selectbox(
            "Complaint template",
            options=templates.index.tolist(),
            format_func=_template_label,
            key="name_swap_template",
        )
    with c2:
        sys_label = st.selectbox(
            "Scoring system",
            options=[n for n, _ in available],
            index=0,
            key="name_swap_system",
        )

    score_col = dict(available)[sys_label]
    sub = df[df["Template_Number"] == tmpl_num].copy()

    max_i = sub[score_col].idxmax()
    min_i = sub[score_col].idxmin()
    spread = sub.loc[max_i, score_col] - sub.loc[min_i, score_col]

    m1, m2, m3 = st.columns(3)
    m1.metric("Score spread (max − min)", f"{spread:.4f}")
    m2.metric(
        "Most negative",
        f"{sub.loc[min_i, 'Name']}",
        f"{sub.loc[min_i, score_col]:.4f}",
    )
    m3.metric(
        "Least negative",
        f"{sub.loc[max_i, 'Name']}",
        f"{sub.loc[max_i, score_col]:.4f}",
    )

    chart = (
        alt.Chart(sub)
        .mark_bar()
        .encode(
            x=alt.X("Name:N", sort="-y", title=None,
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{score_col}:Q", title="Sentiment score",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("Demographic_Group:N", title="Demographic"),
            tooltip=[
                "Name", "Demographic_Group",
                alt.Tooltip(f"{score_col}:Q", format=".4f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, width="stretch")

    table_cols = ["Name", "Demographic_Group", score_col]
    display = sub.sort_values(score_col).copy()
    if score_col == "BERT_score":
        baseline_model, *_ = load_dp_mitigation_models()
        display["Baseline routing"] = np.where(
            baseline_model.predict(
                display["BERT_score"].to_numpy().reshape(-1, 1)
            ) == 1,
            "Standard queue",
            "Escalated (urgent)",
        )
        table_cols.append("Baseline routing")
        n_dep = int((display["Baseline routing"] == "Standard queue").sum())
        st.caption(
            f"Under the baseline routing model, **{n_dep}/40** names are "
            "deprioritized to the standard queue for this exact complaint — "
            "the rest get escalated. This is how a small per-score difference "
            "becomes a categorical routing difference at a decision threshold."
        )
    st.dataframe(
        display[table_cols].rename(columns={score_col: "Score"}),
        width="stretch",
        hide_index=True,
    )


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

    bert_pipe = load_bert_pipeline()

    # Sidebar: Pre-loaded examples
    st.sidebar.header("📋 Pre-loaded Examples")
    st.sidebar.markdown("Click to load an example:")

    mitigation_method = st.session_state.get(
        "mitigation_method",
        "Post-processing: ThresholdOptimizer (DP)",
    )
    examples = load_preloaded_examples(str(mitigation_method))

    selected_example = st.sidebar.radio(
        "Choose example:", list(examples.keys()), index=0
    )

    st.sidebar.subheader("⚖️ Mitigation technique")
    mitigation_method = st.sidebar.selectbox(
        "Choose mitigation method",
        options=[
            "Post-processing: ThresholdOptimizer (DP)",
            "In-processing: Exponentiated Gradient (DP)",
        ],
        index=0,
        key="mitigation_method",
    )

    input_mode = st.sidebar.radio(
        "Input source",
        ["Pre-loaded dataset example", "Custom input"],
        index=0,
    )
    custom_mode = input_mode == "Custom input"

    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "analysis_example_key" not in st.session_state:
        st.session_state.analysis_example_key = None

    # Input section
    col1, col2 = st.columns([1, 2])

    with col1:
        if custom_mode:
            customer_name = st.text_input(
                "👤 Customer Name",
                value="Jamal",
                help="Names are mapped to demographic signals using the project's 40-name dictionary; unrecognised names use the neutral baseline.",
            )
            complaint_text = st.text_area(
                "💬 Complaint Text",
                value="Jamal is angry about the delayed delivery",
                height=120,
            )
        else:
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
        if custom_mode:
            st.info("""
            **How this works (custom input):**
                    1. Enter any customer name and complaint text
                    2. The app scores the text live with DistilBERT (VADER fallback if unavailable)
                    3. It shows a **routing decision** (a small classifier trained on the project dataset)
                       and then applies a fairness mitigation method:
                       - **Post-processing:** ThresholdOptimizer (Demographic Parity)
                       - **In-processing:** Exponentiated Gradient (Demographic Parity)
                    4. The explanation highlights which complaint words push sentiment

            The name only affects the sensitive feature used by the fairness mitigator —
            it does not change the sentiment score itself.
            """)
        else:
            st.info("""
            **How this works:**
                    1. Choose one of the pre-loaded dataset examples
                    2. The app uses the stored BERT score from `sentiment_scores_all_systems.csv`
                        3. It shows a **routing decision** (a small classifier trained on the project dataset)
                                and then applies a fairness mitigation method:
                                - **Post-processing:** ThresholdOptimizer (Demographic Parity)
                                - **In-processing:** Exponentiated Gradient (Demographic Parity)
                  *Note:* `Escalated (urgent)` means the complaint is flagged negative and sent to a
                  human — the better outcome for the customer. `Standard queue` means deprioritized.
            4. The explanation highlights which complaint words push sentiment

            The inputs are locked so the demo always uses exact dataset rows.
            """)

    # Persist analysis results across Streamlit reruns so other UI interactions
    # (e.g., privacy sweep button clicks) don't make results disappear.
    if custom_mode:
        current_example_key = (
            "custom",
            str(customer_name),
            str(complaint_text),
            str(mitigation_method),
        )
    else:
        current_example_key = (
            str(examples[selected_example]["name"]),
            str(examples[selected_example]["full_text"]),
            str(mitigation_method),
        )
    if st.session_state.analysis_example_key != current_example_key:
        st.session_state.analysis = None
        st.session_state.analysis_example_key = None

    # Analyze button
    analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", width="stretch")
    if analyze_clicked:
        if custom_mode:
            full_text = str(complaint_text).strip()
            if not full_text:
                st.error("Please enter a complaint text to analyze.")
                st.stop()

            race, gender = detect_demographic(customer_name)
            example_row = {
                "name": str(customer_name),
                "sentence_id": None,
                "template_category": "custom",
                "emotion_intensity": "",
                "full_text": full_text,
                "race": race,
                "gender": gender,
            }

            if bert_pipe is not None:
                probs = bert_predict_proba(bert_pipe, [full_text])[0]
                p_neg, p_pos = float(probs[0]), float(probs[1])
                bert_score = float(p_pos - p_neg)
                bert_label = "NEGATIVE" if p_neg >= p_pos else "POSITIVE"
                bert_confidence = float(max(p_neg, p_pos))
                score_source = "Live DistilBERT prediction"
            else:
                vs = load_vader().polarity_scores(full_text)
                bert_score = float(vs["compound"])
                bert_label = "NEGATIVE" if vs["compound"] < 0 else "POSITIVE"
                bert_confidence = float(abs(vs["compound"]))
                score_source = "VADER compound (transformers unavailable)"
        else:
            example_row = get_dataset_example(
                examples[selected_example]["name"],
                examples[selected_example]["full_text"],
            )
            if example_row is None:
                st.error("Couldn't find the selected pre-loaded example in sentiment_scores_all_systems.csv.")
                st.stop()

            full_text = example_row["full_text"]
            bert_score = float(example_row["bert_score"])
            bert_label = str(example_row.get("bert_label", ""))
            bert_confidence = float(example_row.get("bert_confidence", 0.0))
            race = example_row["race"]
            gender = example_row["gender"]
            score_source = "Stored dataset BERT score"

        complaint_body = extract_complaint_body(customer_name, full_text)

        # Load mitigation models (trained from the project dataset)
        baseline_model, dp_post, dp_eg, known_groups = load_dp_mitigation_models()

        X_one = np.array([[bert_score]], dtype=float)
        baseline_pred = int(baseline_model.predict(X_one)[0])
        mitigated_pred, mitigation_method_label = _predict_mitigated_label(
            mitigation_method,
            X_one=X_one,
            race=str(race),
            baseline_model=baseline_model,
            dp_post=dp_post,
            dp_eg=dp_eg,
            known_groups=known_groups,
        )

        st.session_state.analysis = {
            "example_row": example_row,
            "complaint_body": complaint_body,
            "bert_score": bert_score,
            "bert_label": bert_label,
            "bert_confidence": bert_confidence,
            "race": race,
            "gender": gender,
            "baseline_pred": baseline_pred,
            "mitigated_pred": mitigated_pred,
            "mitigation_method": mitigation_method,
            "mitigation_method_label": mitigation_method_label,
            "bert_pipe_is_none": (bert_pipe is None),
            "score_source": score_source,
            "custom": custom_mode,
        }
        st.session_state.analysis_example_key = current_example_key

    analysis = st.session_state.analysis
    if analysis is not None and st.session_state.analysis_example_key == current_example_key:
        example_row = analysis["example_row"]
        bert_score = float(analysis["bert_score"])
        bert_label = str(analysis.get("bert_label", ""))
        bert_confidence = float(analysis.get("bert_confidence", 0.0))
        race = str(analysis["race"])
        gender = str(analysis["gender"])
        baseline_pred = int(analysis["baseline_pred"])
        mitigated_pred = int(analysis["mitigated_pred"])
        mitigation_method = str(analysis.get("mitigation_method", "Post-processing: ThresholdOptimizer (DP)"))
        mitigation_method_label = str(analysis.get("mitigation_method_label", "ThresholdOptimizer (DP)"))
        score_source = str(analysis.get("score_source", "Stored dataset BERT score"))
        is_custom = bool(analysis.get("custom", False))
        # Detected demographic
        st.markdown("---")
        if race != "Unknown":
            st.markdown(f"**Detected Demographic Signal:** {race} {gender}")
        else:
            st.markdown("**Detected Demographic Signal:** Not recognized "
                       "(using neutral baseline)")

        # Results: Side by side
        st.markdown("## 📊 Results")

        st.markdown("### 🧾 Sentiment score")
        st.caption(
            f"Score source: {score_source}. "
            "Some words can push towards POSITIVE in local explanations while the overall prediction remains NEGATIVE."
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("BERT Label", bert_label if bert_label else "(missing)")
        s2.metric("BERT Confidence", f"{bert_confidence:.3f}")
        s3.metric("BERT Score", f"{bert_score:.3f}")

        st.markdown("### 🚦 Routing decision (baseline vs fairness mitigation)")
        st.caption(
            "This routing model is trained on the project dataset using a VADER-derived target label. "
            "**Escalated (urgent)** = flagged as strongly negative and sent to a human — the better outcome for the customer. "
            "**Standard queue** = deprioritized. Mitigation changes the *decision*, not the sentiment score."
        )

        r0, r1, r2 = st.columns(3)
        r0.metric("Mitigation method", mitigation_method_label)
        baseline_decision_label = decision_label(baseline_pred)
        mitigated_decision_label = decision_label(mitigated_pred)
        decision_shift = f"{baseline_decision_label} -> {mitigated_decision_label}"
        r1.metric("Baseline decision", baseline_decision_label)
        r2.metric("After mitigation", mitigated_decision_label)
        if baseline_pred != mitigated_pred:
            if mitigated_pred == 0:
                st.success(
                    f"Mitigation changed this routing decision: {decision_shift}. "
                    "The baseline was deprioritizing this complaint — mitigation restored urgent treatment."
                )
            else:
                st.success(
                    f"Mitigation changed this routing decision: {decision_shift}. "
                    "The baseline was escalating this complaint — mitigation moved it to the standard queue."
                )
        elif race == "White" and baseline_pred == 1:
            st.info(f"White reference example: {decision_shift} after mitigation.")
        else:
            st.info(f"No routing change for this example: {decision_shift}.")

        st.markdown("### 📈 Dataset-Level Fairness (Baseline vs Mitigation)")
        st.caption(
            "These values are computed over the full dataset (not just this one example). "
            "They stay fixed even when you change a single person name in the demo input."
        )
        try:
            summary, diff_df = compute_dataset_level_fairness_summary(mitigation_method)
            if not diff_df.empty:
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

                with st.expander("Show dataset audit tables"):
                    st.dataframe(diff_df, width="stretch", hide_index=True)
                    st.markdown("##### Raw audited rows")
                    st.dataframe(summary, width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"Dataset-level fairness summary unavailable: {exc}")

        # Explainability
        st.markdown("## 🧠 Explainability")
        st.caption(
            "SHAP attributes how much each word/token pushes the BERT sentiment prediction towards NEGATIVE vs POSITIVE. "
            "LIME views come from pre-computed explanations of the project's TF-IDF model."
        )

        tabs = st.tabs([
            "Local explanation (selected text)",
            "Global importance (saved)",
            "Saved LIME example",
        ])

        with tabs[0]:
            pipe = load_bert_pipeline()
            if pipe is None:
                st.info(
                    "Local SHAP needs the Transformers BERT pipeline. Install `transformers` + `torch` to enable it."
                )
            else:
                probs = bert_predict_proba(pipe, [str(example_row["full_text"])])[0]
                p_neg, p_pos = float(probs[0]), float(probs[1])
                live_label = "NEGATIVE" if p_neg >= p_pos else "POSITIVE"

                l1, l2, l3 = st.columns(3)
                l1.metric("Live BERT Label", live_label)
                l2.metric("P(NEGATIVE)", f"{p_neg:.3f}")
                l3.metric("P(POSITIVE)", f"{p_pos:.3f}")

                c1, c2 = st.columns([1, 1])
                with c1:
                    max_evals = st.slider(
                        "SHAP max evaluations",
                        min_value=50,
                        max_value=600,
                        value=200,
                        step=50,
                        help="Lower = faster, higher = more accurate attribution (but slower).",
                    )
                with c2:
                    run_shap = st.button("Generate SHAP explanation", width="stretch")

                shap_key = (
                    st.session_state.get("analysis_example_key"),
                    int(max_evals),
                )
                if "shap_local" not in st.session_state:
                    st.session_state.shap_local = {}

                if run_shap:
                    with st.spinner("Computing SHAP attributions..."):
                        shap_html, token_df, shap_error = generate_shap_text_explanation(
                            str(example_row["full_text"]),
                            max_evals=int(max_evals),
                        )
                    st.session_state.shap_local[shap_key] = {
                        "html": shap_html,
                        "token_df": token_df,
                        "error": shap_error,
                    }

                cached = st.session_state.shap_local.get(shap_key)
                if cached is None:
                    st.info("Click **Generate SHAP explanation** to compute attributions.")
                else:
                    shap_err = cached.get("error")
                    shap_html = cached.get("html")
                    token_df = cached.get("token_df")

                    if shap_html:
                        components.html(shap_html, height=650, scrolling=True)
                    if token_df is not None:
                        st.markdown("#### Token contributions (fallback view)")
                        st.dataframe(
                            token_df.head(40),
                            width="stretch",
                            hide_index=True,
                        )
                    if shap_err:
                        with st.expander("SHAP render details", expanded=False):
                            st.caption("The interactive HTML view could not be rendered, but the fallback token table may still be available.")
                            st.code(str(shap_err))

                    if (not shap_html) and (token_df is None) and (not shap_err):
                        st.warning(
                            "SHAP ran but returned no renderable output. Try increasing **SHAP max evaluations** or rerun."
                        )

        with tabs[1]:
            imp = load_shap_feature_importance(top_k=40)
            if imp is None:
                st.info("No saved SHAP table found at `04_Results/shap_feature_importance.csv`.")
            else:
                st.markdown("### Top features by mean |SHAP|")
                st.dataframe(imp, width="stretch", hide_index=True)

        with tabs[2]:
            if is_custom:
                st.info("Saved LIME explanations exist only for pre-generated dataset examples.")
            else:
                lime_html, lime_file = load_precomputed_lime_html(
                    race, gender, example_row.get("template_category", "")
                )
                if lime_html is None:
                    st.info(
                        "No pre-computed LIME explanation for this example. "
                        "Saved LIME files exist for a fixed set of (race, gender, category) pairs."
                    )
                else:
                    st.caption(
                        f"Pre-computed LIME explanation (`{lime_file}`) — explains the project's "
                        "TF-IDF + Logistic Regression model, not the BERT score above. "
                        "Loaded on demand (the file is ~1.3 MB of interactive HTML)."
                    )
                    if st.button("Load saved LIME explanation", key="load_lime_btn"):
                        components.html(lime_html, height=600, scrolling=True)
    else:
        st.info("Click **Analyze Sentiment** to generate results.")

    name_swap_section()

    st.markdown("---")
    with st.expander("🔒 Privacy: Differential Privacy (DP)", expanded=False):
        st.caption(
            "Note: in this app, 'DP mitigation' above refers to Demographic Parity. "
            "This section is about Differential Privacy (ε, δ)."
        )

        st.markdown("### Saved project outputs")
        eps_df = load_privacy_artifact_csv("privacy_epsilon_analysis.csv")
        if eps_df is None:
            st.info("No saved privacy sweep found at `04_Results/privacy_epsilon_analysis.csv`.")
        else:
            st.dataframe(eps_df, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("### Run a Differential Privacy sweep (output perturbation)")
        st.caption(
            "This trains a small TF-IDF + Logistic Regression model and adds Laplace noise to the learned coefficients. "
            "Lower ε means stronger privacy and typically more score drift."
        )

        eps_options = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
        default_eps = [0.01, 0.1, 1.0, 10.0, float("inf")]

        def _fmt_eps(v):
            return "∞ (no privacy)" if not np.isfinite(float(v)) else f"ε={float(v):g}"

        selected_eps = st.multiselect(
            "Epsilon values (ε)",
            options=eps_options,
            default=default_eps,
            format_func=_fmt_eps,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            n_seeds = st.number_input("Random seeds", min_value=1, max_value=15, value=5, step=1)
        with c2:
            max_features = st.number_input("TF-IDF max features", min_value=50, max_value=2000, value=200, step=50)
        with c3:
            run_sweep = st.button("Run DP sweep", width="stretch")

        if run_sweep:
            if not selected_eps:
                st.error("Select at least one ε value.")
            else:
                with st.spinner("Training models and applying DP noise..."):
                    sweep_df = run_output_perturbation_sweep(
                        epsilons=selected_eps,
                        n_seeds=int(n_seeds),
                        max_features=int(max_features),
                        random_state=42,
                    )

                st.dataframe(
                    sweep_df[[
                        "epsilon_label",
                        "accuracy_mean",
                        "accuracy_std",
                        "dem_parity_diff_mean",
                        "dem_parity_diff_std",
                        "prob_drift_mean",
                        "prob_drift_std",
                        "score_gap_mean",
                        "score_gap_std",
                    ]],
                    width="stretch",
                    hide_index=True,
                )

                chart_df = sweep_df.set_index("epsilon_label")[
                    ["accuracy_mean", "dem_parity_diff_mean", "prob_drift_mean", "score_gap_mean"]
                ]
                st.line_chart(chart_df)

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
