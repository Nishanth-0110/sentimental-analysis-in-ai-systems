"""
==============================================================================
Streamlit Evaluator App: Fairness Metrics Dashboard
==============================================================================
Evaluator-facing dashboard focused on fairness metrics across demographic groups.

Run with:  streamlit run app2.py
==============================================================================
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Fairness Metrics Evaluator",
    page_icon="📊",
    layout="wide",
)


GROUP_OPTIONS = {
    "Race": "Race",
    "Gender": "Gender",
    "Demographic Group": "Demographic_Group",
}

SYSTEM_OPTIONS = {
    "Baseline Model": "baseline_pred",
    "After Mitigation (ThresholdOptimizer)": "mitigated_pred",
}


@st.cache_data
def load_project_dataset():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "02_Data", "sentiment_scores_all_systems.csv")
    return pd.read_csv(data_path)


@st.cache_data
def prepare_audit_data():
    df = load_project_dataset().copy()
    df = df.dropna(subset=["BERT_score", "VADER_compound", "Race", "Gender", "Demographic_Group"]).copy()

    vader_median = float(df["VADER_compound"].median())
    df["y_true"] = (df["VADER_compound"] >= vader_median).astype(int)

    X = df[["BERT_score"]].to_numpy().astype(float)
    y = df["y_true"].to_numpy().astype(int)
    sensitive = df["Race"].astype(str).to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    sens_train = sensitive[train_idx]
    sens_test = sensitive[test_idx]

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)

    mitigator = ThresholdOptimizer(
        estimator=baseline,
        constraints="demographic_parity",
        objective="accuracy_score",
        prefit=True,
    )
    mitigator.fit(X_train, y_train, sensitive_features=sens_train)

    audit_df = df.iloc[test_idx].copy()
    audit_df["baseline_pred"] = baseline.predict(X_test)
    audit_df["mitigated_pred"] = mitigator.predict(X_test, sensitive_features=sens_test)
    return audit_df.reset_index(drop=True)


def compute_equal_opportunity_difference(y_true, y_pred, sensitive_features):
    metric_frame = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return float(metric_frame.difference())


def build_group_rates(audit_df, group_col, pred_col):
    mf = MetricFrame(
        metrics={
            "Selection Rate": selection_rate,
            "True Positive Rate": true_positive_rate,
        },
        y_true=audit_df["y_true"],
        y_pred=audit_df[pred_col],
        sensitive_features=audit_df[group_col],
    )

    rates_df = mf.by_group.reset_index()
    rates_df.columns = [group_col, "Selection Rate", "True Positive Rate"]
    rates_df["Selection Rate"] = rates_df["Selection Rate"].astype(float).round(3)
    rates_df["True Positive Rate"] = rates_df["True Positive Rate"].astype(float).round(3)

    sample_counts = audit_df.groupby(group_col).size().reset_index(name="Samples")
    rates_df = rates_df.merge(sample_counts, on=group_col, how="left")
    return rates_df[[group_col, "Samples", "Selection Rate", "True Positive Rate"]]


def add_gap_summary(rates_df):
    selection_gap = float(rates_df["Selection Rate"].max() - rates_df["Selection Rate"].min())
    tpr_gap = float(rates_df["True Positive Rate"].max() - rates_df["True Positive Rate"].min())
    return round(selection_gap, 3), round(tpr_gap, 3)


def compute_overall_metrics(audit_df, group_col, pred_col):
    y_true = audit_df["y_true"]
    y_pred = audit_df[pred_col]
    sensitive = audit_df[group_col]

    dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive))
    di_ratio = float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive))
    eo_diff = float(compute_equal_opportunity_difference(y_true, y_pred, sensitive))
    eod_diff = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive))

    return {
        "Demographic Parity Difference": round(dp_diff, 3),
        "Disparate Impact": round(di_ratio, 3),
        "Equal Opportunity Difference": round(eo_diff, 3),
        "Equalized Odds Difference": round(eod_diff, 3),
    }


def build_metric_cards(metrics):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demographic Parity Difference", f"{metrics['Demographic Parity Difference']:.3f}")
    c2.metric("Disparate Impact", f"{metrics['Disparate Impact']:.3f}")
    c3.metric("Equal Opportunity Difference", f"{metrics['Equal Opportunity Difference']:.3f}")
    c4.metric("Equalized Odds Difference", f"{metrics['Equalized Odds Difference']:.3f}")


def load_overall_comparison_csv():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(
        repo_root,
        "04_Results",
        "mitigation_comparison_bert_score_constraints_with_deltas.csv",
    )
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    keep = df[
        df["method"].isin(
            ["Baseline (BERT_score -> LR)", "ThresholdOptimizer (Demographic Parity)"]
        )
    ].copy()
    return keep if not keep.empty else None


def main():
    st.title("Fairness Metrics Dashboard")
    st.caption(
        "Evaluator dashboard showing fairness metrics across demographic groups using the project dataset."
    )

    audit_df = prepare_audit_data()

    st.sidebar.header("Controls")
    group_label = st.sidebar.selectbox("Group by", list(GROUP_OPTIONS.keys()), index=2)
    group_col = GROUP_OPTIONS[group_label]

    st.sidebar.markdown("**Dataset:** `02_Data/sentiment_scores_all_systems.csv`")
    st.sidebar.markdown("**Fairness target:** final predicted decision")

    baseline_metrics = compute_overall_metrics(audit_df, group_col, "baseline_pred")
    mitigated_metrics = compute_overall_metrics(audit_df, group_col, "mitigated_pred")
    baseline_rates = build_group_rates(audit_df, group_col, "baseline_pred")
    mitigated_rates = build_group_rates(audit_df, group_col, "mitigated_pred")
    baseline_sel_gap, baseline_tpr_gap = add_gap_summary(baseline_rates)
    mitigated_sel_gap, mitigated_tpr_gap = add_gap_summary(mitigated_rates)

    st.markdown("## Overall Fairness Metrics")
    left, right = st.columns(2)
    with left:
        st.markdown("### Baseline Model")
        build_metric_cards(baseline_metrics)
    with right:
        st.markdown("### After Mitigation")
        build_metric_cards(mitigated_metrics)

    st.markdown("## Difference After Mitigation")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "DP Difference",
        f"{mitigated_metrics['Demographic Parity Difference']:.3f}",
        delta=f"{mitigated_metrics['Demographic Parity Difference'] - baseline_metrics['Demographic Parity Difference']:+.3f}",
    )
    d2.metric(
        "Disparate Impact",
        f"{mitigated_metrics['Disparate Impact']:.3f}",
        delta=f"{mitigated_metrics['Disparate Impact'] - baseline_metrics['Disparate Impact']:+.3f}",
    )
    d3.metric(
        "Equal Opportunity Diff",
        f"{mitigated_metrics['Equal Opportunity Difference']:.3f}",
        delta=f"{mitigated_metrics['Equal Opportunity Difference'] - baseline_metrics['Equal Opportunity Difference']:+.3f}",
    )
    d4.metric(
        "Equalized Odds Diff",
        f"{mitigated_metrics['Equalized Odds Difference']:.3f}",
        delta=f"{mitigated_metrics['Equalized Odds Difference'] - baseline_metrics['Equalized Odds Difference']:+.3f}",
    )

    st.info(
        "Interpretation: lower is better for Demographic Parity Difference, Equal Opportunity Difference, "
        "and Equalized Odds Difference. Disparate Impact should be closer to 1.0, with 0.80+ often used "
        "as a practical threshold."
    )

    st.markdown("## Group-wise Selection Rate")
    selection_compare = baseline_rates[[group_col, "Selection Rate"]].rename(
        columns={"Selection Rate": "Baseline Selection Rate"}
    ).merge(
        mitigated_rates[[group_col, "Selection Rate"]].rename(
            columns={"Selection Rate": "Mitigated Selection Rate"}
        ),
        on=group_col,
        how="inner",
    )
    selection_compare["Selection Rate Change"] = (
        selection_compare["Mitigated Selection Rate"] - selection_compare["Baseline Selection Rate"]
    ).round(3)
    selection_fig = px.bar(
        selection_compare.melt(
            id_vars=[group_col, "Selection Rate Change"],
            value_vars=["Baseline Selection Rate", "Mitigated Selection Rate"],
            var_name="System",
            value_name="Selection Rate",
        ),
        x=group_col,
        y="Selection Rate",
        color="System",
        text="Selection Rate",
        barmode="group",
        title=f"Selection Rate by {group_label}: Baseline vs Mitigated",
    )
    selection_fig.update_layout(xaxis_title=group_label, yaxis_title="Selection Rate")
    st.plotly_chart(selection_fig, width="stretch")

    gap1, gap2 = st.columns(2)
    gap1.metric("Selection Rate Gap", f"{mitigated_sel_gap:.3f}", delta=f"{mitigated_sel_gap - baseline_sel_gap:+.3f} vs baseline")
    gap2.metric("TPR Gap", f"{mitigated_tpr_gap:.3f}", delta=f"{mitigated_tpr_gap - baseline_tpr_gap:+.3f} vs baseline")

    st.markdown("## Group-wise Equal Opportunity View")
    tpr_compare = baseline_rates[[group_col, "True Positive Rate"]].rename(
        columns={"True Positive Rate": "Baseline TPR"}
    ).merge(
        mitigated_rates[[group_col, "True Positive Rate"]].rename(
            columns={"True Positive Rate": "Mitigated TPR"}
        ),
        on=group_col,
        how="inner",
    )
    tpr_compare["TPR Change"] = (tpr_compare["Mitigated TPR"] - tpr_compare["Baseline TPR"]).round(3)
    tpr_fig = px.bar(
        tpr_compare.melt(
            id_vars=[group_col, "TPR Change"],
            value_vars=["Baseline TPR", "Mitigated TPR"],
            var_name="System",
            value_name="True Positive Rate",
        ),
        x=group_col,
        y="True Positive Rate",
        color="System",
        text="True Positive Rate",
        barmode="group",
        title=f"True Positive Rate by {group_label}: Baseline vs Mitigated",
    )
    tpr_fig.update_layout(xaxis_title=group_label, yaxis_title="True Positive Rate")
    st.plotly_chart(tpr_fig, width="stretch")

    st.markdown("## Group-wise Fairness Table")
    group_compare = baseline_rates.rename(
        columns={
            "Selection Rate": "Baseline Selection Rate",
            "True Positive Rate": "Baseline TPR",
        }
    ).merge(
        mitigated_rates.rename(
            columns={
                "Selection Rate": "Mitigated Selection Rate",
                "True Positive Rate": "Mitigated TPR",
            }
        ),
        on=[group_col, "Samples"],
        how="inner",
    )
    group_compare["Selection Rate Change"] = (
        group_compare["Mitigated Selection Rate"] - group_compare["Baseline Selection Rate"]
    ).round(3)
    group_compare["TPR Change"] = (
        group_compare["Mitigated TPR"] - group_compare["Baseline TPR"]
    ).round(3)
    st.dataframe(group_compare, width="stretch", hide_index=True)

    st.markdown("## Baseline vs Mitigated Comparison")
    comparison_df = pd.DataFrame(
        [
            {
                "System": "Baseline Model",
                **baseline_metrics,
            },
            {
                "System": "After Mitigation (ThresholdOptimizer)",
                **mitigated_metrics,
            },
        ]
    )
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    overall_csv = load_overall_comparison_csv()
    if overall_csv is not None:
        st.markdown("## Project Audit CSV Reference")
        st.caption("These are the saved overall mitigation metrics from the project results CSV.")
        st.dataframe(overall_csv, width="stretch", hide_index=True)

    comparison_melt = comparison_df.melt(id_vars="System", var_name="Metric", value_name="Value")
    compare_fig = px.bar(
        comparison_melt,
        x="Metric",
        y="Value",
        color="System",
        barmode="group",
        title=f"Fairness Metric Comparison by {group_label}",
    )
    st.plotly_chart(compare_fig, width="stretch")


if __name__ == "__main__":
    main()
