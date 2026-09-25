# Auditing Gender & Race Bias in Customer-Service Sentiment AI

> **Live demo:** https://sentimental-analysis-in-ai-systems.streamlit.app

**Does a sentiment model treat identical complaints differently when only the
customer's name — and therefore its perceived race/gender — changes?**

This project builds a controlled dataset of 800 customer-service complaints
where the *only* thing that varies between sentences is the customer name,
scores every sentence with four sentiment systems (VADER, TextBlob,
DistilBERT, and a Twitter-trained RoBERTa), and then runs a full
responsible-AI audit: bias detection,
Fairlearn metrics, bias-mitigation algorithms, SHAP/LIME explainability and
differential-privacy analysis. An interactive Streamlit app lets you compare a
baseline routing model against fairness-mitigated models on real examples or
on custom input.

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [What the project actually does](#what-the-project-actually-does)
3. [Architecture](#architecture)
4. [Repository structure](#repository-structure)
5. [Installation](#installation)
6. [Running the pipeline](#running-the-pipeline)
7. [Running the demo apps](#running-the-demo-apps)
8. [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
9. [Results](#results)
10. [Limitations](#limitations)
11. [Future improvements](#future-improvements)

---

## Problem statement

AI systems that triage customer complaints can behave differently depending on
signals correlated with a customer's demographic identity — most commonly the
name. A biased router that systematically deprioritizes complaints from some
demographic groups creates both ethical and legal exposure (e.g., the EEOC
"80% rule" for disparate impact).

To measure this cleanly, the project uses a **controlled comparison**: every
complaint template is instantiated once per customer name, so any score
difference between demographic groups is attributable to the name alone.

## What the project actually does

| Component | Implementation |
|---|---|
| Controlled dataset | `01_dataset_generation.py` — 20 complaint templates × 8 demographic groups × 5 names = **800 rows** |
| Sentiment scoring | `02_sentiment_analysis.py` — **VADER**, **TextBlob**, **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`, scores mapped to `[-1, 1]`), and **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`, signed score = *P*(pos) − *P*(neg)) |
| Bias detection | `03_bias_detection.py` — group statistics vs a White-Male baseline, Welch t-tests, Mann–Whitney U, ANOVA, Cohen's *d* effect sizes, intersectional analysis |
| Fairness metrics | `04_fairness_metrics.py` — **Fairlearn** Demographic Parity, Equal Opportunity, Equalized Odds, Disparate Impact, Calibration |
| Bias mitigation | `05_bias_mitigation.py` — **CDA** (counterfactual name-swap augmentation), **Exponentiated Gradient**, **ThresholdOptimizer** on TF-IDF + Logistic Regression |
| BERT-score mitigation | `05b_bias_mitigation_bert_score.py` — Reweighing, Exponentiated Gradient and ThresholdOptimizer applied to a 1-feature (`BERT_score`) routing classifier |
| Explainability | `06_explainability.py` — **SHAP** (Transformer + Linear explainers) and **LIME**, including same-template cross-demographic comparisons |
| Privacy | `07_privacy.py` — Laplace **output perturbation** across ε, three-way accuracy/fairness/privacy tradeoff, and **Opacus DP-SGD** |
| Visualizations | `08_visualizations.py` — ten charts in `04_Results/` |
| Interactive demo | `07_Demo/app.py` — baseline-vs-mitigated routing demo, live BERT scoring, custom input mode, SHAP explanations, and a "same complaint / 40 names" comparison view across all four systems |
| Metrics dashboard | `07_Demo/app2.py` — selection rates, TPR parity, per-group confusion matrices, mitigation deltas |

## Architecture

```
01_dataset_generation.py        800 controlled complaint rows
            │
            ▼
02_Data/complaint_dataset_800.csv
            │
            ▼
02_sentiment_analysis.py   VADER + TextBlob + DistilBERT + RoBERTa scores
            │
            ▼
02_Data/sentiment_scores_all_systems.csv  ◄────────────┐
            │                                        │
   ┌────────┼──────────┬─────────────┬───────────┐   │
   ▼        ▼          ▼             ▼           ▼   │
03_bias  04_fairness  05_mitigate   06_shap    07_privacy
_detection  _metrics   + 05b         _lime     (DP-SGD)
   │        │          │             │           │
   └────────┴────┬─────┴─────────────┴───────────┘
                 ▼
        04_Results/ (csvs, charts, LIME html)
                 │
                 ▼
        07_Demo/app.py  +  app2.py   (Streamlit)
```

## Repository structure

```
.
├── 02_Data/                       # generated dataset + sentiment scores
├── 03_Code/                       # pipeline scripts + run_all.py master runner
├── 04_Results/                    # generated CSVs, PNG charts, LIME html
├── 05_Analysis/                   # manual chatbot-testing template
├── 07_Demo/                       # Streamlit apps (+ its own requirements.txt)
│   └── models/                    # small artifacts saved by step 5 (not used by app)
├── _demo_images/                  # archived UI screenshots
├── requirements.txt               # pinned deps for the pipeline (Python 3.11)
├── runtime.txt                    # python-3.11
└── README.md
```

## Installation

Requires **Python 3.11** (see `runtime.txt`).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

`requirements.txt` pins the exact verified versions and installs a **CPU-only**
PyTorch wheel, so no CUDA toolkit is needed.

## Running the pipeline

```bash
cd 03_Code
python run_all.py              # steps 1–8 (first run downloads ~760MB of transformer models)
python run_all.py --step 4     # single step
python run_all.py --skip-bert  # quick pass WITHOUT the transformer models
```

> **Warning:** `--skip-bert` writes *synthetic placeholder* BERT/RoBERTa columns
> and marks them with `BERT_synthetic = True` / `RoBERTa_synthetic = True`.
> Downstream analyses on those columns are not real results — re-run step 2
> without the flag before using them.

Auxiliary scripts (not run by `run_all.py`):

```bash
python 05b_bias_mitigation_bert_score.py   # mitigation on the BERT-score router
python 07_privacy.py                       # also runs the Opacus DP-SGD section
python generate_testing_template.py        # chatbot testing matrix -> 05_Analysis
```

## Running the demo apps

From the repository root:

```bash
streamlit run 07_Demo/app.py     # main demo
streamlit run 07_Demo/app2.py    # fairness metrics dashboard
```

The main app:

- loads the stored BERT scores,
- trains a tiny **routing classifier** (`BERT_score → Logistic Regression`,
  labels = VADER-median severity split) plus two Fairlearn mitigators
  (ThresholdOptimizer, Exponentiated Gradient) at startup,
- offers **pre-loaded dataset examples** (chosen because the mitigation
  actually changes their routing decision) **and a custom-input mode** where
  any name + complaint text is scored live with DistilBERT,
- shows live SHAP token attributions and saved LIME explanations,
- includes a Differential-Privacy sweep on the routing model.

`Not Negative` in the UI means *not routed as negative* — it is a routing
label, not a POSITIVE sentiment label.

## Deploying to Streamlit Community Cloud

This project is deployed at
**https://sentimental-analysis-in-ai-systems.streamlit.app** (entrypoint
`07_Demo/app.py`, Python 3.11). To redeploy from your own fork:

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) → *New app* →
   pick the repo, **`07_Demo/app.py`** as the entrypoint, and **Python 3.11**.
3. No secrets are required. Community Cloud installs dependencies from
   `07_Demo/requirements.txt` (searched in the entrypoint directory first).

Notes:

- `07_Demo/requirements.txt` requires `streamlit>=1.49`-era API
  (`width="stretch"`); it is pinned to `streamlit==1.64.0` which is the version
  verified end-to-end. Do **not** downgrade below the `width="stretch"` API.
- The DistilBERT model (~260 MB) downloads from Hugging Face on first run and
  is cached afterwards. First cold start is therefore slower; no token needed
  (a public model — HF may warn about unauthenticated rate limits, which is
  harmless).
- If the app ever needs to run **without** torch/transformers (lighter deploy),
  it degrades gracefully: stored scores still display, custom input falls back
  to VADER, and live SHAP shows an info message. Just delete
  `transformers`, `torch`, `shap` and the `--extra-index-url` line from
  `07_Demo/requirements.txt`.

## Results

All numbers below come from the regenerated artifacts in `04_Results/` and
`02_Data/sentiment_scores_all_systems.csv` (scored with the real DistilBERT
model — earlier commits of this repo contained *placeholder* BERT scores
produced by the `--skip-bert` fallback; those artifacts have been regenerated).

**Bias detection (real model scores)**

- VADER and TextBlob produce **byte-identical** scores across demographic
  groups — expected, since they never see the name semantics. Their measured
  bias is exactly 0.
- DistilBERT shows small group differences (largest ≈ +0.020), but **no
  statistically significant race effect** (ANOVA *p* ≈ 0.066).
- **RoBERTa (Twitter-trained) shows the largest name effects**: Indian_Male
  names score ≈ +0.051 *less negative* than White_Male names for identical
  complaint text (≈ +7.2% relative bias; Chinese_Female ≈ +0.043, Indian_Female
  ≈ +0.035). Per-group t-tests remain non-significant at *n* = 100, but the
  group-mean ordering is consistent and visible in the app's name-swap view.

**Fairness metrics (step 4)**

- VADER/TextBlob pass all fairness checks trivially (no group variance).
- For the routing-severity task, BERT's group-conditioned equal-opportunity /
  equalized-odds differences are ~0.16, and **RoBERTa's are ~0.25** (above the
  0.10 threshold). The disparate-impact ratio is degenerate for both
  transformers because almost no rows receive a positive routing label — it is
  reported as N/A rather than a violation.

**Mitigation (steps 5 / 5b)**

- On the TF-IDF + LR task the dataset is near-perfectly separable (baseline
  accuracy 100%). CDA does not change predictions; Exponentiated Gradient and
  ThresholdOptimizer *slightly worsen* the demographic-parity difference on
  this dataset (0.119 → 0.153 / 0.136). The report prints this honestly
  ("HONEST RESULT: no mitigation method improved...").
- Mitigation still changes **per-example routing decisions** — the demo
  pre-loads examples whose decisions actually flip.

**Privacy (step 7)**

- Output-perturbation noise has no measurable effect at any tested ε on this
  separable dataset (accuracy 100% at every ε).
- **Opacus DP-SGD** does show a real tradeoff: ε = 1.0 → accuracy ≈ 62.5%,
  demographic-parity difference ≈ 0.081.

## Limitations

- **Synthetic dataset.** 20 templates × 40 names. Real complaints are far
  noisier; conclusions about "no significant bias" are specific to this setup.
- **Proxy labels.** There is no human-labelled ground truth; the routing
  target is a VADER-median severity split (the project consistently uses this
  proxy for the severity task).
- **Effect sizes.** VADER/TextBlob are name-blind by construction; DistilBERT's
  name effect is small and insignificant; RoBERTa's is the largest measured
  (group means differ by up to ≈ 0.05) but individual t-tests are still
  non-significant at *n* = 100 per group. The project is best read as an *audit
  framework* showing that *which* system you deploy changes the conclusion.
- **Degenerate metrics.** With near-zero positive routing labels, disparate
  impact and the ε-sweep are technically valid but not informative.

## Future improvements

- Larger, more diverse complaint corpora (and real labelled data).
- Per-group calibrated thresholds learned via proper cross-validation.
- Additional sensitive attributes (e.g., dialectal text features rather than names).
- Persist mitigator artifacts and load them in the app instead of refitting.
- Continuous integration to regenerate `04_Results/` on each change.
