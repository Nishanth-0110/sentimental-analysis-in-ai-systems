# Auditing Gender & Race Bias in Customer Service AI

A comprehensive study examining bias in AI-powered customer service systems across gender and racial demographics, implementing fairness metrics, bias mitigation, explainability, and differential privacy.

## Project Structure

```
RESAI/
├── 01_Research/           # Literature review and references
├── 02_Data/               # Generated datasets and scored data
├── 03_Code/               # All analysis scripts
│   ├── 01_dataset_generation.py      # Generate 800-sentence dataset
│   ├── 02_sentiment_analysis.py      # VADER, TextBlob, BERT analysis
│   ├── 03_bias_detection.py          # Statistical bias testing
│   ├── 04_fairness_metrics.py        # 5 fairness metrics (Fairlearn)
│   ├── 05_bias_mitigation.py         # 3 mitigation techniques
│   ├── 06_explainability.py          # SHAP + LIME analysis
│   ├── 07_privacy.py                 # Differential privacy (Opacus)
│   ├── 08_visualizations.py          # 10 professional charts
│   ├── run_all.py                    # Master pipeline runner
│   └── generate_testing_template.py  # ChatGPT testing template
├── 04_Results/            # Analysis outputs, charts, CSV results
├── 05_Analysis/           # Chatbot testing data
├── 06_Report/             # Final report and presentation
├── 07_Demo/               # Streamlit interactive demo
│   ├── app.py
│   └── models/            # Saved models from mitigation
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
cd 03_Code
python run_all.py
```

This runs all 8 steps sequentially:
1. **Dataset Generation** — Creates 800 controlled complaint sentences (20 templates × 8 demographic groups × 5 names)
2. **Sentiment Analysis** — Runs VADER, TextBlob, and BERT on all sentences
3. **Bias Detection** — Statistical tests (t-tests, ANOVA, Cohen's d, Mann-Whitney U)
4. **Fairness Metrics** — 5 metrics via Microsoft Fairlearn
5. **Bias Mitigation** — Reweighing, Exponentiated Gradient, Threshold Optimizer
6. **Explainability** — SHAP (global) and LIME (local) explanations
7. **Differential Privacy** — Epsilon sensitivity analysis, privacy-fairness-accuracy tradeoff
8. **Visualizations** — 10 publication-ready charts

### Run a Single Step

```bash
python run_all.py --step 1        # Only generate dataset
python run_all.py --step 2        # Only run sentiment analysis
python run_all.py --skip-bert     # Run all but skip BERT (faster)
```

### Run Individual Scripts

Each script can also run independently:

```bash
python 01_dataset_generation.py
python 02_sentiment_analysis.py
python 03_bias_detection.py
# ... etc
```

### 3. Launch Interactive Demo

```bash
cd 07_Demo
streamlit run app.py
```

### 4. Generate ChatGPT Testing Template

```bash
cd 03_Code
python generate_testing_template.py
```

This creates `05_Analysis/chatbot_testing_template.csv` with 80 test scenarios (10 complaints × 8 demographic names) for manual ChatGPT testing.

## Technical Components

### Sentiment Analysis Systems
| System | Type | Score Range | Key Metric |
|--------|------|-------------|------------|
| VADER | Rule-based dictionary | -1 to +1 | Compound score |
| TextBlob | ML + rules hybrid | -1 to +1 | Polarity |
| BERT | Deep learning (DistilBERT) | -1 to +1 | Confidence-weighted |

### Fairness Metrics (via Fairlearn)
1. **Demographic Parity Difference** — Max selection rate gap across groups (threshold: < 0.10)
2. **Equal Opportunity Difference** — TPR gap for positive class (threshold: < 0.10)
3. **Equalized Odds Difference** — Combined TPR + FPR gap (threshold: < 0.10)
4. **Disparate Impact Ratio** — Min/max selection rate ratio (threshold: > 0.80)
5. **Calibration Difference** — Score calibration gap across groups (threshold: < 0.10)

### Bias Mitigation Techniques
1. **Reweighing** (Pre-processing) — Adjusts training sample weights to balance demographic representation
2. **Exponentiated Gradient** (In-processing) — Constrained optimization with DemographicParity constraint
3. **Threshold Optimizer** (Post-processing) — Adjusts decision thresholds per group for equalized odds

### Explainability
- **SHAP** (SHapley Additive exPlanations) — Global feature importance using LinearExplainer
- **LIME** (Local Interpretable Model-agnostic Explanations) — Local per-instance explanations with counterfactual analysis

### Differential Privacy
- **Laplace Noise Injection** — Output perturbation with configurable epsilon
- **DP-SGD via Opacus** — Differentially private stochastic gradient descent (Meta's framework)
- **Three-Way Tradeoff Analysis** — Privacy vs. Fairness vs. Accuracy across 4 configurations

## Dataset Design

- **20 complaint templates** across 4 emotion categories (angry, frustrated, disappointed, demanding)
- **8 demographic groups**: Indian Male/Female, White Male/Female, Black Male/Female, Chinese Male/Female
- **5 culturally representative names per group** (40 names total)
- **Total: 800 sentences** with controlled variation

## Key Dependencies

- `fairlearn` — Microsoft's fairness toolkit
- `transformers` — HuggingFace (DistilBERT)
- `shap` — Explainability
- `lime` — Local explanations
- `opacus` — Meta's differential privacy
- `streamlit` — Interactive demo
- `vaderSentiment` — Rule-based sentiment
- `textblob` — ML sentiment

## Output Files

After running the pipeline, find results in `04_Results/`:
- `bias_by_demographic_*.csv` — Bias scores per demographic group
- `statistical_tests_*.csv` — t-test, Cohen's d, Mann-Whitney U results
- `fairness_metrics_all_systems.csv` — 5 fairness metrics for all 3 systems
- `mitigation_comparison.csv` — Before/after mitigation comparison
- `privacy_epsilon_analysis.csv` — Epsilon sensitivity results
- `three_way_tradeoff.csv` — Privacy-fairness-accuracy tradeoff
- `shap_*.png`, `lime_*.png` — Explainability visualizations
- `chart_*.png` — 10 publication-ready charts
