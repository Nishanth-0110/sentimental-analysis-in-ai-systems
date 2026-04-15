"""
==============================================================================
Step 2: Sentiment Analysis with VADER, TextBlob, and BERT
==============================================================================
Runs all 800 sentences through three sentiment analysis systems and records
scores for each. This produces the raw data for bias detection.
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# 1. LOAD DATASET
# =============================================================================

def load_dataset(data_path=None):
    """Load the 800-sentence complaint dataset."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "02_Data",
            "complaint_dataset_800.csv"
        )
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} sentences")
    return df


# =============================================================================
# 2. VADER SENTIMENT ANALYSIS
# =============================================================================

def analyze_vader(texts):
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner)
    - Rule-based, dictionary approach
    - Returns compound score from -1 (most negative) to +1 (most positive)
    - Used by Twitter, social media companies
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in tqdm(texts, desc="VADER Analysis"):
        vs = analyzer.polarity_scores(text)
        scores.append({
            "VADER_compound": vs["compound"],
            "VADER_positive": vs["pos"],
            "VADER_neutral":  vs["neu"],
            "VADER_negative": vs["neg"],
        })
    return pd.DataFrame(scores)


# =============================================================================
# 3. TEXTBLOB SENTIMENT ANALYSIS
# =============================================================================

def analyze_textblob(texts):
    """
    TextBlob sentiment analysis.
    - ML + rule-based hybrid
    - Returns polarity (-1 to +1) and subjectivity (0 to 1)
    - Used by small businesses, startups
    """
    from textblob import TextBlob

    scores = []
    for text in tqdm(texts, desc="TextBlob Analysis"):
        blob = TextBlob(text)
        scores.append({
            "TextBlob_polarity":     blob.sentiment.polarity,
            "TextBlob_subjectivity": blob.sentiment.subjectivity,
        })
    return pd.DataFrame(scores)


# =============================================================================
# 4. BERT SENTIMENT ANALYSIS (HuggingFace Transformers)
# =============================================================================

def analyze_bert(texts, batch_size=16):
    """
    BERT-based sentiment analysis using HuggingFace pipeline.
    - Deep learning, state-of-the-art
    - Uses distilbert-base-uncased-finetuned-sst-2-english
    - Returns label (POSITIVE/NEGATIVE) and confidence score
    - Converted to -1 to +1 scale for comparison
    """
    from transformers import pipeline

    print("Loading BERT model (this may take a minute on first run)...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # CPU; change to 0 for GPU
    )

    scores = []
    # Process in batches for efficiency
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Analysis"):
        batch = texts[i:i + batch_size]
        results = sentiment_pipeline(batch, truncation=True, max_length=512)
        for result in results:
            label = result["label"]
            confidence = result["score"]
            # Convert to -1 to +1 scale
            if label == "NEGATIVE":
                bert_score = -confidence
            else:
                bert_score = confidence
            scores.append({
                "BERT_score":      bert_score,
                "BERT_label":      label,
                "BERT_confidence": confidence,
            })

    return pd.DataFrame(scores)


# =============================================================================
# 5. RUN ALL ANALYSES
# =============================================================================

def run_all_sentiment_analyses(df):
    """Run VADER, TextBlob, and BERT on all sentences."""
    texts = df["Full_Text"].tolist()

    print("\n" + "=" * 60)
    print("RUNNING SENTIMENT ANALYSIS ON 800 SENTENCES")
    print("=" * 60)

    # VADER
    print("\n--- VADER ---")
    vader_scores = analyze_vader(texts)

    # TextBlob
    print("\n--- TextBlob ---")
    textblob_scores = analyze_textblob(texts)

    # BERT
    print("\n--- BERT ---")
    bert_scores = analyze_bert(texts)

    # Combine all results
    result_df = pd.concat([
        df.reset_index(drop=True),
        vader_scores.reset_index(drop=True),
        textblob_scores.reset_index(drop=True),
        bert_scores.reset_index(drop=True),
    ], axis=1)

    return result_df


# =============================================================================
# 6. INITIAL EXPLORATION
# =============================================================================

def initial_exploration(df):
    """Calculate basic statistics and show initial bias signals."""
    print("\n" + "=" * 60)
    print("INITIAL DATA EXPLORATION")
    print("=" * 60)

    # Average scores by demographic group
    print("\n1. AVERAGE SENTIMENT SCORES BY DEMOGRAPHIC GROUP:")
    print("-" * 60)

    for system, col in [("VADER", "VADER_compound"),
                        ("TextBlob", "TextBlob_polarity"),
                        ("BERT", "BERT_score")]:
        print(f"\n  {system}:")
        group_means = df.groupby("Demographic_Group")[col].mean().sort_values()
        for group, score in group_means.items():
            print(f"    {group:20s}: {score:+.4f}")

    # Average by race
    print("\n2. AVERAGE SCORES BY RACE:")
    print("-" * 60)
    for system, col in [("VADER", "VADER_compound"),
                        ("TextBlob", "TextBlob_polarity"),
                        ("BERT", "BERT_score")]:
        print(f"\n  {system}:")
        race_means = df.groupby("Race")[col].mean().sort_values()
        for race, score in race_means.items():
            print(f"    {race:10s}: {score:+.4f}")

    # Average by gender
    print("\n3. AVERAGE SCORES BY GENDER:")
    print("-" * 60)
    for system, col in [("VADER", "VADER_compound"),
                        ("TextBlob", "TextBlob_polarity"),
                        ("BERT", "BERT_score")]:
        print(f"\n  {system}:")
        gender_means = df.groupby("Gender")[col].mean().sort_values()
        for gender, score in gender_means.items():
            print(f"    {gender:10s}: {score:+.4f}")

    # Calculate bias (difference from White Male baseline)
    print("\n4. BIAS RELATIVE TO WHITE MALE BASELINE:")
    print("-" * 60)
    baseline_group = "White_Male"

    for system, col in [("VADER", "VADER_compound"),
                        ("TextBlob", "TextBlob_polarity"),
                        ("BERT", "BERT_score")]:
        print(f"\n  {system}:")
        group_means = df.groupby("Demographic_Group")[col].mean()
        baseline = group_means.get(baseline_group, 0)
        for group in sorted(group_means.index):
            bias = group_means[group] - baseline
            print(f"    {group:20s}: bias = {bias:+.4f}"
                  f"{'  ⚠ SIGNIFICANT' if abs(bias) > 0.05 else ''}")


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load dataset
    df = load_dataset()

    # Run all analyses
    results = run_all_sentiment_analyses(df)

    # Initial exploration
    initial_exploration(results)

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "02_Data"
    )
    output_path = os.path.join(output_dir, "sentiment_scores_all_systems.csv")
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Shape: {results.shape}")
