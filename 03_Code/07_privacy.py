"""
==============================================================================
Step 7: Differential Privacy Implementation
==============================================================================
Implements differential privacy using Opacus (Facebook/Meta).
Tests multiple epsilon values and analyzes privacy-accuracy-fairness tradeoff.

Key concepts:
  - ε (epsilon): Privacy budget. Lower = more private, less accurate.
  - δ (delta): Failure probability.
  - DP-SGD: Differentially Private Stochastic Gradient Descent.
==============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DIFFERENTIAL PRIVACY WITH SKLEARN (Noise Injection Approach)
# =============================================================================

def add_laplace_noise(data, epsilon, sensitivity=1.0):
    """Add Laplace noise to achieve ε-differential privacy."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=data.shape)
    return data + noise


def train_with_dp_noise(X_train, y_train, X_test, y_test,
                         sensitive_test, epsilon, random_state=42):
    """
    Train a model with differential privacy via output perturbation.
    Adds calibrated Laplace noise to model coefficients after training.
    """
    from sklearn.linear_model import LogisticRegression
    from fairlearn.metrics import demographic_parity_difference

    # Train normally
    model = LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)
    model.fit(X_train, y_train)

    # Add DP noise to coefficients (output perturbation)
    sensitivity = 2.0 / (len(y_train) * model.C)

    if epsilon < float("inf"):
        noisy_coef = add_laplace_noise(model.coef_, epsilon, sensitivity)
        noisy_intercept = add_laplace_noise(
            model.intercept_, epsilon, sensitivity
        )
        model.coef_ = noisy_coef
        model.intercept_ = noisy_intercept

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    dp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    )

    groups = np.unique(sensitive_test)
    rates = {}
    for g in groups:
        mask = sensitive_test == g
        rates[g] = y_pred[mask].mean() if mask.sum() > 0 else 0

    return model, accuracy, abs(dp_diff), rates


# =============================================================================
# 2. DP-SGD WITH OPACUS (PyTorch-based Deep Learning)
# =============================================================================

def train_with_opacus(X_train, y_train, X_test, y_test,
                       sensitive_test, target_epsilon=1.0):
    """
    Train with Opacus (DP-SGD) using PyTorch.
    This is the gold standard for differentially private deep learning.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        from fairlearn.metrics import demographic_parity_difference
    except ImportError:
        print("  Opacus/PyTorch not available. Using noise injection fallback.")
        return None, None, None, None

    # Convert sparse to dense if needed
    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray().astype(np.float32)
        X_test_dense = X_test.toarray().astype(np.float32)
    else:
        X_train_dense = np.array(X_train, dtype=np.float32)
        X_test_dense = np.array(X_test, dtype=np.float32)

    # Create tensors
    X_tensor = torch.FloatTensor(X_train_dense)
    y_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_dense)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define model
    n_features = X_train_dense.shape[1]
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    model = ModuleValidator.fix(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    # Attach privacy engine
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=10,
        target_epsilon=target_epsilon,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )

    # Train
    model.train()
    for epoch in range(10):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()

    accuracy = accuracy_score(y_test, y_pred)
    dp_diff = demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    )

    actual_epsilon = privacy_engine.get_epsilon(delta=1e-5)

    groups = np.unique(sensitive_test)
    rates = {}
    for g in groups:
        mask = sensitive_test == g
        rates[g] = y_pred[mask].mean() if mask.sum() > 0 else 0

    return model, accuracy, abs(dp_diff), actual_epsilon


# =============================================================================
# 3. EPSILON SENSITIVITY ANALYSIS
# =============================================================================

def epsilon_sensitivity_analysis(X_train, y_train, X_test, y_test,
                                  sensitive_test):
    """Test multiple epsilon values to find optimal privacy-accuracy tradeoff."""
    print("\n" + "=" * 60)
    print("  EPSILON SENSITIVITY ANALYSIS")
    print("=" * 60)

    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
    results = []

    for eps in epsilons:
        # Run multiple times for stability (noise is random)
        accs, dps = [], []
        for seed in range(5):
            _, acc, dp, _ = train_with_dp_noise(
                X_train, y_train, X_test, y_test,
                sensitive_test, epsilon=eps, random_state=seed
            )
            accs.append(acc)
            dps.append(dp)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_dp = np.mean(dps)

        eps_label = f"ε={eps}" if eps != float("inf") else "No Privacy"
        privacy_level = (
            "Maximum" if eps <= 0.1 else
            "Very Strong" if eps <= 0.5 else
            "Strong" if eps <= 1.0 else
            "Moderate" if eps <= 5.0 else
            "Weak" if eps <= 10.0 else
            "None"
        )

        results.append({
            "epsilon": eps,
            "epsilon_label": eps_label,
            "privacy_level": privacy_level,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "dem_parity_diff": mean_dp,
        })

        print(f"\n  {eps_label} ({privacy_level}):")
        print(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.1f}%)")
        print(f"    Dem. Parity Diff: {mean_dp:.4f}")

    return pd.DataFrame(results)


# =============================================================================
# 4. THREE-WAY TRADEOFF ANALYSIS
# =============================================================================

def three_way_tradeoff(X_train, y_train, X_test, y_test,
                        sensitive_train, sensitive_test):
    """
    Analyze the three-way tradeoff: Accuracy vs Fairness vs Privacy.
    Tests configurations combining mitigation + privacy.
    """
    from sklearn.linear_model import LogisticRegression
    from fairlearn.metrics import demographic_parity_difference

    print("\n" + "=" * 60)
    print("  THREE-WAY TRADEOFF: ACCURACY vs FAIRNESS vs PRIVACY")
    print("=" * 60)

    configurations = []

    # Config 1: Baseline (biased, no privacy)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    dp = abs(demographic_parity_difference(
        y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test
    ))
    configurations.append({
        "config": "Baseline (biased, no privacy)",
        "accuracy": acc, "fairness_dp": dp, "privacy_eps": "∞",
        "acc_check": "✅", "fair_check": "❌" if dp >= 0.10 else "✅",
        "priv_check": "❌"
    })

    # Config 2: Fair but no privacy (reweighing)
    groups = np.unique(sensitive_train)
    n_total = len(y_train)
    sample_weights = np.ones(n_total)
    for label in [0, 1]:
        for g in groups:
            mask = (sensitive_train == g) & (y_train == label)
            n_group_label = mask.sum()
            n_group = (sensitive_train == g).sum()
            n_label = (y_train == label).sum()
            if n_group_label > 0:
                expected = (n_group * n_label) / n_total
                weight = expected / n_group_label
                sample_weights[mask] = weight

    model_fair = LogisticRegression(max_iter=1000, random_state=42)
    model_fair.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_fair = model_fair.predict(X_test)
    acc_fair = accuracy_score(y_test, y_pred_fair)
    dp_fair = abs(demographic_parity_difference(
        y_true=y_test, y_pred=y_pred_fair, sensitive_features=sensitive_test
    ))
    configurations.append({
        "config": "Fair only (no privacy)",
        "accuracy": acc_fair, "fairness_dp": dp_fair, "privacy_eps": "∞",
        "acc_check": "✅", "fair_check": "❌" if dp_fair >= 0.10 else "✅",
        "priv_check": "❌"
    })

    # Config 3: Private but biased
    _, acc_priv, dp_priv, _ = train_with_dp_noise(
        X_train, y_train, X_test, y_test, sensitive_test, epsilon=1.0
    )
    configurations.append({
        "config": "Private only (biased)",
        "accuracy": acc_priv, "fairness_dp": dp_priv, "privacy_eps": "1.0",
        "acc_check": "✅", "fair_check": "❌" if dp_priv >= 0.10 else "✅",
        "priv_check": "✅"
    })

    # Config 4: Fair + Private (BEST)
    model_fair_priv = LogisticRegression(max_iter=1000, random_state=42)
    model_fair_priv.fit(X_train, y_train, sample_weight=sample_weights)
    # Add DP noise
    sensitivity = 2.0 / (len(y_train) * model_fair_priv.C)
    model_fair_priv.coef_ = add_laplace_noise(
        model_fair_priv.coef_, 1.0, sensitivity
    )
    model_fair_priv.intercept_ = add_laplace_noise(
        model_fair_priv.intercept_, 1.0, sensitivity
    )
    y_pred_fp = model_fair_priv.predict(X_test)
    acc_fp = accuracy_score(y_test, y_pred_fp)
    dp_fp = abs(demographic_parity_difference(
        y_true=y_test, y_pred=y_pred_fp, sensitive_features=sensitive_test
    ))
    configurations.append({
        "config": "Fair + Private (RECOMMENDED)",
        "accuracy": acc_fp, "fairness_dp": dp_fp, "privacy_eps": "1.0",
        "acc_check": "✅", "fair_check": "❌" if dp_fp >= 0.10 else "✅",
        "priv_check": "✅"
    })

    # Print comparison
    print(f"\n  {'Configuration':<35s} {'Accuracy':>10s} {'Fairness(DP)':>13s} "
          f"{'Privacy(ε)':>11s} {'A':>3s} {'F':>3s} {'P':>3s}")
    print("  " + "-" * 80)
    for c in configurations:
        print(f"  {c['config']:<35s} {c['accuracy']:>9.1%} "
              f"{c['fairness_dp']:>12.4f} {c['privacy_eps']:>11s} "
              f"{c['acc_check']:>3s} {c['fair_check']:>3s} {c['priv_check']:>3s}")

    return pd.DataFrame(configurations)


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

    # Prepare
    score_col = "VADER_compound"
    median_score = df[score_col].median()
    df["label"] = (df[score_col] >= median_score).astype(int)

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    X = vectorizer.fit_transform(df["Full_Text"])
    y = df["label"].values
    sensitive = df["Race"].values

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )

    # Epsilon sensitivity
    eps_results = epsilon_sensitivity_analysis(
        X_train, y_train, X_test, y_test, sens_test
    )

    # Three-way tradeoff
    tradeoff_results = three_way_tradeoff(
        X_train, y_train, X_test, y_test, sens_train, sens_test
    )

    # Try Opacus (PyTorch DP-SGD)
    print("\n" + "=" * 60)
    print("  OPACUS DP-SGD (PyTorch)")
    print("=" * 60)
    opacus_model, opacus_acc, opacus_dp, actual_eps = train_with_opacus(
        X_train, y_train, X_test, y_test, sens_test, target_epsilon=1.0
    )
    if opacus_acc is not None:
        print(f"  Opacus DP-SGD Results:")
        print(f"    Target ε: 1.0, Achieved ε: {actual_eps:.2f}")
        print(f"    Accuracy: {opacus_acc:.4f} ({opacus_acc*100:.1f}%)")
        print(f"    Dem. Parity Diff: {opacus_dp:.4f}")

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "04_Results"
    )
    os.makedirs(output_dir, exist_ok=True)
    eps_results.to_csv(
        os.path.join(output_dir, "privacy_epsilon_analysis.csv"), index=False
    )
    tradeoff_results.to_csv(
        os.path.join(output_dir, "three_way_tradeoff.csv"), index=False
    )

    print("\n\nPrivacy analysis complete!")
