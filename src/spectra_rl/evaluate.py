"""Benchmark runner for neural and classical methods."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .data.loader import load_dataset
from .data.splits import stratified_split, subsample_labels
from .models.classical import PLSDA, PCAForest
from .preprocessing import Pipeline
from .train import load_encoder, seed_everything, train_autoencoder


def _metric(y, pred) -> tuple[float, float]:
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))


def _confidence(group: pd.DataFrame, column: str) -> tuple[float, float]:
    """Return mean and 95% t-interval half-width."""
    n = len(group)
    mean, se = group[column].mean(), group[column].std(ddof=1) / np.sqrt(n)
    return float(mean), float(t.ppf(0.975, max(1, n - 1)) * se if n > 1 else 0)


def run_evaluation(config: dict) -> Path:
    """Run reproducible label, noise and transfer benchmarks and write CSV/JSON."""
    out = Path(config["evaluation"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "autoencoder.pt"
    if not checkpoint.exists():
        train_autoencoder(config)
    X, y, domains, source = load_dataset(**config["data"])
    X = Pipeline(**config["preprocessing"]).fit_transform(X)
    train_idx, test_idx = stratified_split(y, domains, config["data"]["seed"])
    model = load_encoder(checkpoint)
    with torch.no_grad():
        Z = model.encode(torch.from_numpy(X)).numpy()
    np.savez(out / "embeddings.npz", embeddings=Z, labels=y, source=source)
    rows = []
    for seed in range(config["evaluation"]["seeds"]):
        seed_everything(seed)
        for fraction in config["evaluation"]["label_fractions"]:
            labelled = subsample_labels(y, train_idx, fraction, seed)
            for name, features in [("autoencoder+logreg", Z), ("pca+rf", X), ("pls-da", X)]:
                estimator = LogisticRegression(max_iter=300, random_state=seed) if name.startswith("auto") else (PCAForest(seed=seed) if name == "pca+rf" else PLSDA())
                estimator.fit(features[labelled], y[labelled])
                pred = estimator.predict(features[test_idx])
                acc, f1 = _metric(y[test_idx], pred)
                rows.append({"source": source, "method": name, "condition": "label_efficiency",
                             "level": float(fraction), "seed": seed, "accuracy": acc, "macro_f1": f1})
        for snr in config["evaluation"]["noise_snr_db"]:
            rng = np.random.default_rng(seed)
            signal = np.std(X[test_idx])
            noisy = X[test_idx] + rng.normal(0, signal / (10 ** (snr / 20)), X[test_idx].shape)
            est = LogisticRegression(max_iter=300, random_state=seed).fit(Z[train_idx], y[train_idx])
            pred = est.predict(model.encode(torch.from_numpy(noisy.astype(np.float32))).detach().numpy())
            acc, f1 = _metric(y[test_idx], pred)
            rows.append({"source": source, "method": "autoencoder+logreg", "condition": "noise",
                         "level": float(snr), "seed": seed, "accuracy": acc, "macro_f1": f1})
        est = LogisticRegression(max_iter=300, random_state=seed).fit(Z[train_idx], y[train_idx])
        acc, f1 = _metric(y[test_idx], est.predict(Z[test_idx]))
        rows.append({"source": source, "method": "autoencoder+logreg", "condition": "transfer",
                     "level": 1.0, "seed": seed, "accuracy": acc, "macro_f1": f1})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "results.csv", index=False)
    summaries = []
    for keys, group in frame.groupby(["source", "method", "condition", "level"]):
        n = len(group)
        a, aci = _confidence(group, "accuracy"); f, fci = _confidence(group, "macro_f1")
        summaries.append(dict(zip(["source", "method", "condition", "level"], keys),
                              accuracy_mean=a, accuracy_ci95=aci, macro_f1_mean=f, macro_f1_ci95=fci, n=n))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    return out / "results.csv"
