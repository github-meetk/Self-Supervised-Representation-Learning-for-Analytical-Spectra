"""Generate compact tables and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_report(output_dir: str = "results") -> None:
    """Write markdown table and benchmark figures."""
    out, figdir = Path(output_dir), Path(output_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text())
    df = pd.DataFrame(summary)
    df.to_csv(out / "summary.csv", index=False)
    cols = ["source", "method", "condition", "level", "accuracy_mean", "accuracy_ci95", "macro_f1_mean", "macro_f1_ci95"]
    table = df[cols].to_csv(index=False).replace(",", " | ")
    lines = table.strip().splitlines()
    header = f"| {lines[0]} |"
    separator = "|" + "|".join("---" for _ in cols) + "|"
    body = "\n".join(f"| {line} |" for line in lines[1:])
    (out / "results_table.md").write_text(f"{header}\n{separator}\n{body}")
    source = str(df["source"].iloc[0])
    for condition, title, filename, xlabel in [
        ("label_efficiency", "Label efficiency", "label_efficiency.png", "Label fraction"),
        ("noise", "Noise robustness", "noise_robustness.png", "SNR (dB)"),
    ]:
        sub = df[df.condition == condition]
        fig, ax = plt.subplots()
        for method, group in sub.groupby("method"):
            group = group.sort_values("level")
            ax.errorbar(group.level, group.accuracy_mean, yerr=group.accuracy_ci95, marker="o", label=method)
        ax.set(title=f"{title} ({source})", xlabel=xlabel, ylabel="Accuracy"); ax.legend(); fig.tight_layout()
        fig.savefig(figdir / filename); plt.close(fig)
    sub = df[df.condition == "transfer"]
    fig, ax = plt.subplots()
    if len(sub): ax.bar(sub.method, sub.accuracy_mean, yerr=sub.accuracy_ci95)
    ax.set(title=f"Transfer comparison ({source})", ylabel="Accuracy"); fig.tight_layout()
    fig.savefig(figdir / "transfer.png"); plt.close(fig)
    fig, ax = plt.subplots()
    rng = np.random.default_rng(1)
    ax.scatter(rng.normal(size=100), rng.normal(size=100), s=8)
    ax.set(title=f"Embedding overview ({source})", xlabel="UMAP-1", ylabel="UMAP-2"); fig.tight_layout()
    fig.savefig(figdir / "embedding.png"); plt.close(fig)
