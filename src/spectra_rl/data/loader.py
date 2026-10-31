"""Real and synthetic dataset loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .synthetic import make_synthetic


def load_dataset(
    source: str = "auto",
    raw_dir: str | Path = "data/raw",
    n_classes: int = 6,
    n_per_class: int = 80,
    n_points: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load documented arrays or return the labelled synthetic fallback."""
    path = Path(raw_dir)
    spectra_file, labels_file = path / "spectra.npy", path / "labels.npy"
    if source in {"auto", "real"} and spectra_file.exists() and labels_file.exists():
        spectra = np.load(spectra_file)
        labels = np.load(labels_file)
        domains = np.load(path / "domains.npy") if (path / "domains.npy").exists() else np.zeros(len(labels), int)
        return spectra.astype(np.float32), labels.astype(int), domains.astype(int), "real"
    if source == "real":
        print("Real data not found in data/raw; continuing with synthetic data.")
    return (*make_synthetic(n_classes, n_per_class, n_points, seed), "synthetic")

