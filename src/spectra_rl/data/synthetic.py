"""Deterministic Raman-like synthetic data."""
from __future__ import annotations

import numpy as np


def make_synthetic(
    n_classes: int = 6, n_per_class: int = 80, n_points: int = 256, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate spectra, integer labels and two-domain instrument identifiers."""
    rng = np.random.default_rng(seed)
    x = np.linspace(200, 1800, n_points)
    spectra, labels, domains = [], [], []
    centers = np.linspace(350, 1600, 5)
    for cls in range(n_classes):
        for i in range(n_per_class):
            domain = int(i >= n_per_class * 0.6)
            shift = 0.0 if domain == 0 else 8.0
            gain = 1.0 if domain == 0 else 1.08
            y = 0.0000008 * (x - 900) ** 2 + 0.12 * np.exp(-(x - 250) / 700)
            for j, center in enumerate(centers):
                width = 18 + 4 * (j % 3)
                amp = (0.25 + 0.06 * cls) * (1 + 0.1 * np.sin(cls + j))
                c = center + cls * 11 + shift
                y += amp * np.exp(-0.5 * ((x - c) / width) ** 2)
                y += 0.08 * amp / (1 + ((x - c) / (width * 1.5)) ** 2)
            y *= gain
            y += rng.normal(0, 0.018, n_points)
            spectra.append(y.astype(np.float32))
            labels.append(cls)
            domains.append(domain)
    return np.asarray(spectra), np.asarray(labels), np.asarray(domains)

