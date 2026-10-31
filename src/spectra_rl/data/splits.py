"""Leak-free stratified splitting and labelled subset selection."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(
    labels: np.ndarray, domains: np.ndarray, seed: int = 42, train_fraction: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so training is reference domain and test is transfer domain when possible."""
    idx = np.arange(len(labels))
    if len(np.unique(domains)) > 1:
        train = idx[domains == np.min(domains)]
        test = idx[domains != np.min(domains)]
        if len(test) and len(train):
            return train, test
    train, test = train_test_split(idx, train_size=train_fraction, stratify=labels, random_state=seed)
    return np.asarray(train), np.asarray(test)


def subsample_labels(labels: np.ndarray, indices: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Select a stratified labelled subset, retaining at least one item per class."""
    if fraction >= 1:
        return np.asarray(indices)
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(labels[indices]):
        group = indices[labels[indices] == cls]
        count = max(1, int(np.ceil(len(group) * fraction)))
        selected.extend(rng.choice(group, size=min(count, len(group)), replace=False))
    return np.asarray(selected, dtype=int)
