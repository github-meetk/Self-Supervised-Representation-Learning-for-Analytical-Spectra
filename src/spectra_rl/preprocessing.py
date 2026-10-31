"""Pure preprocessing functions for Raman spectra."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def baseline_correct(X: np.ndarray, degree: int = 3) -> np.ndarray:
    """Subtract a polynomial baseline from each spectrum."""
    X = np.asarray(X, dtype=float)
    grid = np.linspace(-1, 1, X.shape[1])
    out = np.empty_like(X)
    for i, row in enumerate(X):
        coef = np.polyfit(grid, row, degree)
        out[i] = row - np.polyval(coef, grid)
    return out


def smooth(X: np.ndarray, window_length: int = 9, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along the spectral axis."""
    X = np.asarray(X, dtype=float)
    window_length = int(window_length)
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, X.shape[1] if X.shape[1] % 2 else X.shape[1] - 1)
    return savgol_filter(X, window_length, min(polyorder, window_length - 1), axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    """Standard-normal-variate normalize each spectrum independently."""
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


def minmax(X: np.ndarray) -> np.ndarray:
    """Scale each spectrum to [0, 1]."""
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    return (X - lo) / (hi - lo + 1e-8)


class Pipeline:
    """Reusable preprocessing pipeline with training-only metadata."""

    def __init__(self, baseline: bool = True, smoothing: bool = True, normalization: str = "snv",
                 window_length: int = 9, polyorder: int = 2) -> None:
        self.baseline, self.smoothing, self.normalization = baseline, smoothing, normalization
        self.window_length, self.polyorder = window_length, polyorder

    def fit(self, X: np.ndarray) -> Pipeline:
        """Fit pipeline metadata on training data and return self."""
        np.asarray(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform spectra without mutating input."""
        out = np.asarray(X, dtype=float).copy()
        if self.baseline:
            out = baseline_correct(out)
        if self.smoothing:
            out = smooth(out, self.window_length, self.polyorder)
        if self.normalization == "snv":
            out = snv(out)
        elif self.normalization == "minmax":
            out = minmax(out)
        if not np.isfinite(out).all():
            raise ValueError("Preprocessing produced non-finite values")
        return out.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform training spectra."""
        return self.fit(X).transform(X)
