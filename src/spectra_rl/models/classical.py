"""Classical chemometric baselines."""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PCAForest:
    """PCA followed by a random forest classifier."""
    def __init__(self, n_components: int = 20, seed: int = 42) -> None:
        self.n_components, self.seed = n_components, seed
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> PCAForest:
        components = min(self.n_components, X.shape[0], X.shape[1])
        self.model = make_pipeline(PCA(n_components=components), RandomForestClassifier(
            n_estimators=80, random_state=self.seed, n_jobs=1))
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class PLSDA:
    """PLS regression against one-hot labels with argmax decoding."""
    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> PLSDA:
        self.classes_ = np.unique(y)
        Y = np.eye(len(self.classes_))[y]
        self.model = make_pipeline(StandardScaler(), PLSRegression(n_components=min(self.n_components, X.shape[1], len(self.classes_))))
        self.model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.asarray(self.model.predict(X)).argmax(axis=1)]
