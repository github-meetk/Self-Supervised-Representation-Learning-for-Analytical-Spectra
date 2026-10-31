"""Compact one-dimensional CNN classifier."""
from __future__ import annotations

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    """CNN classifier returning logits and optionally embeddings."""

    def __init__(self, n_points: int, n_classes: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.embedding = nn.Linear(32 * 8, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        z = self.encode(x)
        return self.classifier(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding vectors."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.embedding(self.features(x).flatten(1))


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder with configurable latent representation."""

    def __init__(self, n_points: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.to_embedding = nn.Linear(32 * 8, embedding_dim)
        self.from_embedding = nn.Linear(embedding_dim, 32 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )
        self.n_points = n_points

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spectra into latent vectors."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.to_embedding(self.encoder(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct spectra."""
        z = self.from_embedding(self.encode(x)).view(-1, 32, 8)
        return torch.nn.functional.interpolate(self.decoder(z), size=self.n_points, mode="linear", align_corners=False).squeeze(1)
