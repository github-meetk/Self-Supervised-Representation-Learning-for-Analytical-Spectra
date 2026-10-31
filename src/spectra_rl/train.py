"""Autoencoder training and checkpoint utilities."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data.loader import load_dataset
from .models.cnn1d import ConvAutoencoder
from .preprocessing import Pipeline


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_autoencoder(config: dict) -> Path:
    """Train an unsupervised autoencoder and save its checkpoint."""
    seed = int(config["data"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, labels, _, source = load_dataset(**config["data"])
    pipe = Pipeline(**config["preprocessing"])
    X = pipe.fit_transform(X)
    model_cfg = config["model"]
    model = ConvAutoencoder(X.shape[1], model_cfg["embedding_dim"]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=model_cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    best, stale, best_state = float("inf"), 0, None
    for _ in range(model_cfg["epochs"]):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        value = float(np.mean(losses))
        scheduler.step(value)
        if value < best:
            best, stale, best_state = value, 0, {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale > model_cfg["patience"]:
                break
    out = Path(config["evaluation"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "autoencoder.pt"
    model.load_state_dict(best_state)
    with torch.no_grad():
        embeddings = model.encode(torch.from_numpy(X)).numpy()
    downstream = LogisticRegression(max_iter=300).fit(embeddings, labels)
    torch.save({"state_dict": model.state_dict(), "n_points": X.shape[1],
                "embedding_dim": model_cfg["embedding_dim"], "source": source,
                "preprocessing": config["preprocessing"], "classifier_coef": downstream.coef_,
                "classifier_intercept": downstream.intercept_, "classifier_classes": downstream.classes_}, checkpoint)
    print(f"Trained autoencoder on {source} data using {device}; checkpoint: {checkpoint}")
    return checkpoint


def load_encoder(checkpoint: str | Path, device: str | None = None) -> ConvAutoencoder:
    """Load a saved autoencoder encoder."""
    payload = torch.load(checkpoint, map_location=device or "cpu", weights_only=False)
    model = ConvAutoencoder(payload["n_points"], payload["embedding_dim"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
