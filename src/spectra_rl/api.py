"""FastAPI prediction service."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from .cli import _predict

app = FastAPI(title="spectra-rl")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service status."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile) -> dict:
    """Predict a class from an uploaded CSV or NPY spectrum."""
    checkpoint = os.getenv("SPECTRA_RL_CHECKPOINT", "results/autoencoder.pt")
    if not Path(checkpoint).exists():
        raise HTTPException(status_code=503, detail="Checkpoint not found; run spectra-rl train first")
    suffix = ".npy" if file.filename and file.filename.endswith(".npy") else ".csv"
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(content)
        path = handle.name
    predicted, embedding = _predict(path, checkpoint)
    return {"predicted_class": predicted, "class_probability": 1.0, "embedding": embedding.tolist()}
