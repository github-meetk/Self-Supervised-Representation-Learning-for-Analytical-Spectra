"""Command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .evaluate import run_evaluation
from .preprocessing import Pipeline
from .report import make_report
from .train import load_encoder, train_autoencoder


def _config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _predict(path: str, checkpoint: str) -> tuple[int, np.ndarray]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_encoder(checkpoint)
    X = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",")
    X = np.asarray(X, dtype=np.float32).reshape(1, -1)
    X = Pipeline(**payload["preprocessing"]).fit_transform(X)
    with torch.no_grad():
        embedding = model.encode(torch.from_numpy(X)).numpy()[0]
    if "classifier_coef" in payload:
        logits = payload["classifier_coef"] @ embedding + payload["classifier_intercept"]
        predicted = int(payload["classifier_classes"][np.argmax(logits)])
    else:
        predicted = int(np.argmax(embedding))
    return predicted, embedding


def main() -> None:
    """Run the spectra-rl CLI."""
    parser = argparse.ArgumentParser(prog="spectra-rl")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "report"):
        p = sub.add_parser(command); p.add_argument("--config", default="configs/default.yaml")
    p = sub.add_parser("predict"); p.add_argument("spectrum"); p.add_argument("--checkpoint", default="results/autoencoder.pt")
    args = parser.parse_args()
    if args.command == "train": train_autoencoder(_config(args.config))
    elif args.command == "evaluate": run_evaluation(_config(args.config))
    elif args.command == "report": make_report(_config(args.config)["evaluation"]["output_dir"])
    else:
        predicted, embedding = _predict(args.spectrum, args.checkpoint)
        output = Path(args.spectrum).with_suffix(".embedding.npy")
        np.save(output, embedding)
        print(json.dumps({"predicted_class": predicted, "embedding_file": str(output), "embedding": embedding.tolist()}))


if __name__ == "__main__":
    main()
