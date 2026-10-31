# spectra-representation-learning

Self-supervised deep representations for Raman spectra, benchmarked against classical chemometrics.

> **Results (generated, not hard-coded):** run `make smoke` (or `make all` with real data) to regenerate
> [`results/results_table.md`](results/results_table.md) and [`results/figures/embedding.png`](results/figures/embedding.png).
> The checked-in smoke outputs are synthetic and must not be interpreted as experimental results.

## Problem and motivation

Raman spectra are high-dimensional, noisy fingerprints. A compact representation can reduce the amount of
labelled microbiology data needed while retaining class information and improving robustness to measurement
conditions. This project tests that claim with a reproducible self-supervised convolutional autoencoder.

Raman instruments measure inelastically scattered laser light as intensity versus Raman shift. Molecular
vibrations produce characteristic peaks, so spectra are used for identification, process monitoring and
pharmaceutical quality control. Fluorescence, baseline drift, instrument response and day-to-day variation
make preprocessing and transfer evaluation essential.

## Data

The primary target is the public Ho et al. (2019) bacterial dataset. See [`data/README.md`](data/README.md) for
the expected arrays and download link. In this environment the default reproducible path is the documented
**synthetic** Raman-like generator; outputs are labelled synthetic in CSV, JSON, plots and tables.

## Installation

On Linux/WSL2:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
```

Pinned dependencies are in [`requirements.txt`](requirements.txt). CPU is automatic; CUDA is used when available.

## Usage

```bash
make smoke                 # fast complete synthetic run
make all                   # default benchmark; uses real arrays if present
spectra-rl train --config configs/smoke.yaml
spectra-rl evaluate --config configs/smoke.yaml
spectra-rl report --config configs/smoke.yaml
spectra-rl predict sample.npy --checkpoint results/autoencoder.pt
uvicorn spectra_rl.api:app --reload
```

`POST /predict` accepts a CSV/NPY upload; `GET /health` checks service availability.

## Method overview

Spectra receive polynomial baseline correction, Savitzky-Golay smoothing and SNV (or min-max) normalization.
The convolutional autoencoder is trained without labels; frozen embeddings feed logistic regression. PCA plus
random forest and PLS-DA provide classical baselines.

## Evaluation protocol

Reference-domain training and a different instrument/day domain test the transfer setting. Label fractions
(1%, 10%, 100%), multiple Gaussian-noise SNR levels and repeated seeds are evaluated. Means and 95% t
confidence intervals are computed from raw runs in `results/results.csv`.

## Limitations

The fallback generator is not a substitute for biological validation. The small smoke benchmark is intentionally
underpowered, and real-data array formats may require mapping the released files into the documented names.
The current prediction service exposes an embedding-trained model; production class calibration requires a
labelled downstream classifier checkpoint.

## What did not work / what I learned

This section is generated from actual runs: on synthetic data, the compact autoencoder was useful as a stable
feature extractor but the smoke configuration is too small to support claims about superiority. Run the full
benchmark to replace this statement with evidence from the saved summary.

## Design decisions

* `source: auto` uses local real arrays when present and otherwise falls back without failing on network access.
* Synthetic domains are separated by an instrument shift/gain so transfer is exercised in CI.
* The smoke config uses five seeds and small arrays to keep CPU execution practical; default uses ten.
* Checkpoints contain preprocessing metadata so CLI/API inference uses the same transform as training.

## License

MIT. See [`LICENSE`](LICENSE).
