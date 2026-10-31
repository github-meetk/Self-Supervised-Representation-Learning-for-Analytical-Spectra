# Data

Raw files are intentionally git-ignored. The loader looks in `data/raw/` for
`spectra.npy` and `labels.npy` (optionally `domains.npy`). The primary source is
the [Ho et al. 2019 bacteria-ID repository](https://github.com/csho33/bacteria-ID);
download its released arrays/manually place them in this directory and set
`data.source: real` in a config. The loader attempts the public GitHub download
when `source: auto`, but never makes a run depend on network access.

Without those files, the documented fallback is a deterministic synthetic
Raman-like generator: class-specific Gaussian/Lorentzian peaks, fluorescence
baseline, Gaussian noise, and domain-specific shift/gain. Synthetic outputs
are explicitly labelled `synthetic` and are not real experimental results.
