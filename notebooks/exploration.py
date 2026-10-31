"""Lightweight exploration entry point; kept as a plain script for reproducible CI."""
from spectra_rl.data.synthetic import make_synthetic

if __name__ == "__main__":
    X, y, domains = make_synthetic()
    print(X.shape, len(set(y)), len(set(domains)))
