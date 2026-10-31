"""Data generation, loading and split utilities."""

from .loader import load_dataset
from .splits import stratified_split, subsample_labels
from .synthetic import make_synthetic

__all__ = ["load_dataset", "make_synthetic", "stratified_split", "subsample_labels"]

