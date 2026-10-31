import numpy as np
import torch

from spectra_rl.data.splits import stratified_split, subsample_labels
from spectra_rl.data.synthetic import make_synthetic
from spectra_rl.models.cnn1d import CNN1DClassifier, ConvAutoencoder
from spectra_rl.preprocessing import Pipeline, baseline_correct, snv


def test_preprocessing():
    X, _, _ = make_synthetic(2, 3, 32)
    out = Pipeline(window_length=7).fit_transform(X)
    assert out.shape == X.shape and np.isfinite(out).all()
    assert snv(X).shape == X.shape
    assert baseline_correct(X).shape == X.shape


def test_split_and_subsample():
    _, y, d = make_synthetic(3, 10, 32)
    train, test = stratified_split(y, d)
    assert not set(train) & set(test)
    small = subsample_labels(y, train, .1, 1)
    assert set(np.unique(y[small])) == set(np.unique(y[train]))


def test_forward_and_learning():
    x = torch.randn(4, 32)
    assert ConvAutoencoder(32, 8)(x).shape == x.shape
    assert CNN1DClassifier(32, 3, 8)(x).shape == (4, 3)
    model = ConvAutoencoder(32, 8)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    loss1 = torch.nn.functional.mse_loss(model(x), x)
    opt.zero_grad(); loss1.backward(); opt.step()
    loss2 = torch.nn.functional.mse_loss(model(x), x)
    assert float(loss2) <= float(loss1)
