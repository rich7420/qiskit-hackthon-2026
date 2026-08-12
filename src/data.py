"""Data loading and preprocessing for the QNN smoke test."""

from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DatasetName = Literal["mnist", "digits", "auto"]


def load_mnist_binary(
    digits: tuple[int, int] = (0, 1),
    n_features: int = 4,
    n_train: int = 100,
    n_test: int = 100,
    seed: int = 42,
    dataset: DatasetName = "mnist",
):
    """Return PCA-reduced data and ``{-1, +1}`` labels for two digit classes.

    ``dataset="mnist"`` fails explicitly if OpenML is unavailable. Use
    ``dataset="digits"`` for the fully offline sklearn 8x8 dataset. ``"auto"`` may
    fall back to digits, but emits a warning because the two datasets are not directly
    comparable. PCA and scaling are fit on training data only; both splits are clipped
    to the documented ``[0, pi]`` feature range.
    """
    if len(digits) != 2 or digits[0] == digits[1]:
        raise ValueError("digits must contain two distinct class labels")
    if min(n_features, n_train, n_test) <= 0:
        raise ValueError("n_features, n_train, and n_test must all be positive")
    if dataset not in {"mnist", "digits", "auto"}:
        raise ValueError("dataset must be 'mnist', 'digits', or 'auto'")

    if dataset in {"mnist", "auto"}:
        from sklearn.datasets import fetch_openml

        try:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            X, y = mnist.data, mnist.target.astype(int)
            source = "mnist_784"
        except Exception as exc:  # noqa: BLE001 - OpenML exposes several client errors
            if dataset == "mnist":
                raise RuntimeError(
                    "MNIST could not be loaded from OpenML. Use dataset='digits' for "
                    "an explicit offline run or dataset='auto' to permit fallback."
                ) from exc
            warnings.warn(
                f"MNIST unavailable ({type(exc).__name__}: {exc}); using sklearn digits. "
                "This is a different dataset and results are not directly comparable.",
                RuntimeWarning,
                stacklevel=2,
            )
            X, y, source = _load_sklearn_digits()
    else:
        X, y, source = _load_sklearn_digits()

    mask = np.isin(y, digits)
    X, y = X[mask], y[mask]

    if n_train + n_test > len(X):
        raise ValueError(
            f"requested {n_train + n_test} samples but {source} contains only "
            f"{len(X)} examples for digits {digits}"
        )
    if n_features > min(n_train, X.shape[1]):
        limit = min(n_train, X.shape[1])
        raise ValueError(f"n_features={n_features} exceeds the PCA limit of {limit}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[: n_train + n_test]
    X, y = X[idx], y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=seed, stratify=y
    )

    pca = PCA(n_components=n_features, random_state=seed).fit(X_tr)
    scaler = MinMaxScaler(feature_range=(0, np.pi), clip=True).fit(pca.transform(X_tr))
    X_tr = scaler.transform(pca.transform(X_tr))
    X_te = scaler.transform(pca.transform(X_te))

    y_tr = np.where(y_tr == digits[0], -1, 1)
    y_te = np.where(y_te == digits[0], -1, 1)
    return X_tr, X_te, y_tr, y_te, source


def _load_sklearn_digits():
    """Load the explicit offline alternative without hiding a dataset change."""
    from sklearn.datasets import load_digits

    digits_ds = load_digits()
    return digits_ds.data, digits_ds.target, "sklearn_digits_8x8"
