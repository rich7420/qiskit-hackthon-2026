"""MNIST preprocessing for amplitude-encoded experiments."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class AmplitudeSplits(NamedTuple):
    """Train/validation/test arrays plus the source dataset name."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    source: str


def _normalize_amplitudes(array: np.ndarray) -> np.ndarray:
    """Return row-wise unit vectors, rejecting invalid all-zero states."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("PCA produced an all-zero sample, which is not a valid amplitude state")
    return array / norms


def _fit_transform_pca(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    *,
    n_features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on training data only, then normalize all three splits."""
    pca = PCA(n_components=n_features, random_state=seed).fit(X_train)
    return tuple(
        _normalize_amplitudes(pca.transform(split).astype(np.float64))
        for split in (X_train, X_validation, X_test)
    )


def load_mnist_amplitude_pca(
    digits: tuple[int, int] = (0, 1),
    n_features: int = 16,
    n_train: int = 600,
    n_validation: int = 200,
    n_test: int = 200,
    seed: int = 42,
) -> AmplitudeSplits:
    """Load MNIST, split it, fit train-only PCA, and L2-normalize each row.

    Labels are mapped to ``{-1, +1}``. The held-out test split is returned for the
    final frozen configuration only; model and epoch selection must use validation.
    """
    from sklearn.datasets import fetch_openml

    if len(set(digits)) != 2:
        raise ValueError("digits must contain exactly two distinct classes")
    if min(n_features, n_train, n_validation, n_test) <= 0:
        raise ValueError("feature and split sizes must be positive")

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)
    mask = np.isin(y, digits)
    X, y = X[mask], y[mask]

    requested = n_train + n_validation + n_test
    if requested > len(X):
        raise ValueError(f"requested {requested} samples but only {len(X)} are available")

    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(X))[:requested]
    X, y = X[selected], y[selected]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        train_size=n_train,
        test_size=n_validation + n_test,
        random_state=seed,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        train_size=n_validation,
        test_size=n_test,
        random_state=seed,
        stratify=y_holdout,
    )
    X_train, X_validation, X_test = _fit_transform_pca(
        X_train,
        X_validation,
        X_test,
        n_features=n_features,
        seed=seed,
    )

    def encode(labels: np.ndarray) -> np.ndarray:
        return np.where(labels == digits[0], -1, 1)

    return AmplitudeSplits(
        X_train,
        X_validation,
        X_test,
        encode(y_train),
        encode(y_validation),
        encode(y_test),
        "mnist_784",
    )
