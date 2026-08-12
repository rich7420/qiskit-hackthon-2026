"""Two-task data preparation for sequential MNIST -> Fashion-MNIST training."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class Task(NamedTuple):
    """One binary classification task in a shared feature space."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _normalize_amplitudes(array: np.ndarray) -> np.ndarray:
    """Return row-wise unit vectors, rejecting invalid all-zero states."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("PCA produced an all-zero sample, which is not a valid amplitude state")
    return array / norms


def _load_openml_binary(
    name: str,
    classes: tuple[int, int],
    *,
    seed: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    dataset = fetch_openml(name, version=1, as_frame=False, parser="auto")
    X, y = dataset.data, dataset.target.astype(int)
    mask = np.isin(y, classes)
    X, y = X[mask], y[mask]
    if n_samples > len(X):
        raise ValueError(f"requested {n_samples} {name} samples but only {len(X)} are available")
    indices = np.random.default_rng(seed).permutation(len(X))[:n_samples]
    return X[indices], y[indices]


def _build_tasks(
    mnist_features: np.ndarray,
    mnist_labels: np.ndarray,
    fashion_features: np.ndarray,
    fashion_labels: np.ndarray,
    *,
    n_features: int,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[Task, Task]:
    """Build both tasks using one PCA fit on MNIST training data only."""
    mnist_train, mnist_test, mnist_y_train, mnist_y_test = train_test_split(
        mnist_features,
        mnist_labels,
        train_size=n_train,
        test_size=n_test,
        random_state=seed,
        stratify=mnist_labels,
    )
    fashion_train, fashion_test, fashion_y_train, fashion_y_test = train_test_split(
        fashion_features,
        fashion_labels,
        train_size=n_train,
        test_size=n_test,
        random_state=seed,
        stratify=fashion_labels,
    )

    # Keeping the representation fixed isolates changes in the QNN weights across tasks.
    pca = PCA(n_components=n_features, random_state=seed).fit(mnist_train)

    def prepare(array: np.ndarray) -> np.ndarray:
        transformed = pca.transform(array).astype(np.float64)
        return _normalize_amplitudes(transformed)

    def encode(labels: np.ndarray) -> np.ndarray:
        return np.where(labels == 0, -1, 1)

    task1 = Task(
        "MNIST 0/1",
        prepare(mnist_train),
        encode(mnist_y_train),
        prepare(mnist_test),
        encode(mnist_y_test),
    )
    task2 = Task(
        "Fashion-MNIST 0/1",
        prepare(fashion_train),
        encode(fashion_y_train),
        prepare(fashion_test),
        encode(fashion_y_test),
    )
    return task1, task2


def load_two_tasks(
    n_features: int = 16,
    n_train: int = 800,
    n_test: int = 200,
    seed: int = 42,
) -> tuple[Task, Task]:
    """Return sequential tasks in one PCA space fit on MNIST training data."""
    if min(n_features, n_train, n_test) <= 0:
        raise ValueError("feature and split sizes must be positive")
    n_samples = n_train + n_test
    mnist_features, mnist_labels = _load_openml_binary(
        "mnist_784", (0, 1), seed=seed, n_samples=n_samples
    )
    fashion_features, fashion_labels = _load_openml_binary(
        "Fashion-MNIST", (0, 1), seed=seed, n_samples=n_samples
    )
    return _build_tasks(
        mnist_features,
        mnist_labels,
        fashion_features,
        fashion_labels,
        n_features=n_features,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
