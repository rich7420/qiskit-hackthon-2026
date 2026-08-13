"""Quantum/physics time-series forecasting datasets for continual learning (e009).

Torch-free reimplementation of the generators provided with the hackathon `datasets.zip`
(Apache-2.0, (c) 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen). The original loaders wrap the
series in PyTorch Datasets; here we reproduce the exact series math in numpy and window it into
next-step forecasting pairs, keeping our PennyLane/numpy stack torch-free.

Each task is univariate one-step-ahead forecasting: x = series[i:i+seq_len] (seq_len past
values), y = series[i+seq_len] (next value). Series are scaled to [-1, 1]; the train/test split
is time-ordered 80/20 (no shuffling across the boundary — this is forecasting).

NARMA generators adapted from Samuel Yen-Chi Chen, Sci. Rep. 12 (2022),
https://doi.org/10.1038/s41598-022-05061-w.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.special
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler


class ForecastTask(NamedTuple):
    name: str
    X_train: np.ndarray   # (n_train, seq_len)
    y_train: np.ndarray   # (n_train,)
    X_test: np.ndarray    # (n_test, seq_len)
    y_test: np.ndarray    # (n_test,)


# --- raw 1-D series generators (pure numpy) -------------------------------------------------

def _narma_input(T: int) -> np.ndarray:
    a, b, g = 2.11, 3.73, 4.11
    t = np.arange(T)
    return 0.1 * (np.sin(2 * np.pi * a * t / T) * np.sin(2 * np.pi * b * t / T)
                  * np.sin(2 * np.pi * g * t / T) + 1)


def narma_series(n_0: int = 5, T: int = 400) -> np.ndarray:
    """NARMA-n target series (the forecasting target), following the provided generator."""
    u = _narma_input(T)
    alpha, beta, gamma, delta = 0.3, 0.05, 1.5, 0.1
    y = [0.196]
    u_init = [0.0] * (n_0 - 1)
    y_init = [0.0] * (n_0 - 1)
    for t in range(T):
        y_sum = 0.0
        for j in range(n_0):
            y_sum += y[t - j] if t - j >= 0 else y_init[j - t - 1]
        u_res = u[t - n_0 + 1] if t - n_0 + 1 >= 0 else u_init[n_0 - t - 2]
        y.append(alpha * y[t] + beta * y[t] * y_sum + gamma * u_res * u[t] + delta)
    return np.asarray(y[:-1])


def damped_shm_series(n_points: int = 400) -> np.ndarray:
    """Damped pendulum angular velocity from the 2nd-order ODE (provided generator)."""
    b, g, m = 0.15, 9.81, 1.0

    def system(theta, t):
        return [theta[1], -(b / m) * theta[1] - g * np.sin(theta[0])]

    t = np.linspace(0, 20, n_points)
    theta = odeint(system, [0.0, 3.0], t)
    return theta[:, 1]


def bessel_series(n_points: int = 400) -> np.ndarray:
    """Bessel function of the first kind, order 2 (provided generator)."""
    x = np.linspace(2, 100, n_points)
    return scipy.special.jv(2, x)


def delayed_qc_series() -> np.ndarray:
    """Delayed-quantum-control pulse train: decaying sum of Gaussians (provided generator)."""
    x = np.arange(-2, 20, 0.01)
    data = np.zeros_like(x)
    for n in range(11):
        data += np.exp(-10 * (x - 2 * n) ** 2) * np.exp(-x / 16)
    return data


_SERIES = {
    "narma_5": lambda: narma_series(5, 400),
    "narma_10": lambda: narma_series(10, 400),
    "damped_shm": lambda: damped_shm_series(400),
    "bessel_j2": lambda: bessel_series(400),
    "delayed_qc": lambda: delayed_qc_series(),
}
TASK_NAMES = tuple(_SERIES)


# --- windowing + split ----------------------------------------------------------------------

def _window(series: np.ndarray, seq_len: int):
    s = MinMaxScaler((-1, 1)).fit_transform(np.asarray(series).reshape(-1, 1)).ravel()
    xs, ys = [], []
    for i in range(len(s) - seq_len - 1):
        xs.append(s[i:i + seq_len])
        ys.append(s[i + seq_len])
    return np.asarray(xs), np.asarray(ys)


def load_forecast_task(name: str, seq_len: int = 8, train_frac: float = 0.8) -> ForecastTask:
    """Load one forecasting task, scaled and split time-ordered 80/20."""
    if name not in _SERIES:
        raise ValueError(f"unknown task {name!r}; choose from {TASK_NAMES}")
    X, y = _window(_SERIES[name](), seq_len)
    n_tr = int(train_frac * len(X))
    return ForecastTask(name, X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])


def load_task_sequence(names, seq_len: int = 8) -> list[ForecastTask]:
    """Load an ordered list of forecasting tasks for continual learning."""
    return [load_forecast_task(n, seq_len=seq_len) for n in names]
