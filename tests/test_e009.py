"""Fast tests for e009 quantum time-series forecasting continual learning."""

import numpy as np
from pennylane import numpy as pnp

from src.e009_data import TASK_NAMES, load_forecast_task, load_task_sequence
from src.e009_qtsf import init_weights, make_forecaster, nmse, predict


def test_data_shapes_and_range():
    t = load_forecast_task("narma_5", seq_len=8)
    assert t.X_train.shape[1] == 8
    assert len(t.X_train) == len(t.y_train) and len(t.X_test) == len(t.y_test)
    # scaled to [-1, 1]
    for a in (t.X_train, t.y_train, t.X_test, t.y_test):
        assert a.min() >= -1.0001 and a.max() <= 1.0001
    # time-ordered 80/20 split (test is after train, no overlap by construction)
    assert 0.7 < len(t.X_train) / (len(t.X_train) + len(t.X_test)) < 0.85


def test_all_tasks_load():
    seq = load_task_sequence(list(TASK_NAMES), seq_len=6)
    assert len(seq) == len(TASK_NAMES)
    assert {t.name for t in seq} == set(TASK_NAMES)


def test_forecaster_predicts_in_range():
    qnode, cs, hs = make_forecaster(n_qubits=4, n_layers=1, seq_len=6)
    cw, hw = init_weights(cs, hs, seed=0)
    X = np.random.default_rng(0).uniform(-1, 1, (5, 6))
    pred = np.asarray(predict(qnode, cw, hw, X))
    assert pred.shape == (5,)
    assert np.all(pred >= -1.0) and np.all(pred <= 1.0)   # tanh head


def test_nmse_finite_and_nonneg():
    t = load_forecast_task("damped_shm", seq_len=6)
    qnode, cs, hs = make_forecaster(n_qubits=4, n_layers=1, seq_len=6)
    cw, hw = init_weights(cs, hs, seed=1)
    val = nmse(qnode, cw, hw, t.X_test, t.y_test)
    assert np.isfinite(val) and val >= 0.0


def test_empirical_fisher_nonneg():
    from experiments.e009_continual_forecasting import empirical_fisher

    qnode, cs, hs = make_forecaster(n_qubits=4, n_layers=1, seq_len=6)
    cw, hw = init_weights(cs, hs, seed=2)
    X = np.random.default_rng(2).uniform(-1, 1, (8, 6))
    F = empirical_fisher(qnode, cw, hw, X)
    assert F.shape == (int(np.prod(cs)) + int(np.prod(hs)),)
    assert np.all(F >= 0.0)   # Fisher diagonal is non-negative
