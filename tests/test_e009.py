"""Fast tests for e009 quantum time-series forecasting continual learning."""

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from src.e009_data import TASK_NAMES, load_forecast_task, load_task_sequence
from src.e009_qtsf import (
    init_weights, make_forecaster, make_state_forecaster, nmse, predict,
)


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


# --- gate-count ablation: lock the ansatz variants' real circuit cost (qml.specs) ---

def _resources(n_layers, seq_len=8, **ansatz):
    qnode, cs, hs = make_forecaster(n_qubits=4, n_layers=n_layers, seq_len=seq_len, **ansatz)
    r = qml.specs(qnode)(np.zeros((1, seq_len)), np.zeros(cs)).resources
    return r, int(np.prod(cs)) + int(np.prod(hs))


@pytest.mark.parametrize("n_layers,entangler,encoding,gates,cnot,params", [
    (2, "ring",  "ry_rz", 256, 64, 21),   # baseline = the original model
    (2, "chain", "ry_rz", 240, 48, 21),   # drop the CNOT wrap-around
    (2, "ring",  "ry",    224, 64, 21),   # single-axis encoding drops 32 encode-RZ
    (1, "ring",  "ry_rz", 160, 32, 13),   # one variational layer
    (1, "chain", "ry",    120, 24, 13),   # aggressive: -62% CNOT vs baseline
    (1, "none",  "ry",     96,  0, 13),   # no-entanglement floor
])
def test_ansatz_gate_counts(n_layers, entangler, encoding, gates, cnot, params):
    r, n_params = _resources(n_layers, entangler=entangler, encoding=encoding)
    assert r.num_gates == gates
    assert r.gate_sizes.get(2, 0) == cnot   # two-qubit gate count
    assert n_params == params


def test_default_ansatz_reproduces_original_model():
    # defaults MUST stay the 256-gate / 64-CNOT ring + two-axis model so prior results hold
    r_default, _ = _resources(2)
    assert r_default.num_gates == 256 and r_default.gate_sizes.get(2, 0) == 64


def test_state_forecaster_entangling_matches_forecaster():
    # the QFI state circuit must use the SAME ansatz (same CNOT count) as the trained forecaster
    for kw in (dict(entangler="ring", encoding="ry_rz"), dict(entangler="chain", encoding="ry")):
        qf, cs, _ = make_forecaster(n_qubits=4, n_layers=2, seq_len=8, **kw)
        sf = make_state_forecaster(4, 2, 8, **kw)
        rf = qml.specs(qf)(np.zeros((1, 8)), np.zeros(cs)).resources
        rs = qml.specs(sf)(np.zeros(8), np.zeros(cs)).resources
        assert rf.gate_sizes.get(2, 0) == rs.gate_sizes.get(2, 0)


def test_single_axis_encoding_only_removes_rz():
    r2, _ = _resources(2, entangler="ring", encoding="ry_rz")
    r1, _ = _resources(2, entangler="ring", encoding="ry")
    assert r2.gate_types["RZ"] - r1.gate_types["RZ"] == 32   # 4 qubits x 8 steps of encode-RZ
    assert r2.gate_types["RY"] == r1.gate_types["RY"]        # RY encoding untouched
