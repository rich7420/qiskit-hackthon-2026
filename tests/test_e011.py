"""Formulation tests for E011 (blocked vs interleaved schedule). Fast: no MNIST, no training.

The experiment's validity rests on one invariant: both schedules spend exactly the same
per-task gradient budget, so any difference in retention is due to *ordering* alone. These
tests pin that invariant plus the forgetting metric.
"""

from collections import Counter

import numpy as np

from experiments.e011_interleaved import _metrics_for, _step_order


def test_schedules_share_the_same_per_task_budget():
    n_tasks, epochs = 3, 20
    blocked = _step_order("blocked", n_tasks, epochs)
    interleaved = _step_order("interleaved", n_tasks, epochs)
    # Same total steps and same count per task -> identical compute, only order differs.
    assert len(blocked) == len(interleaved) == n_tasks * epochs
    assert Counter(blocked) == Counter(interleaved) == {t: epochs for t in range(n_tasks)}


def test_blocked_is_contiguous_interleaved_is_round_robin():
    blocked = _step_order("blocked", 3, 4)
    interleaved = _step_order("interleaved", 3, 4)
    # Blocked runs each task in one contiguous run.
    assert blocked == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    # Interleaved cycles through the tasks every round.
    assert interleaved == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_metrics_forgetting_is_best_minus_final():
    # Build a history where task1 peaks at 1.0 then decays to 0.6 (forgetting=0.4),
    # task2 rises monotonically to 0.9 (no forgetting), task3 stays flat at 0.5.
    class T:  # minimal stand-in for src.continual_data.Task (only .name is read)
        def __init__(self, name):
            self.name = name

    tasks = [T("A"), T("B"), T("C")]
    curves = {"task1": [0.5, 1.0, 0.8, 0.6],
              "task2": [0.5, 0.6, 0.8, 0.9],
              "task3": [0.5, 0.5, 0.5, 0.5]}
    history = [{"test_accuracy": {k: curves[k][i] for k in curves}} for i in range(4)]

    m = _metrics_for(history, tasks)
    assert m["tasks"]["task1"]["forgetting"] == 0.4
    assert m["tasks"]["task2"]["forgetting"] == 0.0   # best == final
    assert m["tasks"]["task1"]["test_final"] == 0.6
    # mean earlier = mean(final T1, final T2); mean all includes T3.
    assert np.isclose(m["mean_earlier_task_final"], np.mean([0.6, 0.9]), atol=1e-4)
    assert np.isclose(m["mean_all_task_final"], np.mean([0.6, 0.9, 0.5]), atol=1e-4)


def test_unknown_schedule_rejected():
    try:
        _step_order("nonsense", 3, 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown schedule name")
