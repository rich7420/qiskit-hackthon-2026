"""Quantum neural network construction for the MNIST smoke test.

A textbook variational classifier: a ZZ feature map to encode the input, a RealAmplitudes
ansatz with trainable weights, and a single Z-parity expectation as the output. Runs on a
local statevector simulator (no hardware needed).
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN


def build_estimator_qnn(
    n_qubits: int = 4,
    ansatz_reps: int = 1,
    feature_reps: int = 1,
    seed: int | None = None,
) -> EstimatorQNN:
    """Build a single-output EstimatorQNN on ``n_qubits`` qubits.

    Output is the expectation of Z on every qubit (a parity observable), which lives in
    ``[-1, +1]`` and pairs with ``{-1, +1}`` labels for binary classification.
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2 for a ZZ feature map")
    if min(ansatz_reps, feature_reps) <= 0:
        raise ValueError("ansatz_reps and feature_reps must be positive")

    feature_map = zz_feature_map(n_qubits, reps=feature_reps)
    ansatz = real_amplitudes(n_qubits, reps=ansatz_reps)

    circuit = QuantumCircuit(n_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp("Z" * n_qubits)
    estimator = StatevectorEstimator(seed=seed)

    return EstimatorQNN(
        circuit=circuit,
        estimator=estimator,
        gradient=ParamShiftEstimatorGradient(estimator),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
