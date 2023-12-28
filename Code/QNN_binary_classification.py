import cirq
import numpy as np
from scipy.optimize import minimize

def build_circuit(qubits, params):
    """
    Builds a quantum circuit for binary classification.
    Args:
        qubits: List of qubits to use in the circuit.
        params: Parameters for the quantum gates.
    """
    circuit = cirq.Circuit()

    # Layer of single-qubit rotations based on input parameters
    for i, qubit in enumerate(qubits[:-1]):
        circuit.append(cirq.rx(params[2*i])(qubit))
        circuit.append(cirq.rz(params[2*i + 1])(qubit))

    # Two-qubit entangling gates
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CZ(qubits[i], qubits[i+1]))

    # Final layer of single-qubit rotation on the last qubit
    circuit.append(cirq.rx(params[-2])(qubits[-1]))
    circuit.append(cirq.rz(params[-1])(qubits[-1]))

    return circuit

def cost_function(params, X, y):
    """
    Computes the cost function for the binary classification.
    Args:
        params: Parameters for the quantum circuit.
        X: Input features.
        y: Target labels.
    """
    qubits = cirq.LineQubit.range(len(X[0]) + 1)
    circuit = build_circuit(qubits, params)
    simulator = cirq.Simulator()
    loss = 0

    # Compute the predicted labels and loss
    for i in range(len(X)):
        # Encode data into the circuit
        data_circuit = cirq.Circuit([cirq.rx(x)(qubits[j]) for j, x in enumerate(X[i])])
        full_circuit = data_circuit + circuit
        full_circuit.append(cirq.measure(qubits[-1], key='result'))

        # Simulate the circuit
        result = simulator.run(full_circuit, repetitions=100)
        counts = result.histogram(key='result')
        prob_1 = counts.get(1, 0) / 100  # Probability of measuring |1>
        
        # Binary cross-entropy loss
        label = y[i]
        loss += -label * np.log(prob_1) - (1 - label) * np.log(1 - prob_1)

    return loss / len(X)

# Data and labels
X = [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6], [6.4, 2.8, 5.6, 2.2], [5.0, 3.4, 1.5, 0.2], [6.0, 2.9, 4.5, 1.5], [6.7, 3.1, 5.6, 2.4]]
y = [1, 0, 0, 1, 0, 0]

def objective(params):
    return cost_function(params, X, y)

initial_params = np.random.rand(2 * (len(X[0]) + 1))
result = minimize(objective, initial_params, method='nelder-mead')

print("Optimized parameters:", result.x)

