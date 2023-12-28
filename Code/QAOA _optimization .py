import cirq
import numpy as np
from scipy.optimize import minimize

# Define a problem Hamiltonian
# The Hamiltonian now includes interactions between multiple qubits
num_qubits = 4
qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
problem_ham = sum(cirq.Z(qubits[i]) * cirq.Z(qubits[(i + 1) % num_qubits]) for i in range(num_qubits))

# Define an enhanced mixing Hamiltonian
# Incorporates X gates on all qubits
mixing_ham = sum(cirq.X(q) for q in qubits)

# Increase the number of QAOA layers
num_layers = 5

# Define the QAOA circuit with parameterized gates
def qaoa_circuit(params):
    circuit = cirq.Circuit()
    # Start with a superposition state using Hadamard gates
    circuit.append(cirq.H(q) for q in qubits)

    for i in range(num_layers):
        # Add problem unitary
        circuit.append(cirq.ZZPowGate(exponent=params[2 * i])(qubits[j], qubits[(j + 1) % num_qubits]) for j in range(num_qubits))
        # Add mixing unitary
        circuit.append(cirq.XPowGate(exponent=params[2 * i + 1])(q) for q in qubits)

    # Add measurement at the end
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit

# Define a function to evaluate the expectation value of the problem Hamiltonian
def objective_function(params):
    circuit = qaoa_circuit(params)
    result = simulator.run(circuit, repetitions=100)
    measurements = result.measurements['result']
    return np.mean([problem_ham.expectation_from_state_vector(cirq.final_state_vector(result), qubit_map={q: i for i, q in enumerate(qubits)}) for result in results])

# Optimize the parameters
initial_params = np.random.rand(2 * num_layers)
simulator = cirq.Simulator()
result = minimize(objective_function, initial_params, method='BFGS')

# Print the optimal parameters and corresponding value
print("Optimal parameters:", result.x)
print("Minimum objective value:", result.fun)
