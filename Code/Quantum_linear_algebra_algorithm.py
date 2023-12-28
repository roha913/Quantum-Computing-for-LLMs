import cirq
import numpy as np

def create_matrix_circuit(qubits, matrix):
    """
    Creates a Cirq circuit for matrix multiplication with the given matrix and qubits.
    Assumes the matrix is a square matrix and the number of qubits is appropriate for the matrix size.
    """
    # Verify that the number of qubits matches the matrix size
    if len(qubits) != len(matrix):
        raise ValueError("Number of qubits must be equal to the size of the matrix.")

    # Initialize the circuit
    circuit = cirq.Circuit()

    # Apply the Hadamard gate to all input qubits to create superposition
    circuit.append(cirq.H(qubit) for qubit in qubits)

    # Apply controlled unitary gates to implement matrix multiplication
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            # Construct diagonal gate for the current matrix element
            diagonal_gate = cirq.Rz(2 * np.pi * matrix[row][col])
            # Apply the diagonal gate controlled by the appropriate qubit
            circuit.append(diagonal_gate.on(qubits[col]).controlled_by(qubits[row]))

    # Apply the inverse Quantum Fourier Transform to the qubits
    circuit.append(cirq.inverse(cirq.qft(*qubits)))

    # Add measurement gates to the output qubits
    output_qubits = [cirq.GridQubit(qubits[0].row, i + qubits[-1].col + 1) for i in range(len(qubits))]
    circuit.append(cirq.measure(*output_qubits, key='result'))

    return circuit

# Usage
num_qubits = 3  # For a 3x3 matrix
qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
example_matrix = np.random.rand(num_qubits, num_qubits)  # Random 3x3 matrix
circuit = create_matrix_circuit(qubits, example_matrix)

# Print the circuit
print(circuit)
