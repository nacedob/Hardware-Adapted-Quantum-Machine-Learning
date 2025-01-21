import pennylane as qml
import jax
from math import pi
from .BaseQNN import BaseQNN

jax.config.update("jax_enable_x64", True)


class GateQNN(BaseQNN):

    def __init__(self, num_qubits: int = 2, num_layers: int = 5, n_workers: int = 1, interface: str = 'jax',
                 realistic_gates: bool = True, seed: int = None):
        """
        Quantum Neural Network (QNN) with a gate-based architecture.

        Args:
            num_qubits: The number of qubits in the QNN. Defaults to 2.
            num_layers: The number of layers in the QNN. Defaults to 5.
            n_workers: The number of workers for parallel execution (currently not effective). Defaults to 1.
            interface: The interface used for the QNN ('jax' or 'pennylane'). Defaults to 'jax'.
            realistic_gates: Whether to use realistic gates. Defaults to True.
            seed: The random seed used for reproducibility. Defaults to None.

        Attributes:
            num_qubits: The number of qubits in the QNN.
            num_layers: The number of layers in the QNN.
            params_per_layer: The number of parameters per layer = 3.
            interface: The interface used for the QNN ('jax' or 'pennylane').
            seed: The random seed used for reproducibility.
            params: The parameters of the QNN circuit.
            projection_angles: The angles used for projection measurements.
            trained: A boolean indicating whether the QNN has been trained.
            training_info: A dictionary to store training information.
            dev: The PennyLane quantum device used for simulations.
            n_workers: The number of workers for parallel execution (currently not effective).
            name: The name of the specific QNN architecture.
            model_name: The name of the overall model.
            realistic_gates: A boolean indicating whether to use realistic
                         (hardware-efficient) gates or arbitrary rotations.

        Raises:
            ValueError: If the number of qubits is greater than the number of layers + 1.
        """
        super().__init__(num_qubits, num_layers, n_workers, params_per_layer=3, interface=interface, seed=seed)
        self.name = f'GateQNN with {num_qubits=}, {num_layers=}. Interface={interface}. N_workers={n_workers}'
        self.model_name = 'GateQNN'
        self.realistic_gates = realistic_gates

    def _base_circuit(self, x, params=None) -> None:
        """
        This is the base circuit of the gate model.
        In few words
            - as encoding gate it uses an arbitrary SU(2) rotation RZ(x[2])RY(x[1])RZ(x[0]).
            - as parametrized one qubit gate uses an arbitrary SU(2) rotation RZ(param[2])RY(param[1])RZ(param[0]).
            - as parametrized two qubit gate:
                · If realistic_gates: multirotations RXZ(0, wire)
                · If not realistic_gates: controlled arbitrary SU(2) rotation RZ(param[2])RY(param[1])RZ(param[0]).
                    Control wire is the corresponding qubit and target is 0.
        Args:
            x: (array-like) single input point
            params: (array-like) parameters for the circuit
        Returns:
            None
        """
        if params is None:
            params = self.params

        n_qubits = (params.shape[0] + 1) // 2
        for layer in range(self.num_layers):
            for qubit in range(n_qubits):
                qml.Rot(*x * pi, wires=qubit)
                if qubit != 0:
                    qml.Rot(*params[2 * qubit - 1][layer, :], wires=qubit)
                    if self.realistic_gates:
                        # RXZ rotation
                        qml.Hadamard(wires=qubit)
                        qml.MultiRZ(theta=params[2 * qubit][layer, 0], wires=[0, qubit])  # params 2 and 3 are discarded
                        qml.Hadamard(wires=qubit)
                    else:
                        qml.CRot(*params[2 * qubit][layer, :], wires=[qubit, 0])
                else:
                    qml.Rot(*params[qubit][layer, :], wires=qubit)

    def get_n_pulses(self) -> int:
        """
        Computes the number of pulses that this model executes for each input point.
        1 qubit gates are composed of 3 gates: RZ * RY * RZ. Since the first and last gates can be performed with VZ
            gates, it is like having one pulse.
        2 qubits gates depend on the self.realistic_gates attribute:
            - If realistic_gates: 1 pulse: a RXZ rotation
            - If not realistic_gates: an arbitrary controlled rotation. Call this number n_gates_per_crot
             TODO: calcular cuantos pulsos hacen falta

        Now, each layer is composed of:
            - Qubit 0 has self.num_layers 1 qubit layers -> self.num_layers pulses
            - Qubit q != 0 has a single qubit rotation plus a two qubit gate. This has:
                - If realistic_gates: self.num_layers RXZ gates -> self.num_layers pulses
                - If not realistic_gates: self.num_layers CRot gates -> self.num_layers pulses * n_gates_per_crot
        TODO: compute this n_gates_per_crot. Pennylane lo transpila como:
        cnot(0,1) Rz_0() Ry_0() cnot(0,1). Como cada cnot se trnaspila luego como Cross Resonance gate como
        Rz0(pi/2) Rx1(pi/2) CR(-pi/2), asumo que cada Cnot son 2 puertas (rotacion en z no la cuento)
        De este modo, asumo que cada CRot son 5 puertas
        :return: number of gates of the total circuit

        Nota: como transpilar en pennylane (aunque le basis_set se lo pasa por el culo)
        ```
        qnode = qml.QNode(
            self._base_circuit,
            self.dev
        )
        transpiled = qml.compile(qnode, basis_set=["CNOT", "RX", "RZ"])
        specs = qml.specs(transpiled)
        x = rand(self.params_per_layer)
        p = rand(2 * self.num_qubits - 1, self.num_layers, self.params_per_layer)
        print('Original')
        print(qml.draw(qnode)(x, p))
        print('\n\nTranspiled')
        print(qml.draw(transpiled)(x, p))
        ic(specs.__dict__)    # Aqui hay informacion sobre el numero de puertas
        ```
        """
        if self.realistic_gates:
            n_gates_per_crot = 1  # RXZ gate is 1 pulses
        else:
            n_gates_per_crot = 5  # CRot gate is 5 pulses:
        return self.num_layers * self.num_qubits + (self.num_qubits - 1) * n_gates_per_crot
