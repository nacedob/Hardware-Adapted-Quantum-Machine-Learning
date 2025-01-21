"""
Este test es basicamente para ver si tengo que meter un hbar en los coefficientes de los hamiltonianos.
Xanadu no los ha metido, pero hay cuentas que si no no sale. Del mismo modo, al meterlos, hay otras cuentas que no
salen.
Vamos a mirar que se puede hacer
"""

import unittest
from src.pennypulse.constants import hbar
import pennylane as qml
from icecream import ic
from copy import copy
from pennylane import numpy as np

class Testhbar(unittest.TestCase):

    def test_pi_x_rotation(self):
        # Una rotacion con un pulse de amplitud constante en X es una rotacion de angulo amp * duration

        w0 = 2
        # pennylane hamiltonian without hbar
        h_no_hbar = qml.pulse.transmon_interaction(w0, wires=range(2), connections=[[0,1]], coupling=1e-10)

        # pennylane hamiltonian with hbar
        h_hbar = copy(h_no_hbar)
        h_hbar.coeffs_fixed = [x * hbar for x in h_hbar.coeffs_fixed]

        # pulse
        angle = np.pi   # from 0 to 1
        amplitude = 5
        duration = hbar * angle / amplitude * 2  # Esto la duration que deberia de tener en teoria para acabar en |1>
        pulse = qml.X(0) * amplitude

        dev = qml.device('default.qubit', wires=range(2))

        @qml.qnode(dev, interface="jax")
        def circuit(hamiltonian_base):
            qml.evolve(hamiltonian_base + pulse)(params=[], t=duration)
            return qml.density_matrix(0)


        dm_no_hbar = circuit(h_no_hbar)
        dm_hbar = circuit(h_hbar)

        ic(dm_no_hbar, dm_hbar)

        prob_1_no_hbar = dm_no_hbar[1,1]
        prob_1_hbar = dm_hbar[1,1]

        ic(prob_1_hbar, prob_1_no_hbar)


if __name__ == '__main__':
    unittest.main()
