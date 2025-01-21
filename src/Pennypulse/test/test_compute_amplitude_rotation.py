from icecream import ic
import src.pennypulse as pennypulse
import pennylane as qml
from pennylane import numpy as np


def test_x_pi_2_rotation():
    """
    The idea is to do a X gate using the transmon drive and computing the amplitude for a fixed duration.
    """

    w0 = 1.
    ham_base = pennypulse.transmon_interaction(qubit_freq=w0, wires=0)
    duration = 50

    @qml.qnode(qml.device('default.qubit', wires=1))
    def circuit():
        # Select parameters to do a X gate (pi/2 X rotation)
        phase = 0
        angle_x = np.pi/2
        amp_x = pennypulse.utils.compute_amplitude_rotation(angle_x, duration)
        pulse_x = ham_base + pennypulse.transmon_drive(qml.pulse.constant, phase, w0, 0)

        qml.evolve(pulse_x)(params=[amp_x], t=duration)

        return qml.state()

    state = circuit()
    ic(state)
    assert np.isclose(state[1], 1)
    print("[OK] Test passed")

def test_y_pi_2_rotation():
    """
    The idea is to do a Y gate using the transmon drive and computing the amplitude for a fixed duration.
    """

    w0 = 1.
    ham_base = pennypulse.transmon_interaction(qubit_freq=w0, wires=0)
    duration = 50

    @qml.qnode(qml.device('default.qubit', wires=1))
    def circuit():
        # Select parameters to do a Y gate (pi/2 X rotation)
        phase = np.pi/2
        angle_y = np.pi/2
        amp_y = pennypulse.utils.compute_amplitude_rotation(angle_y, duration)
        pulse_y = ham_base + pennypulse.transmon_drive(qml.pulse.constant, phase, w0, 0)

        qml.evolve(pulse_y)(params=[amp_y], t=duration)

        return qml.state()

    state = circuit()
    ic(state)
    assert np.isclose(state[1], 1)
    print("[OK] Test passed")


if __name__ == '__main__':
    test_x_pi_2_rotation()
