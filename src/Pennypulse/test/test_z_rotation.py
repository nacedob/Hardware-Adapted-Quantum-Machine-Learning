from icecream import ic
import src.pennypulse as pennypulse
import pennylane as qml
from pennylane import numpy as np


def test_vz_rotation():
    """
    The idea is to do a Hadamard gate (rx with np.pi/4), a VZ rotation that changes from |+> to |i>.
    Then, applying a RY rotation should have no effect
    """

    w0 = 1.
    ham_base = pennypulse.transmon_interaction(qubit_freq=w0, wires=0)
    duration = 50

    @qml.qnode(qml.device('default.qubit', wires=1))
    def circuit_before_y_rot(phase):
        # Prepare first Hadamard gate
        angle_hadarmard = np.pi/4
        amp_hadamard = pennypulse.utils.compute_amplitude_rotation(angle_hadarmard, duration)
        pulse_hadamard = ham_base + amp_hadamard * qml.X(0)
        qml.evolve(pulse_hadamard)(params=[], t=duration)
        # Now we have |+>

        # apply VZ gate
        pennypulse.vz_rotation(angle=np.pi/2, wire=0, phases=phase)

        return qml.state()

    @qml.qnode(qml.device('default.qubit', wires=1))
    def circuit_y_rotation(state, phase):
        # inti state
        qml.QubitStateVector(state, wires=0)
        # apply RY rotation
        phase[0] += np.pi / 2
        angle_y = np.pi / 4
        amp_y = pennypulse.utils.compute_amplitude_rotation(angle_y, duration)
        pulse_y = ham_base + pennypulse.transmon_drive(amp_y, phase[0], w0, 0)
        qml.evolve(pulse_y)(params=[], t=duration)
        return qml.state()

    phase = [0]
    state_before = circuit_before_y_rot(phase)
    state_after = circuit_y_rotation(state_before, phase)

    ic(state_before, state_after)
    ps_before = list(map(lambda x: abs(x)**2, state_before))
    ps_after = list(map(lambda x: abs(x)**2, state_after))

    for p_bef, p_aft in zip(ps_before, ps_after):
        assert np.isclose(p_bef, p_aft, atol=1e-5)

    print('[OK] Test passed')


if __name__ == '__main__':
    test_vz_rotation()

