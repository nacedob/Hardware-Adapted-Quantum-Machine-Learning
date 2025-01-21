import numpy as np

import src.pennypulse as pennypulse
import pennylane as qml
from icecream import ic


def test_gaussian_shape():
    w0 = 2
    h_base = pennypulse.transmon_interaction(w0, 0)


    @qml.qnode(qml.device('default.qubit', wires=0))
    def circuit(amp, sigma, duration):
        pulse = pennypulse.gaussian_pulse(amp, sigma, duration=duration, phase=0, freq=w0, wires=0)
        h_tot = h_base + pulse
        qml.evolve(h_base)(params=[], t=duration)
        return qml.density_matrix(wires=0)


    # Define a state with some arbitrary params
    base_amp = 10
    base_sigma = 4
    base_duration = 25
    base_state = circuit(base_amp, base_sigma, base_duration)


    # Assert callable
    base_pulse = pennypulse.gaussian_pulse(base_amp, base_sigma, duration=base_duration, phase=0, freq=w0, wires=0)
    ic(base_pulse(params=[1], t=base_duration-1).scalar)
    assert base_pulse(params=[1], t=1).scalar, float   # el params me trae por la called la amargura, para que se calle
    assert base_pulse(params=[1], t=1).scalar != 1.   # el params me trae por la called la amargura, para que se calle

    ic(base_state)
    # Assert a change in amplitude has some effect
    amp_state = circuit(2 * base_amp, base_sigma, base_duration)
    ic(amp_state)
    assert not np.allclose(amp_state, base_state)

    # Assert a change in sigma has some effect
    sigma_state = circuit(base_amp, 2 * base_sigma, base_duration)
    ic(sigma_state)
    assert not np.allclose(sigma_state, base_state)

    # Assert a change in duration has some effect
    duration_state = circuit(base_amp, base_sigma, 2 * base_duration)
    ic(duration_state)
    assert not np.allclose(duration_state, base_state)

    print('[OK] Test run successfully')


if __name__ == '__main__':
    test_gaussian_shape()
