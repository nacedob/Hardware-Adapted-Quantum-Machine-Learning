from icecream import ic
import pennylane as qml
from .utils.integration import integrate_ranges
from jax import numpy as jnp
from jax import vmap


def transmon_trotter_suzuki_2q_drive1q(q_freqs: list[float], coupling: float, amplitude_func: callable,
                                       drive_freq: float, drive_phase: float, wire: int,
                                       n_trotter: int = 5,
                                       duration: float = 10, t_start: float = 0):
    """
    Applies a driving to qubit defined by wire. It has into account transmon_interaction plus driving.
    This function only works for 2 qubits.
    The driving phase is supposed to be constant for the pulse.
    Exact hamiltonian:

    H = - Σ_{q=1,2} (ω_q / 2) σ_q^z
            + J (σ_1^x σ_2^x + σ_1^y σ_2^y)
                + Ω(t) sin(ω_d t - γ) σ_d^x
    """
    drive_function = lambda t: amplitude_func(t) * jnp.sin(drive_freq * t + drive_phase)
    step = duration / n_trotter

    # Evaluate the driving function for middle time points
    t_vals = jnp.arange(t_start, t_start + duration, step)
    driving_values = 2 * integrate_ranges(drive_function, t_vals)

    # Evaluate qubit frequency rotations values in the middle of the time interval
    freq_rotation_value = [-freq * step / (2) for freq in q_freqs]
    coupling_value = coupling * step

    def layer(step) -> None:
        # natural freq of qubit term
        for q in range(2):
            qml.RZ(freq_rotation_value[q], wires=q)

            # Coupling term
            # XX interaction
            qml.Hadamard(wires=q)
        qml.MultiRZ(coupling_value, wires=[0, 1])
        for q in range(2):
            qml.Hadamard(wires=q)

            # YY interaction
            qml.S(wires=q)
            qml.Hadamard(wires=q)
        qml.MultiRZ(coupling_value, wires=[0, 1])
        for q in range(2):
            qml.Hadamard(wires=q)
            qml.adjoint(qml.S(wires=q))

        # Driving term
        qml.RX(driving_values[step], wires=wire)

        # Coupling term
        # XX interaction
        for q in range(2):
            qml.Hadamard(wires=q)
        qml.MultiRZ(coupling_value, wires=[0, 1])
        for q in range(2):
            qml.Hadamard(wires=q)

            # YY interaction
            qml.S(wires=q)
            qml.Hadamard(wires=q)
        qml.MultiRZ(coupling_value, wires=[0, 1])
        for q in range(2):
            qml.Hadamard(wires=q)
            qml.adjoint(qml.S(wires=q))

            # natural freq of qubit term
            qml.RZ(freq_rotation_value[q], wires=q)

    for i in range(n_trotter - 1):
        layer(i)


def transmon_trotter_suzuki_1q_drive(q_freq: float, amplitude_func: callable,
                                     drive_freq: float, drive_phase: float, wire: int,
                                     n_trotter: int = 5,
                                     duration: float = 10, t_start: float = 0):
    """
    Applies a driving to qubit defined by wire. It has into account transmon_interaction plus driving.
    This function only works for 2 qubits.
    The driving phase is supposed to be constant for the pulse.
    Exact hamiltonian:

    H = - (ω_q / 2) σ^z
                + Ω(t) sin(ω_d t - γ) σ^x
    """
    drive_function = lambda t: amplitude_func(t) * jnp.sin(drive_freq * t + drive_phase)
    step = duration / n_trotter

    # Evaluate the driving function for middle time points
    t_vals = jnp.arange(t_start, t_start + duration, step)
    driving_values = 2 * integrate_ranges(drive_function, t_vals)

    # Evaluate qubit frequency rotations values in the middle of the time interval
    freq_rotation_value = -q_freq * step / 2

    def layer(step) -> None:
        # natural freq of qubit term
        qml.RZ(freq_rotation_value, wires=wire)

        # Driving term
        qml.RX(driving_values[step], wires=wire)

        # natural freq of qubit term
        qml.RZ(freq_rotation_value, wires=wire)

    for i in range(n_trotter - 1):
        layer(i)
