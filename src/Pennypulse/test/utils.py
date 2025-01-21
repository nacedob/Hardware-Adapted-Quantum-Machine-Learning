import numpy as np
from scipy.integrate import solve_ivp
import pennylane as qml

X = qml.X(0).matrix()
Y = qml.Y(0).matrix()
Z = qml.Z(0).matrix()
Identity = qml.Identity(0).matrix()


def evolve(h, psi0, t0, tend,  *, time_steps=1000):
    """
    Evolve a quantum state under a time-dependent Hamiltonian.

    Parameters:
        psi0 (ndarray): Initial state vector.
        t0 (float): Initial time.
        tend (float): Final time.
        h (callable or ndarray): Time-dependent Hamiltonian. If callable, should have signature h(t).
                               If constant, should be a square matrix.
        time_steps (int): Number of time steps for the evolution.

    Returns:
        ndarray: Final state vector after evolution.
    """
    psi0 = np.asarray(psi0, dtype=complex)

    # Define the Schrödinger equation
    def schrodinger_equation(t, psi):
        if callable(h):
            h_t = h(t)
        else:
            h_t = h
        return -1j * np.dot(h_t, psi)

    # Solve the time-dependent Schrödinger equation
    times = np.linspace(t0, tend, time_steps)
    sol = solve_ivp(schrodinger_equation, [t0, tend], psi0, t_eval=times, method="RK45")

    if not sol.success:
        raise RuntimeError("Failed to solve the Schrödinger equation.")

    # Return the final state
    return sol.y[:, -1]