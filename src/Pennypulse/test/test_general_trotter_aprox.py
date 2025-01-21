"""
Quiero ver que la trotterization funciona.
No tiene nada que ver con pulsos. Solo comprueba como se partiría exp(i (X+Y)) en general
"""
import pytest
from src.Pennypulse.test.utils import evolve
from src.Pennypulse.src.pennypulse.utils.integration import integrate_ranges
import numpy as np
from jax import numpy as jnp
from scipy.linalg import expm
import pennylane as qml
from icecream import ic


X = qml.ops.X(0).matrix()
Y = qml.ops.Y(0).matrix()
Z = qml.ops.Z(0).matrix()
Identity = qml.ops.Identity(0).matrix()
duration = 0.1

RX = lambda angle: qml.RX(angle, wires=[0]).matrix()
RY = lambda angle: qml.RY(angle, wires=[0]).matrix()
RZ = lambda angle: qml.RZ(angle, wires=[0]).matrix()

def test_trotter():
    exact = qml.exp(qml.ops.X(0) + qml.ops.Y(0), -1j * duration).matrix()
    trotter = qml.exp(qml.ops.X(0), 1/2 * -1j * duration).matrix() @ qml.exp(qml.ops.Y(0), -1j * duration).matrix()  @ qml.exp(qml.ops.X(0), 1/2 * -1j * duration).matrix()
    print()
    ic(exact, trotter, abs(exact - trotter))
    np.testing.assert_almost_equal(abs(exact - trotter), np.zeros((2, 2)), decimal=1)

    #With numpy
    n_trotter = 10
    exact_np = expm(-1j * duration * (X + Y))
    trotter_np = Identity
    for _ in range(n_trotter):
        trotter_np = trotter_np @  expm(X * 1/2 * -1j * duration/n_trotter) @ expm(Y * -1j * duration/n_trotter)  @ expm(X * 1/2 * -1j * duration/n_trotter)

    ic(exact_np, trotter_np, abs(exact_np - trotter_np))
    np.testing.assert_almost_equal(abs(exact_np - trotter_np), np.zeros((2, 2)), decimal=3)

def test_trotter_with_rotation():
    """
    Por ejemplo hagamos aproximacion de e^{i(3X-2Y)t}.
    Recuerda que RX = exp{-itheta/2 X}
    """

    # Before anything check equivalence of matrix and rotation:
    # X rotation
    n_trotter = 2
    matrix_form = expm(1j * 3 / 2 * X * duration / n_trotter)
    rotation_form = RX(-3 * duration/n_trotter)
    np.testing.assert_almost_equal(matrix_form, rotation_form)
    # Y rotation
    matrix_form = expm(-1j * 2 * Y * duration/n_trotter)
    rotation_form = RY(4 * duration/n_trotter)
    np.testing.assert_almost_equal(matrix_form, rotation_form)

    exact = expm(1j * (3*X - 2*Y) * duration)

    n_trotter = 10
    trotter_matrix_exp = Identity
    for _ in range(n_trotter):
        trotter_matrix_exp = trotter_matrix_exp @ expm(1j * 3 / 2 * X * duration/n_trotter) @ expm(-1j * 2 * Y * duration/n_trotter) @ expm(1j * 3 / 2 * X * duration/n_trotter)
    error_matrix = abs(trotter_matrix_exp - exact)

    # trotter with rotations
    trotter_matrix_rotations = Identity
    for _ in range(n_trotter):
        trotter_matrix_rotations = trotter_matrix_rotations @ RX(-3 * duration/n_trotter) @ RY(4 * duration/n_trotter) @ RX(-3 * duration/n_trotter)
    error_rotations = abs(trotter_matrix_rotations - exact)

    ic(error_rotations, error_matrix)
    np.testing.assert_almost_equal(abs(error_rotations), np.zeros((2, 2)), decimal=3)
    np.testing.assert_almost_equal(abs(error_matrix), np.zeros((2, 2)), decimal=3)



def test_trotter_three_gates():
    """
    Por ejemplo hagamos aproximacion de e^{-i(3X-2Y+Z)t}.
    """
    n_trotter = 100
    exact = qml.exp(3 * qml.ops.X(0) -2 * qml.ops.Y(0) + qml.ops.Z(0), -1j * duration).matrix()
    trotter = Identity
    for _ in range(n_trotter):
        trotter = trotter  @ qml.exp(qml.ops.X(0), 3 / 2 * (-1j) * duration/n_trotter).matrix() @ \
                      qml.exp(qml.ops.Y(0), (-2)/2 * (-1j) * duration/n_trotter).matrix() @ \
                            qml.exp(qml.ops.Z(0), (-1j) * duration/n_trotter).matrix() @ \
                                qml.exp(qml.ops.Y(0), (-2)/2 * (-1j) * duration/n_trotter).matrix() @ \
                                    qml.exp(qml.ops.X(0), 3 / 2 * (-1j) * duration/n_trotter).matrix()
    print()
    ic(exact, trotter, abs(exact - trotter))
    np.testing.assert_almost_equal(abs(exact - trotter), np.zeros((2, 2)), decimal=1)

def test_trotter_three_gates_vs_numerical_integration():
    """
    Por ejemplo hagamos aproximacion de e^{-i(3X-2Y+Z)t}.
    """
    psi0 = np.array([1, 0])

    # Trotterization
    n_trotter = 100
    trotter = Identity
    for _ in range(n_trotter):
        trotter = trotter  @ qml.exp(qml.ops.X(0), 3 / 2 * (-1j) * duration/n_trotter).matrix() @ \
                      qml.exp(qml.ops.Y(0), (-2)/2 * (-1j) * duration/n_trotter).matrix() @ \
                            qml.exp(qml.ops.Z(0), (-1j) * duration/n_trotter).matrix() @ \
                                qml.exp(qml.ops.Y(0), (-2)/2 * (-1j) * duration/n_trotter).matrix() @ \
                                    qml.exp(qml.ops.X(0), 3 / 2 * (-1j) * duration/n_trotter).matrix()
    psi_trotter = trotter @ psi0

    # Numerical integration
    h = 3 * X - 2 * Y + Z
    psi_numerical = evolve(h, psi0, t0=0, tend=duration, time_steps=1000)

    ic(psi_numerical, psi_trotter, abs(psi_numerical) - abs(psi_trotter))
    np.testing.assert_almost_equal(psi_numerical, psi_trotter, decimal=3)

def test_trotter_with_trotter_vs_numerical_integration():
    """
    Por ejemplo hagamos aproximacion de e^{-i(3XX-2YY+ZI)t}.
    """

    psi0 = np.array([1, 0, 0, 0])

    # Trotterization
    n_trotter = 100
    trotter = np.kron(Identity, Identity)
    for _ in range(n_trotter):
        trotter = trotter  @ qml.exp(qml.ops.X(0)@qml.ops.X(1), 3 / 2 * (-1j) * duration/n_trotter).matrix() @ \
                      qml.exp(qml.ops.Y(0)@qml.ops.Y(1), (-2)/2 * (-1j) * duration/n_trotter).matrix() @ \
                            qml.exp(qml.ops.Z(0) @ qml.Identity(1), (-1j) * duration/n_trotter).matrix() @ \
                                qml.exp(qml.ops.Y(0)@qml.ops.Y(1), (-2)/2 * (-1j) * duration/n_trotter).matrix() @ \
                                    qml.exp(qml.ops.X(0)@qml.ops.X(1), 3 / 2 * (-1j) * duration/n_trotter).matrix()
    psi_trotter = trotter @ psi0

    # Numerical integration
    h = 3 * np.kron(X, X) - 2 * np.kron(Y, Y) + np.kron(Z, Identity)
    psi_numerical = evolve(h, psi0, t0=0, tend=duration, time_steps=1000)

    ic(psi_numerical, psi_trotter, abs(psi_numerical) - abs(psi_trotter))
    np.testing.assert_almost_equal(psi_numerical, psi_trotter, decimal=3)

def test_trotter_two_gates_depending_on_time():
    """
    Por ejemplo hagamos aproximacion de e^{-i(cos(t) * X + 3t * Y)t}.
    """
    drive_0 = lambda t: jnp.cos(t)
    drive_1 = lambda t: 3 * (t + 50)
    n_trotter = 200
    step = duration / n_trotter
    t_vals = jnp.arange(0, duration, step) + step / 2
    h = lambda t: drive_0(t) * X + drive_1(t) * Y
    psi0 = jnp.array([1, 0])
    numerical = evolve(h, psi0, t0=0, tend=duration, time_steps=len(t_vals))
    driving_values_0 = integrate_ranges(drive_0, t_vals)
    driving_values_1 = integrate_ranges(drive_1, t_vals)

    trotter = Identity
    for i in range(n_trotter):
        trotter = trotter @ qml.exp(qml.ops.X(0), driving_values_0[i] / (2 * n_trotter) * (-1j)).matrix() @ \
                  qml.exp(qml.ops.Y(0), driving_values_1[i] * (-1j)).matrix() @ \
                  qml.exp(qml.ops.X(0), driving_values_0[i] / (2 * n_trotter) * (-1j)).matrix()
    trotterized = trotter @ psi0
    print()
    ic(numerical, trotterized, abs(numerical - trotterized))
    tolerance = 1e-2
    assert (abs(numerical - trotterized) < tolerance * jnp.ones(2)).all()

    # Numerical integration
if __name__ == '__main__':
    pytest.main()