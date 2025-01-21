from jax import numpy as jnp
import numpy as np
import pennylane as qml
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import unittest
from icecream import ic
from src.Pennypulse.test.utils import evolve, X, Y, Z, Identity
from src.Pennypulse.src.pennypulse import transmon_trotter_suzuki_2q_drive1q
from src.Pennypulse.src.pennypulse import transmon_interaction, transmon_drive
from src.Pennypulse.src.pennypulse.utils.integration import integrate_ranges


class TestTransmonTrotter(unittest.TestCase):

    def test_transmon_trotter_suzuki_2q_drive1q_runs_without_error(self):


        q_freqs = [1.3, 2.1]
        coupling = 0.1
        ampl_0 = 0.5
        driving = lambda t: ampl_0 * t
        phase = np.pi / 3

        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit():
            transmon_trotter_suzuki_2q_drive1q(q_freqs, coupling, driving, q_freqs[0], phase, wire=0, n_trotter=5)
            return [qml.expval(qml.Z(0)), qml.expval(qml.Z(1))]

        try:

            print(qml.draw(circuit)())
            result = circuit()

            for r in result:
                self.assertGreaterEqual(r, -1)
                self.assertLessEqual(r, 1)
        except Exception as e:
            self.fail(f"Exception was raised_ {e}")


    def test_trotter_suzuki_different_n_trotter_same_result(self):
        q_freqs = [1.3, 2.1]
        coupling = 0.01
        ampl_0 = 50
        driving = lambda t:  ampl_0 * t      # for instance
        phase = np.pi / 3
        drive_wire = 0
        duration = 50

        # TROTTER EVOLUTION
        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit_trotter(n):
            transmon_trotter_suzuki_2q_drive1q(q_freqs, coupling, driving, q_freqs[0], phase, wire=drive_wire,
                                                    duration=duration, n_trotter=n)
            return [qml.expval(qml.Z(0)), qml.expval(qml.Z(1))]

        result_trotter_50 = np.array(circuit_trotter(50))
        result_trotter_60 = np.array(circuit_trotter(60))

        ic(result_trotter_50, result_trotter_60, abs(result_trotter_50 - result_trotter_60))
        self.assertTrue((abs(result_trotter_50 - result_trotter_60) < 5e-2).all())


    def test_trotter_suzuki_vs_exact_evolution_pennylane(self):
        q_freqs = [1.3, 2.1]
        coupling = 0.01
        ampl_0 = 0.5
        driving_with_params = lambda ampl, t: ampl * t  # random driving
        driving = lambda t: driving_with_params(ampl_0, t)
        phase = np.pi / 3
        drive_wire = 0
        duration = 20
        n_trotter = 50

        # TROTTER EVOLUTION
        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit_trotter():
            transmon_trotter_suzuki_2q_drive1q(q_freqs, coupling, driving, q_freqs[0], phase, wire=drive_wire,
                                                    duration=duration, n_trotter=n_trotter)
            return [qml.expval(qml.Z(0)), qml.expval(qml.Z(1))]

        # EXACT EVOLUTION

        base_ham = transmon_interaction(q_freqs, range(2), coupling, [[0, 1]])
        h = base_ham + \
            transmon_drive(amplitude=driving_with_params,
                           phase=phase,
                           freq=q_freqs[0],
                           wires=drive_wire)

        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit_exact():
            qml.evolve(h)(params=[ampl_0], t=duration)
            return [qml.expval(qml.Z(0)), qml.expval(qml.Z(1))]

        result_trotter = np.array(circuit_trotter())
        result_exact = np.array(circuit_exact())

        ic(result_trotter, result_exact, abs(result_trotter - result_exact))
        self.assertTrue((abs(result_trotter - result_exact) < 5e-2).all())


    def test_trotter_suzuki_vs_exact_evolution_numerical_integration_with_qml(self):
        q_freqs = [1.3, 2.1]
        coupling = 0.01
        ampl_0 = 10
        driving_with_params = lambda ampl, t: ampl * t  # random driving
        driving = lambda t: driving_with_params(ampl_0, t)
        phase = np.pi / 3
        drive_wire = 0
        duration = 0.5

        # PENNYLANE EVOLUTION
        base_ham = transmon_interaction(q_freqs, range(2), coupling, [[0, 1]])
        h_qml = base_ham + \
            transmon_drive(amplitude=driving_with_params,
                           phase=phase,
                           freq=q_freqs[0],
                           wires=drive_wire)

        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit_exact():
            qml.evolve(h_qml)(params=[ampl_0], t=duration)
            return [qml.expval(qml.Z(0)), qml.expval(qml.Z(1))]


        # EXACT EVOLUTION
        h_indep = (-q_freqs[0] / 2 * np.kron(Z, Identity) - q_freqs[1] / 2 * np.kron(Identity, Z) +
                   coupling * (np.kron(X, X) + np.kron(Y, Y)))
        h = lambda t: h_indep + driving(t) * np.sin(q_freqs[1] * t + phase) * np.kron(X, Identity)
        psi_0 = np.array([1, 0, 0, 0])
        psi_end = evolve(h, psi_0, t0=0, tend=duration, time_steps=10000)

        tol = 0.01
        assert (np.linalg.norm(abs(psi_end)) - 1) < tol
        exp_z_0 = np.conj(psi_end).T @ np.kron(Z, Identity) @ psi_end
        exp_z_1 = np.conj(psi_end).T @ np.kron(Identity, Z) @ psi_end

        # Compare results
        result_qml = np.array(circuit_exact())
        result_exact = np.array([exp_z_0, exp_z_1])

        ic(result_qml, result_exact, abs(result_qml - result_exact))
        self.assertTrue((abs(result_qml - result_exact) < 5e-2).all())

    def test_numerical_integration_scipy_vs_numerical_integration_with_qml(self):
        try:
            q_freqs = [1.3, 2.1]
            coupling = 0.01
            ampl_0 = 10
            driving_with_params = lambda ampl, t: ampl * t  # random driving
            driving = lambda t: driving_with_params(ampl_0, t)
            phase = np.pi / 3
            drive_wire = 0
            duration = 0.5

            # PENNYLANE EVOLUTION
            base_ham = transmon_interaction(q_freqs, range(2), coupling, [[0, 1]])
            h_qml = base_ham + \
                transmon_drive(amplitude=driving_with_params,
                               phase=phase,
                               freq=q_freqs[0],
                               wires=drive_wire)

            @qml.qnode(qml.device('default.qubit', wires=2))
            def circuit_exact():
                qml.evolve(h_qml)(params=[ampl_0], t=duration)
                return qml.state()
            result_qml = np.array(circuit_exact())


            # EVOLUTION WITH SCIPY INTEGRATE PACKAGE
            h_indep = - q_freqs[0] * np.kron(Z, Identity) / 2 - q_freqs[1] * np.kron(Identity, Z) / 2 \
                      + coupling * (np.kron(X, X) + np.kron(Y, Y))
            h = lambda t: h_indep + driving(t) * np.sin(q_freqs[1] * t + phase) * np.kron(X, Identity)
            psi_0 = np.array([1, 0, 0, 0])
            result_exact = evolve(h, psi_0, t0=0, tend=duration, time_steps=5000)

            ic(result_qml, result_exact, abs(result_qml - result_exact))
            self.assertTrue((abs(result_qml - result_exact) < 5e-2).all())
            raise ValueError("Pennylane integration is correct")
        except AssertionError as e:
            print(f"Pennylane integration is not correct")
            self.assertTrue(True)

    def test_trotter_vs_exact_evolution_numerical_integration(self):
        q_freqs = [1.3, 2.1]
        coupling = 0.01
        ampl_0 = 50
        driving_with_params = lambda ampl, t: ampl * t  # random driving
        driving = lambda t: driving_with_params(ampl_0, t)
        phase = np.pi / 3
        duration = 0.5
        n_trotter = 500

        # TROTTER EVOLUTION
        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit_trotter():
            transmon_trotter_suzuki_2q_drive1q(q_freqs=q_freqs, coupling=coupling, amplitude_func=driving,
                                                    drive_freq=q_freqs[1], drive_phase=phase, wire=0,
                                                    duration=duration, n_trotter=n_trotter, t_start=0)
            return qml.state()
        result_trotter = np.array(circuit_trotter())

        # EXACT EVOLUTION
        h_indep = - q_freqs[0] * np.kron(Z, Identity) / 2 - q_freqs[1] * np.kron(Identity, Z) / 2 \
                    + coupling * (np.kron(X, X) + np.kron(Y, Y))
        h = lambda t: h_indep + driving(t) * np.sin(q_freqs[1] * t + phase) * np.kron(X, Identity)
        psi_0 = np.array([1, 0, 0, 0])
        result_exact = evolve(h, psi_0, t0=0, tend=duration, time_steps=5000)

        ic(result_trotter, result_exact, abs(result_trotter - result_exact))
        self.assertTrue((abs(result_trotter - result_exact) < 5e-2).all())

    def test_trotter_vs_exact_evolution_numerical_integration_no_driving(self):
        q_freqs = [1.3, 2.1]
        coupling = 0.01
        ampl_0 = 50
        driving = lambda t: 0
        phase = np.pi / 3
        duration = 0.5
        n_trotter = 500

        # TROTTER EVOLUTION
        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit_trotter():
            transmon_trotter_suzuki_2q_drive1q(q_freqs=q_freqs, coupling=coupling, amplitude_func=driving,
                                                    drive_freq=q_freqs[1], drive_phase=phase, wire=0,
                                                    duration=duration, n_trotter=n_trotter, t_start=0)
            return qml.state()
        result_trotter = np.array(circuit_trotter())

        # EXACT EVOLUTION
        h_indep = - q_freqs[0] * np.kron(Z, Identity) / 2 - q_freqs[1] * np.kron(Identity, Z) / 2 \
                    + coupling * (np.kron(X, X) + np.kron(Y, Y))
        h = lambda t: h_indep + driving(t) * np.sin(q_freqs[1] * t + phase) * np.kron(X, Identity)
        psi_0 = np.array([1, 0, 0, 0])
        result_exact = evolve(h, psi_0, t0=0, tend=duration, time_steps=5000)

        ic(result_trotter, result_exact, abs(result_trotter - result_exact))
        self.assertTrue((abs(result_trotter - result_exact) < 5e-2).all())


    def test_qubit_h_vs_gate(self):
        """
        To see if I am correctly doing the trotter step corresponding to:
        H = - (q_freqs[0] / 2 * np.kron(Z, Identity) - q_freqs[1] / 2 * np.kron(Identity, Z))
        e^{-iHΔt/2}
        """
        q_freqs = [1.3, 2.1]
        delta = 1

        # exact hamiltonian
        H = - q_freqs[0] * np.kron(Z, Identity) / 2 - q_freqs[1] * np.kron(Identity, Z) / 2
        U_exact = expm(-1j * delta * H / 2)

        # Rotation
        U_rotation = (qml.RZ(-q_freqs[0] * delta / 2, wires=0) @ qml.RZ(-q_freqs[1] * delta / 2, wires=1)).matrix()

        # Compare
        ic(U_exact, U_rotation)
        self.assertTrue(np.allclose(U_exact, U_rotation, atol=1e-5))

    def test_coupling_h_vs_gate(self):
        """
        To see if I am correctly doing the trotter step corresponding to:
        H = J (XX + YY)
        e^{-iHΔt/2}
        """
        J = 2
        delta = 1

        # exact hamiltonian
        H = J * (np.kron(X, X) + np.kron(Y, Y))
        U_exact = expm(-1j * delta * H / 2)

        # Rotation
        U_rotation = qml.Identity(0) @ qml.Identity(1)
        # XX interaction
        U_rotation = U_rotation @ qml.Hadamard(wires=0) @ qml.Hadamard(wires=1)
        U_rotation = U_rotation @ qml.MultiRZ(J * delta, wires=[0, 1])
        U_rotation = U_rotation @ qml.Hadamard(wires=0) @ qml.Hadamard(wires=1)
        # YY interaction
        U_rotation = U_rotation @ qml.S(wires=0) @ qml.S(wires=1)
        U_rotation = U_rotation @ qml.Hadamard(wires=0) @ qml.Hadamard(wires=1)
        U_rotation = U_rotation @ qml.MultiRZ(J * delta, wires=[0, 1])
        U_rotation = U_rotation @ qml.Hadamard(wires=0) @ qml.Hadamard(wires=1)
        U_rotation = U_rotation @ qml.adjoint(qml.S)(wires=0) @ qml.adjoint(qml.S)(wires=1)
        # Compare
        U_rotation = np.round(U_rotation.matrix(), 4)
        ic(U_exact, U_rotation)
        self.assertTrue(np.allclose(U_exact, U_rotation, atol=1e-3))

    def test_drive_h_vs_gate(self):
        """
        To see if I am correctly doing the trotter step corresponding to:
        H = ∫_Δt( amplitude(t) * sin(w_d * t + phase) dt) * X0
        e^{-iHΔt}
        """
        drive_freq = 2
        amplitude_function = lambda t: 1e-2 * jnp.exp(4 * t)
        phase = np.pi / 3
        delta = 3


        n_trotter = 1000
        step = delta / n_trotter
        t_vals = jnp.arange(0, delta, step)

        # exact hamiltonian approximated (as little rotations separated in n_points)
        U_exact = np.eye(4)
        amplitude_with_sin = lambda t: amplitude_function(t) * jnp.sin(drive_freq * t + phase)
        amp_with_sin_integration = integrate_ranges(amplitude_with_sin, t_vals)

        for i in range(n_trotter - 1):
            U_exact = U_exact @ expm(-1j * amp_with_sin_integration[i] * jnp.kron(X, Identity))

        # Rotation
        U_rotation = np.eye(4)
        for i in range(n_trotter - 1):
            U_rotation = U_rotation @ np.kron(qml.RX(2 * amp_with_sin_integration[i], wires=0).matrix(), Identity)

        # Compare
        ic(U_exact, U_rotation)
        self.assertTrue(np.allclose(U_exact, U_rotation, atol=1e-5))




if __name__ == '__main__':
    unittest.main()
