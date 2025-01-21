from pennylane.pulse.transmon import AmplitudeAndPhaseAndFreq
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.wires import Wires
from pennylane.math import sin
from pennylane import X, dot, Hamiltonian, QubitUnitary, RX, RY
from .constants import hbar
from .utils.integration import integrate_ranges
from jax.scipy.linalg import expm
import numpy as np
from icecream import ic
from typing import Callable
from .utils.reorder_AmpPhaseFreq import _reorder_AmpPhaseFreq
from jax import numpy as jnp

def transmon_drive(amplitude: Callable, phase: float, freq: float, wires: int, d=2):
    r"""

    HE MODIFICADO PARA QUE EL DRIVING SEA EN X Y NO EN Y. EL RESTO LO HE DEJADO IGUAL,
    INCLUYENDO ESTA DOCUMENTACION DE ABAJO

    Returns a :class:`ParametrizedHamiltonian` representing the drive term of a transmon qubit.

    The Hamiltonian is given by

    .. math::

        \Omega(t) \sin\left(\phi(t) + \nu t\right) \sum_q Y_q

    where :math:`\{Y_q\}` are the Pauli-Y operators on ``wires`` :math:`\{q\}`.
    The arguments ``amplitude``, ``phase`` and ``freq`` correspond to :math:`\Omega / (2\pi)`, :math:`\phi`
    and :math:`\nu / (2\pi)`, respectively, and can all be either fixed numbers (``float``) or depend on time
    (``callable``). If they are time-dependent, they need to abide by the restrictions imposed
    in :class:`ParametrizedHamiltonian` and have a signature of two parameters, ``(params, t)``.

    Together with the qubit :math:`Z` terms in :func:`transmon_interaction`, driving with this term can generate
    :math:`X` and :math:`Y` rotations by setting :math:`\phi` accordingly and driving on resonance
    (see eqs. (79) - (92) in `1904.06560 <https://arxiv.org/abs/1904.06560>`_).
    Further, it can generate entangling gates by driving at cross-resonance with a coupled qubit
    (see eqs. (131) - (137) in `1904.06560 <https://arxiv.org/abs/1904.06560>`_).
    Such a coupling is described in :func:`transmon_interaction`.

    For realistic simulations, one may restrict the amplitude, phase and drive frequency parameters.
    For example, the authors in `2008.04302 <https://arxiv.org/abs/2008.04302>`_ impose the restrictions of
    a maximum amplitude :math:`\Omega_{\text{max}} = 20 \text{MHz}` and the carrier frequency to deviate at most
    :math:`\nu - \omega = \pm 1 \text{GHz}` from the qubit frequency :math:`\omega`
    (see :func:`~.transmon_interaction`).
    The phase :math:`\phi(t)` is typically a slowly changing function of time compared to :math:`\Omega(t)`.

    .. note:: Currently only supports ``d=2`` with qudit support planned in the future.
        For ``d>2``, we have :math:`Y \mapsto i (\sigma^- - \sigma^+)`
        with lowering and raising operators  :math:`\sigma^{\mp}`.

    .. note:: Due to convention in the respective fields, we omit the factor :math:`\frac{1}{2}` present in the related constructor :func:`~.rydberg_drive`

    .. seealso::

        :func:`~.rydberg_drive`, :func:`~.transmon_interaction`

    Args:
        amplitude (Union[float, callable]): Float or callable representing the amplitude of the driving field.
            This should be in units of frequency (GHz), and will be converted to angular frequency
            :math:`\Omega(t)` internally where needed, i.e. multiplied by :math:`2 \pi`.
        phase (Union[float, callable]): Float or callable returning phase :math:`\phi(t)` (in radians).
            Can be a fixed number (``float``) or depend on time (``callable``)
        freq (Union[float, callable]): Float or callable representing the frequency of the driving field.
            This should be in units of frequency (GHz), and will be converted to angular frequency
            :math:`\nu(t)` internally where needed, i.e. multiplied by :math:`2 \pi`.
        wires (Union[int, list[int]]): Label of the qubit that the drive acts upon. Can be a list of multiple wires.
        d (int): Local Hilbert space dimension. Defaults to ``d=2`` and is currently the only supported value.

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

    **Example**

    We can construct a drive term acting on qubit ``0`` in the following way. We parametrize the amplitude and phase
    via :math:`\Omega(t)/(2 \pi) = A \times \sin^2(\pi t)` and :math:`\phi(t) = \phi_0 (t - \frac{1}{2})`. The squared
    sine ensures that the amplitude will be strictly positive (a requirement for some hardware). For simplicity, we
    set the drive frequency to zero :math:`\nu=0`.

    .. code-block:: python3

        def amp(A, t):
            return A * jnp.exp(-t**2)

        def phase(phi0, t):
            return phi0

        freq = 0

        H = qml.pulse.transmon_drive(amp, phase, freq, 0)

        t = 0.
        A = 1.
        phi0 = jnp.pi/2
        params = [A, phi0]

    Evaluated at :math:`t = 0` with the parameters :math:`A = 1` and :math:`\phi_0 = \pi/2` we obtain
    :math:`2 \pi A \exp(0) \sin(\pi/2 + 0)\sigma^y = 2 \pi \sigma^y`.

    >>> H(params, t)
    6.283185307179586 * Y(0)

    We can combine ``transmon_drive()`` with :func:`~.transmon_interaction` to create a full driven transmon Hamiltonian.
    Let us look at a chain of three transmon qubits that are coupled with their direct neighbors. We provide all
    frequencies in GHz (conversion to angular frequency, i.e. multiplication by :math:`2 \pi`, is taken care of
    internally where needed).

    We use values around :math:`\omega = 5 \times 2\pi \text{GHz}` for resonant frequencies, and coupling strenghts
    on the order of around :math:`g = 0.01 \times 2\pi \text{GHz}`.

    We parametrize the drive Hamiltonians for the qubits with amplitudes as squared sinusodials of
    maximum amplitude :math:`A`, and constant drive frequencies of value :math:`\nu`. We set the
    phase to zero :math:`\phi=0`, and we make the parameters :math:`A` and :math:`\nu` trainable
    for every qubit. We simulate the evolution for a time window of :math:`[0, 5]\text{ns}`.

    .. code-block:: python3

        import jax

        jax.config.update("jax_enable_x64", True)

        qubit_freqs = [5.1, 5., 5.3]
        connections = [[0, 1], [1, 2]]  # qubits 0 and 1 are coupled, as are 1 and 2
        g = [0.02, 0.05]
        H = qml.pulse.transmon_interaction(qubit_freqs, connections, g, wires=range(3))

        def amp(max_amp, t): return max_amp * jnp.sin(t) ** 2
        freq = qml.pulse.constant  # Parametrized constant frequency
        phase = 0.0
        time = 5

        for q in range(3):
            H += qml.pulse.transmon_drive(amp, phase, freq, q)  # Parametrized drive for each qubit

        dev = qml.device("default.qubit", wires=range(3))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(H)(params, time)
            return qml.expval(qml.Z(0) + qml.Z(1) + qml.Z(2))

    We evaluate the Hamiltonian with some arbitrarily chosen maximum amplitudes (here on the order of :math:`0.5 \times 2\pi \text{GHz}`)
    and set the drive frequency equal to the qubit frequencies. Note how the order of the construction
    of ``H`` determines the order with which the parameters need to be passed to
    :class:`~.ParametrizedHamiltonian` and :func:`~.evolve`. We made the drive frequencies
    trainable parameters by providing constant callables through :func:`~.pulse.constant` instead of fixed values (like the phase).
    This allows us to differentiate with respect to both the maximum amplitudes and the frequencies and optimize them.

    >>> max_amp0, max_amp1, max_amp2 = [0.5, 0.3, 0.6]
    >>> fr0, fr1, fr2 = qubit_freqs
    >>> params = [max_amp0, fr0, max_amp1, fr1, max_amp2, fr2]
    >>> qnode(params)
    Array(-1.57851962, dtype=float64)
    >>> jax.grad(qnode)(params)
    [Array(-13.50193649, dtype=float64),
     Array(3.1112141, dtype=float64),
     Array(16.40286521, dtype=float64),
     Array(-4.30485667, dtype=float64),
     Array(4.75813949, dtype=float64),
     Array(3.43272354, dtype=float64)]

    """
    if d != 2:
        raise NotImplementedError(
            "Currently only supports qubits (d=2). Qutrits and qudits support is planned in the future."
        )

    wires = Wires(wires)

    # TODO: use creation and annihilation operators when introducing qutrits
    # Note that exp(-iw)a* + exp(iw)a = cos(w)X - sin(w)Y for a=1/2(X+iY)
    # We compute the `coeffs` and `observables` of the EM field

    # if callable(amplitude):
    #     coeffs = (np.array([phase, freq]) * hbar).tolist()
    #     coeffs = [AmplitudeAndPhaseAndFreq(sin, amplitude, *coeffs)]
    # else:
    #     coeffs = (np.array([amplitude, phase, freq]) * hbar).tolist()
    #     coeffs = [AmplitudeAndPhaseAndFreq(sin, *coeffs)]

    coeffs = [AmplitudeAndPhaseAndFreq(sin, amplitude, phase, freq)]
    drive_x_term = sum(X(wire) for wire in wires)

    observables = [drive_x_term]

    pulses = [HardwarePulse(amplitude, phase, freq, wires)]
    return HardwareHamiltonian(coeffs, observables, pulses=pulses, reorder_fn=_reorder_AmpPhaseFreq)




def vz_rotation(angle: float, wire: int = None, phases: list = None):
    """
    Applies a virtual z gate to qubit wire. In place operations for phases.
    """
    if isinstance(phases, (list, np.ndarray)):
        phases[wire] += angle
    elif isinstance(phases, jnp.ndarray):
        phases = phases.at[wire].add(angle)
    elif isinstance(phases, (float, int)):
        phases += angle
    else:
        raise ValueError("Invalid phase type. Expected float or jax.numpy.ndarray.")
    return phases


def pulse1q(q_freq: float,
            drive_freq: float,
            drive_phase: float,
            amplitude_func: callable,
            duration: float,
            wire: int,
            n_trotter: int = 10,
            t_start: float = 0) -> None:
    """
    se aplica el hamiltoniano de la ecuacion (91) del quantum engineers paper con delta_w = 0
    H = -A * shape(t) [[0, exp{1j * (delta_w * t + phase)}]
                       [exp{-1j * (delta_w * t + phase)}, 0]]
    Entonces:
    exp{-1j * H * t} |ψ> = Π_i (exp{1j * A * ∫_[t1,t2] shape(τ) [[0, exp{1j * (delta_w * τ + phase)}]  dτ)  |ψ>
                               (                                [exp{-1j * (delta_w * τ + phase)}, 0]]}  )
    """
    step = duration / n_trotter

    t_vals = jnp.arange(t_start, t_start + duration, step)
    if not np.isclose(q_freq, drive_freq):
        raise ValueError(f"Mismatch in qubit frequency and drive frequency: {q_freq} GHz vs {drive_freq} GHz")
    integrando_x = lambda t:   amplitude_func(t) * jnp.cos((t + drive_phase))
    integrando_y = lambda t: -2 * amplitude_func(t) * jnp.sin((t + drive_phase))
    integration_x = integrate_ranges(integrando_x, t_vals)
    integration_y = integrate_ranges(integrando_y, t_vals)

    for n in range(n_trotter-1):
        RX(integration_x[n], wires=wire)
        RY(integration_y[n], wires=wire)
        RX(integration_x[n], wires=wire)