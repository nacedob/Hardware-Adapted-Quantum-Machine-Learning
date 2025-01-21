from typing import Optional, Sequence
from .constants import hbar
from typing import Union
from pennylane.pulse.transmon import TransmonSettings, callable_freq_to_angular
from pennylane.pulse import HardwareHamiltonian
from pennylane.wires import Wires
from pennylane import numpy as np
from pennylane import Identity, X, Y, Z, dot
from .utils.reorder_AmpPhaseFreq import _reorder_AmpPhaseFreq
from pennylane.math import ndim
import warnings



def transmon_interaction(
        qubit_freq: Union[float, list],
        wires: Union[Sequence, int],
        coupling: Optional[Union[float, list]] = None,
        connections: Optional[list] = None,
        anharmonicity=None,
        d=2,
):
    r"""

    HE MODIFICADO PARA QUE EL NO SE HAGAN CALCULOS CON OPERADORES ESCALERA Y QUE SE
    PONGA DIRECTAMENTE. ADEMAS LA INTERACCION ENTRE HAMILTONIANOS ES DIRECTAMENTE YY,
    NO XX + YY.
    ADEMAS, PERMITO QUE SOLO HAYA 1 QUBIT
    EL RESTO LO HE DEJADO IGUAL,INCLUYENDO ESTA DOCUMENTACION DE ABAJO

    Returns a :class:`ParametrizedHamiltonian` representing the circuit QED Hamiltonian of a
    superconducting transmon system.

    The Hamiltonian is given by

    .. math::

        H = \sum_{q\in \text{wires}} \omega_q b^\dagger_q b_q
        + \sum_{(i, j) \in \mathcal{C}} g_{ij} \left(b^\dagger_i b_j + b_j^\dagger b_i \right)
        + \sum_{q\in \text{wires}} \alpha_q b^\dagger_q b^\dagger_q b_q b_q

    where :math:`[b_p, b_q^\dagger] = \delta_{pq}` are creation and annihilation operators.
    The first term describes the effect of the dressed qubit frequencies ``qubit_freq`` :math:`= \omega_q/ (2\pi)`,
    the second term their ``coupling`` :math:`= g_{ij}/(2\pi)` and the last the
    ``anharmonicity`` :math:`= \alpha_q/(2\pi)`, which all can vary for
    different qubits. In practice, these operators are restricted to a finite dimension of the
    local Hilbert space (default ``d=2`` corresponds to qubits).
    In that case, the anharmonicity is set to :math:`\alpha=0` and ignored.

    The values of :math:`\omega` and :math:`\alpha` are typically around :math:`5 \times 2\pi \text{GHz}`
    and :math:`0.3 \times 2\pi \text{GHz}`, respectively.
    It is common for different qubits to be out of tune with different energy gaps. The coupling strength
    :math:`g` typically varies between :math:`[0.001, 0.1] \times 2\pi \text{GHz}`. For some example parameters,
    see e.g. `arXiv:1804.04073 <https://arxiv.org/abs/1804.04073>`_,
    `arXiv:2203.06818 <https://arxiv.org/abs/2203.06818>`_, or `arXiv:2210.15812 <https://arxiv.org/abs/2210.15812>`_.

    .. note:: Currently only supporting ``d=2`` with qudit support planned in the future. For ``d=2``, we have :math:`b:=\frac{1}{2}(\sigma^x + i \sigma^y)`.

    .. seealso::

        :func:`~.transmon_drive`

    Args:
        qubit_freq (Union[float, list[float], Callable]): List of dressed qubit frequencies. This should be in units
            of frequency (GHz), and will be converted to angular frequency :math:`\omega` internally where
            needed, i.e. multiplied by :math:`2 \pi`. When passing a single float all qubits are assumed to
            have that same frequency. When passing a parametrized function, it must have two
            arguments, the first one being the trainable parameters and the second one being time.
        connections (list[tuple(int)]): List of connections ``(i, j)`` between qubits i and j.
            When the wires in ``connections`` are not contained in ``wires``, a warning is raised.
        coupling (Union[float, list[float]]): List of coupling strengths. This should be in units
            of frequency (GHz), and will be converted to angular frequency internally where
            needed, i.e. multiplied by :math:`2 \pi`. Needs to match the length of ``connections``.
            When passing a single float need explicit ``wires``.
        anharmonicity (Union[float, list[float]]): List of anharmonicities. This should be in units
            of frequency (GHz), and will be converted to angular frequency internally where
            needed, i.e. multiplied by :math:`2 \pi`. Ignored when ``d=2``.
            When passing a single float all qubits are assumed to have that same anharmonicity.
        wires (list): Needs to be of the same length as qubit_freq. Note that there can be additional
            wires in the resulting operator from the ``connections``, which are treated independently.
        d (int): Local Hilbert space dimension. Defaults to ``d=2`` and is currently the only supported value.

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

    **Example**

    We can set up the transmon interaction Hamiltonian with uniform coefficients by passing ``float`` values.

    .. code-block::

        connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
        H = qml.pulse.transmon_interaction(qubit_freq=0.5, connections=connections, coupling=1., wires=range(6))

    The resulting :class:`~.HardwareHamiltonian:` consists of ``4`` coupling terms and ``6`` qubits
    because there are six different wire indices in ``connections``.

    >>> print(H)
    HardwareHamiltonian: terms=10

    We can also provide individual values for each of the qubit energies and coupling strengths,
    here of order :math:`0.1 \times 2\pi\text{GHz}` and :math:`1 \times 2\pi\text{GHz}`, respectively.

    .. code-block::

        qubit_freqs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.]
        couplings= [1., 2., 3., 4.]
        H = qml.pulse.transmon_interaction(qubit_freq=qubit_freqs,
                                           connections=connections,
                                           coupling=couplings,
                                           wires=range(6))

    The interaction term is dependent only on the typically fixed transmon energies and coupling strengths.
    Executing this as a pulse program via :func:`~.evolve` would correspond to all driving fields being turned off.
    To add a driving field, see :func:`~.transmon_drive`.

    """
    if coupling is not None and connections is None:
        raise ValueError("Must provide connections if coupling is not provided.")
    if connections is not None and coupling is None:
        raise ValueError("Must provide coupling if connections is not provided.")

    if d != 2:
        raise NotImplementedError(
            "Currently only supporting qubits. Qutrits and qudits are planned in the future."
        )

    # if wires is None and qml.math.ndim(omega) == 0:
    #     raise ValueError(
    #         f"Cannot instantiate wires automatically. Either need specific wires or a list of omega."
    #         f"Received wires {wires} and omega of type {type(omega)}"
    #     )

    # wires = wires or list(range(len(omega)))

    if isinstance(wires, int):
        wires = [wires]

    n_wires = len(wires)

    if n_wires > 1 and coupling is None and connections is None:
        raise ValueError(
            "Must provide either connections or coupling when there are multiple wires."
        )
    if connections is not None:
        if not Wires(wires).contains_wires(Wires(np.unique(connections).tolist())):
            warnings.warn(
                f"Caution, wires and connections do not match. "
                f"I.e., wires in connections {connections} are not contained in the wires {wires}"
            )

    # Prepare coefficients
    if anharmonicity is None:
        anharmonicity = [0.0] * n_wires

    # TODO: make coefficients callable / trainable. Currently not supported
    if callable(qubit_freq) or ndim(qubit_freq) == 0:
        qubit_freq = [qubit_freq] * n_wires
    elif len(qubit_freq) != n_wires:
        raise ValueError(
            f"Number of qubit frequencies in {qubit_freq} does not match the provided wires = {wires}"
        )

    if ndim(coupling) == 0 and coupling is not None:
        coupling = [coupling] * len(connections)
    if (coupling is not None or connections is not None):
        if len(coupling) != len(connections):
            raise ValueError(
                f"Number of coupling terms {coupling} does not match the provided connections = {connections}"
            )

    settings = TransmonSettings(connections, qubit_freq, coupling, anharmonicity=anharmonicity)

    omega = [callable_freq_to_angular(f) if callable(f) else (2 * np.pi * f) for f in qubit_freq]
    if coupling is not None:
        g = [callable_freq_to_angular(c) if callable(c) else (2 * np.pi * c) for c in coupling]
    else:
        g = None

    # qubit term
    coeffs = list(omega)
    observables = [0.5 * (Identity(i) - Z(i)) for i in wires]   # [ad(i) @ a(i) for i in wires]

    # coupling term
    if connections is not None or g is not None:
        coeffs += list(g)
        observables += [X(i) @ X(j) + Y(i) @ Y(j) for i, j in connections]
                        # [ad(i) @ a(j) + ad(j) @ a(i) for (i, j) in connections]

    # coeffs = (np.array(coeffs) * hbar).tolist()   #TODO: esto deberia o no estar?


    return HardwareHamiltonian(
        coeffs, observables, settings=settings, reorder_fn=_reorder_AmpPhaseFreq
    )

########### MANUALLY DEFINED HAMILTONIAN

def a(wire):
    return 0.5 * X(wire) + 0.5j * Y(wire)

def ad(wire):
    return 0.5 * X(wire) - 0.5j * Y(wire)

def transmon_manually(freqs: [float, Sequence], coupling: Sequence=None):
    nWires = 1 if isinstance(freqs, (int, float)) else len(freqs)
    if nWires > 1 and coupling is None:
        raise ValueError("Must provide coupling if there are multiple wires.")
    H_base = dot(freqs, [ad(i) @ a(i) for i in range(nWires)])
    if nWires > 1:
        H_base += dot(
            coupling,
            [ad(i) @ a((i + 1) % nWires) + ad((i + 1) % nWires) @ a(i) for i in range(nWires)],
        )
    return H_base



