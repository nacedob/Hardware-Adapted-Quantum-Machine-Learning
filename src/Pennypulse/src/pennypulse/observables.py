
import warnings
from collections.abc import Sequence
from copy import copy
from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike
from pennylane.ops.qubit import QubitUnitary


class Hermitian(Observable):
    r"""
    _Comentarios Nacho_: La diferencia con la clase original esta en que he hecho que no se validen
    los inputs. Falla cuando se usa con jax por hacer comparaciones. Asi lo que tenga que ver con esto
    esta comentado


    An arbitrary Hermitian observable.

    For a Hermitian matrix :math:`A`, the expectation command returns the value

    .. math::
        \braket{A} = \braketT{\psi}{\cdots \otimes I\otimes A\otimes I\cdots}{\psi}

    where :math:`A` acts on the requested wires.

    If acting on :math:`N` wires, then the matrix :math:`A` must be of size
    :math:`2^N\times 2^N`.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        A (array or Sequence): square hermitian matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "F"

    # Qubit case
    _num_basis_states = 2
    _eigs = {}

    def __init__(self, A: TensorLike, wires: WiresLike, id: Optional[str] = None):
        A = np.array(A) if isinstance(A, list) else A
        if not qml.math.is_abstract(A):
            if isinstance(wires, Sequence) and not isinstance(wires, str):
                if len(wires) == 0:
                    raise ValueError(
                        "Hermitian: wrong number of wires. At least one wire has to be given."
                    )
                expected_mx_shape = self._num_basis_states ** len(wires)
            else:
                # Assumably wires is an int; further validation checks are performed by calling super().__init__
                expected_mx_shape = self._num_basis_states

            # Hermitian._validate_input(A, expected_mx_shape)

        super().__init__(A, wires=wires, id=id)

    # @staticmethod
    # def _validate_input(A: TensorLike, expected_mx_shape: Optional[int] = None):
    #     """Validate the input matrix."""
    #     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
    #         raise ValueError("Observable must be a square matrix.")
    #
    #     if expected_mx_shape is not None and A.shape[0] != expected_mx_shape:
    #         raise ValueError(
    #             f"Expected input matrix to have shape {expected_mx_shape}x{expected_mx_shape}, but "
    #             f"a matrix with shape {A.shape[0]}x{A.shape[0]} was passed."
    #         )
    #
    #     # if not qml.math.allclose(A, qml.math.T(qml.math.conj(A))):
    #     #     raise ValueError("Observable must be Hermitian.")

    def label(
        self,
        decimals: Optional[int] = None,
        base_label: Optional[str] = None,
        cache: Optional[dict] = None,
    ) -> str:
        return super().label(decimals=decimals, base_label=base_label or "𝓗", cache=cache)

    @staticmethod
    def compute_matrix(A: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Hermitian.matrix`

        Args:
            A (tensor_like): hermitian matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> qml.Hermitian.compute_matrix(A)
        [[ 6.+0.j  1.-2.j]
         [ 1.+2.j -1.+0.j]]
        """
        A = qml.math.asarray(A)
        # Hermitian._validate_input(A)
        return A

    @property
    def eigendecomposition(self) -> dict[str, TensorLike]:
        """Return the eigendecomposition of the matrix specified by the Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the eigenvectors of the Hermitian observable
        """
        Hmat = self.matrix()
        Hmat = qml.math.to_numpy(Hmat)
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in Hermitian._eigs:
            w, U = np.linalg.eigh(Hmat)
            Hermitian._eigs[Hkey] = {"eigvec": U, "eigval": w}

        return Hermitian._eigs[Hkey]

    def eigvals(self) -> TensorLike:
        """Return the eigenvalues of the specified Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the Hermitian observable
        """
        return self.eigendecomposition["eigval"]

    @staticmethod
    def compute_decomposition(A, wires):  # pylint: disable=arguments-differ
        r"""Decomposes a hermitian matrix as a sum of Pauli operators.

        Args:
            A (array or Sequence): hermitian matrix
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: decomposition of the hermitian matrix

        **Examples**

        >>> op = qml.X(0) + qml.Y(1) + 2 * qml.X(0) @ qml.Z(3)
        >>> op_matrix = qml.matrix(op)
        >>> qml.Hermitian.compute_decomposition(op_matrix, wires=['a', 'b', 'aux'])
        [(
              1.0 * (I('a') @ Y('b') @ I('aux'))
            + 1.0 * (X('a') @ I('b') @ I('aux'))
            + 2.0 * (X('a') @ I('b') @ Z('aux'))
        )]
        >>> op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        >>> qml.Hermitian.compute_decomposition(op, wires=0)
        [(
              0.7071067811865475 * X(0)
            + 0.7071067811865475 * Z(0)
        )]

        """
        A = qml.math.asarray(A)

        if isinstance(wires, (int, str)):
            wires = Wires(wires)

        if len(wires) == 0:
            raise ValueError("Hermitian: wrong number of wires. At least one wire has to be given.")
        # Hermitian._validate_input(A, expected_mx_shape=2 ** len(wires))

        # determined heuristically from test_hermitian_decomposition_performance
        if len(wires) > 7:
            warnings.warn(
                "Decomposition may be inefficient for this large of a matrix.",
                UserWarning,
            )

        return [qml.pauli.conversion.pauli_decompose(A, wire_order=wires, pauli=False)]

    @staticmethod
    def compute_diagonalizing_gates(  # pylint: disable=arguments-differ
        eigenvectors: TensorLike, wires: WiresLike
    ) -> list["qml.operation.Operator"]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Hermitian.diagonalizing_gates`.

        Args:
            eigenvectors (array): eigenvectors of the operator, as extracted from op.eigendecomposition["eigvec"].
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> A = np.array([[-6, 2 + 1j], [2 - 1j, 0]])
        >>> _, evecs = np.linalg.eigh(A)
        >>> qml.Hermitian.compute_diagonalizing_gates(evecs, wires=[0])
        [QubitUnitary(tensor([[-0.94915323-0.j,  0.2815786 +0.1407893j ],
                              [ 0.31481445-0.j,  0.84894846+0.42447423j]], requires_grad=True), wires=[0])]

        """
        return [QubitUnitary(eigenvectors.conj().T, wires=wires)]

    def diagonalizing_gates(self) -> list["qml.operation.Operator"]:
        """Return the gate set that diagonalizes a circuit according to the
        specified Hermitian observable.

        Returns:
            list: list containing the gates diagonalizing the Hermitian observable
        """
        # note: compute_diagonalizing_gates has a custom signature, which is why we overwrite this method
        return self.compute_diagonalizing_gates(self.eigendecomposition["eigvec"], self.wires)
