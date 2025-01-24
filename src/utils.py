import inspect
import json
from jax import numpy as jnp
from pennylane import numpy as np
from numpy import savetxt
from scipy.integrate import solve_ivp
import os
import pickle
from icecream import ic
import optax

### OPTIMIZERS

pickle_extension = 'pkl'

DEFAULT_ADAM_PARAMS = {'learning_rate': 0.05,
                       'beta1': 0.9,
                       'beta2': 0.999}


def get_optimizer(nEpochs, lr, lrBoundaries, beta1, beta2) -> optax.adam:
    if isinstance(lr, list):
        create_schedule = len(lr) > 1
        if create_schedule:
            if lrBoundaries is None:
                lrBoundaries = jnp.linspace(0, nEpochs, len(lr))[1:]
            elif isinstance(lrBoundaries, list):
                if len(lrBoundaries) != len(lr) - 1:
                    raise Exception('lr and lrBoundaries lengths does not match')
            schedule = optax.join_schedules([optax.constant_schedule(lr) for lr in lr],
                                            boundaries=lrBoundaries)
        else:
            schedule = lr[0]
    else:
        schedule = lr

    optimizer = optax.adam(learning_rate=schedule,
                           b1=beta1,
                           b2=beta2
                           )
    return optimizer


#### PROCESS POINTS
def increase_dimensions(dataset, new_dimension, interface='jax'):
    point_dim = dataset.shape[1]
    if point_dim == new_dimension:
        return dataset
    elif point_dim > new_dimension:
        raise ValueError('Cannot reduce dimensionality')
    if interface == 'jax':
        dataset_ = jnp.concatenate([dataset, jnp.zeros((dataset.shape[0], new_dimension - point_dim))], axis=1)
    elif interface == 'pennylane':
        dataset_ = np.concatenate([dataset, np.zeros((dataset.shape[0], new_dimension - point_dim))], axis=1)
    else:
        raise ValueError('Invalid interface')
    return dataset_


#### OPTIMIZATION

def iterate_minibatches(inputs, targets, batch_size):
    """
    A generator for batches of the input data

    Args:
        inputs (array[float]): input data
        targets (array[float]): targets

    Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
    """
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]


# trace qubit
def compute_density_matrix(amplitudes: jnp.ndarray) -> jnp.ndarray:
    """
    Given a two-qubit state represented by amplitudes, this function returns the full density matrix
    (outer product of the state with itself).

    Parameters:
    - amplitudes (jnp.ndarray): An array of the amplitudes for each basis element of the two-qubit state.

    Returns:
    - jnp.ndarray: The full density matrix (outer product of the state with itself).
    """
    # Convert amplitudes into a column vector and calculate the full density matrix
    psi = amplitudes.reshape((-1, 1))  # Convert amplitudes into a column vector
    rho = jnp.dot(psi, psi.T.conj())  # Full density matrix (outer product)
    return rho



def trace_out_dm(rho, remaining_qubit):
    """
    Compute the partial trace of a 4x4 density matrix to get the subsystem of the specified qubit.

    Parameters:
        rho (jax.numpy.ndarray): A 4x4 density matrix.
        remaining_qubit (int): The qubit to keep (0 or 1).

    Returns:
        jax.numpy.ndarray: A 2x2 reduced density matrix for the specified qubit.
    """
    if rho.shape != (4, 4):
        raise ValueError("Input density matrix must have shape (4, 4).")

    if remaining_qubit not in [0, 1]:
        raise ValueError("remaining_qubit must be 0 or 1.")

    # Basis indices for a 2-qubit system: |00>, |01>, |10>, |11>
    # Reshape the matrix to a (2, 2, 2, 2) tensor
    rho_tensor = rho.reshape(2, 2, 2, 2)

    if remaining_qubit == 0:
        # Perform the partial trace over qubit 1 (axes 1 and 3)
        rho_reduced = jnp.trace(rho_tensor, axis1=1, axis2=3)
    else:
        # Perform the partial trace over qubit 0 (axes 0 and 2)
        rho_reduced = jnp.trace(rho_tensor, axis1=0, axis2=2)

    return rho_reduced


def trace_out_state(amplitudes: jnp.ndarray, remaining_wire: int) -> jnp.ndarray:
    """
    Given a two-qubit state represented by amplitudes and a specified wire to remain,
    this function returns the reduced density matrix of the remaining qubit as a jax.numpy.ndarray
    after tracing out the other qubit.

    Parameters:
    - amplitudes (jnp.ndarray): An array of the amplitudes for each basis element of the two-qubit state.
    - remaining_wire (int): The qubit (wire) to remain after tracing out the other (0 for the first qubit, 1 for the second qubit).

    Returns:
    - jnp.ndarray: The reduced density matrix for the remaining qubit after tracing out the other qubit.
    """
    # Compute the full density matrix
    rho = compute_density_matrix(amplitudes)

    # Trace out the other qubit
    rho_remaining = trace_out_dm(rho, remaining_wire)

    return rho_remaining


### CHECK PREDICTIONS

def accuracy_score(actual: jnp.ndarray, expected: jnp.ndarray):
    if actual.ndim > 1:
        actual = actual.flatten()
    if expected.ndim > 1:
        expected = expected.flatten()
    return (actual == expected).sum() / len(actual)


### SAVE TO FILE
def save_array_to_csv(arr, filename):
    if filename[:-4] != '.csv':
        filename = filename + '.csv'
    create_path_if_missing(filename)
    savetxt(filename, arr, delimiter=',')


def save_pickle(path, object):
    path_ = f'{path}.pkl' if path[-4:] != '.pkl' else path

    if not object.trained:
        raise Exception('The model is not trained yet, so you must not save it')

    os.makedirs(os.path.dirname(path_), exist_ok=True)
    with open(path_, 'wb') as file:
        pickle.dump(object, file)


def load_pickle(path: str):
    path_ = f'{path}.pkl' if path[-4:] != '.pkl' else path
    with open(path_, 'rb') as file:
        object = pickle.load(file)
    return object


def create_path_if_missing(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def _process_extension(filename, extension):
    try:
        root, ext = filename.rsplit('.', 1)
    except ValueError:
        # The file has no extension, so we add it
        return filename + '.' + extension
    else:
        # The file already has an extension
        if ext != extension:
            raise ValueError(f"The file '{filename}' already has the extension '.{ext}'. Cannot add the extension '.{extension}'.")
        else:
            # The extension is already correct
            return filename


def save_dict_to_json(dict: dict, path:str):
    path = _process_extension(path, extension='json')
    with open(path, 'w') as file:
        json.dump(dict, file, indent=4)

def load_json_to_dict(path:str):
    path = _process_extension(path, extension='json')
    with open(path, 'r') as file:
        return json.load(file)
    
### RANDOM
def get_current_folder():
    # Obtener el stack frame actual
    frame = inspect.currentframe()
    # Obtener el stack frame del llamador
    caller_frame = inspect.getouterframes(frame, 2)
    # Obtener la ruta del archivo del llamador
    ruta_script = caller_frame[1].filename
    # Obtener el directorio padre
    directorio_actual = os.path.dirname(os.path.abspath(ruta_script))
    # Devolver la ruta del directorio padre
    return directorio_actual

def get_root_path(start_dir=None):
    if start_dir is None:
        start_dir = os.getcwd()

    current_dir = os.path.abspath(start_dir)

    while True:
        if any(os.path.isdir(os.path.join(current_dir, folder)) for folder in ['src', 'tests']):
            return current_dir

        # Move up one directory
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root of the file system
            return None

        current_dir = parent_dir

def get_function(method_name, Class):
    # Get all attributes (including methods) of the class
    attributes = vars(Class)

    # Check if the method exists
    if method_name in attributes:
        func = getattr(Class, method_name)
        return func
    # If method not found or not static, return None
    raise ModuleNotFoundError('Method {} not found in class {}'.format(method_name, Class))


# SOLVE SCHRODINGER EQUATION NUMERICALLY
def evolve(h, psi0, t0, tend, *, time_steps=1000):
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


# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# Prints
def colorize(text, color_code):
  """Prints the given text in the specified color.

  Args:
    text: The text to print.
    color_code: The ANSI escape code for the desired color.
  """
  print(f"\033[{color_code}m" + text + "\033[0m")

def print_in_gray(text):
  """Prints the given text in gray."""
  colorize(text, 90)

def print_in_yellow(text):
  """Prints the given text in yellow."""
  colorize(text, 93)

def print_in_blue(text):
  """Prints the given text in blue."""
  colorize(text, 94)

def print_in_red(text):
  """Prints the given text in red."""
  colorize(text, 91)

def print_in_green(text):
  """Prints the given text in green."""
  colorize(text, 92)

def print_in_orange(text):
  """Prints the given text in orange."""
  colorize(text, 33)

def print_in_purple(text):
  """Prints the given text in purple."""
  colorize(text, 35)

def print_in_cyan(text):
  """Prints the given text in cyan."""
  colorize(text, 36)

def print_in_light_gray(text):
  """Prints the given text in light gray."""
  colorize(text, 37)

def print_in_dark_gray(text):
  """Prints the given text in dark gray."""
  colorize(text, 90)

