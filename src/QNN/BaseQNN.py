import optax
from warnings import warn
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from tqdm import tqdm
from pennylane.measurements import ExpectationMP
import jax
from jax import numpy as jnp
from abc import ABC, abstractmethod
import random
from functools import partial
from joblib import Parallel, delayed
from typing import Union, Sequence
from src.utils import iterate_minibatches, accuracy_score, increase_dimensions, save_pickle, load_pickle
from icecream import ic

try:
    import pennypulse
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "src/Pennypulse"])
    import pennypulse
jax.config.update("jax_enable_x64", True)

DEFAULT_N_EPOCHS = 30
DEFAULT_BATCH_SIZE = 24

DEFAULT_OPT_PARAMS = {
    'lr': 0.1,
    'lr_boundaries': None,
    # Adam parameters
    'beta1': 0.9,
    'beta2': 0.999,
    # RMSProp parameters
    'decay': 0.9,
}

DEFAULT_EARLY_STOPPING = {
    'patience': 5,  # Number of iterations without improvement
    'min_delta': 1e-4,  # Minimum improvement to reset patience
}


class BaseQNN(ABC):

    def __init__(self, num_qubits: int = 2,
                 num_layers: int = 5, n_workers: int = 1,
                 params_per_layer: int = 3,
                 interface: str = 'jax',
                 seed: int = None):
        """
        Base class for Quantum Neural Networks (QNNs). This abstract base class provides a common foundation for 
        different QNN architectures. 
        
        Args:
            num_qubits: The number of qubits in the QNN. Defaults to 2.
            num_layers: The number of layers in the QNN. Defaults to 5.
            n_workers: The number of workers for parallel execution (currently not effective). Defaults to 1.
            params_per_layer: The number of parameters per layer. Defaults to 3.
            interface: The interface used for the QNN ('jax' or 'pennylane'). Defaults to 'jax'.
            seed: The random seed used for reproducibility. Defaults to None.
        
        Attributes:
            num_qubits: The number of qubits in the QNN.
            num_layers: The number of layers in the QNN.
            params_per_layer: The number of parameters per layer.
            interface: The interface used for the QNN ('jax' or 'pennylane').
            seed: The random seed used for reproducibility.
            params: The parameters of the QNN circuit.
            projection_angles: The angles used for projection measurements.
            trained: A boolean indicating whether the QNN has been trained.
            training_info: A dictionary to store training information.
            dev: The PennyLane quantum device used for simulations.
            n_workers: The number of workers for parallel execution (currently not effective).
            name: The name of the specific QNN architecture.
            model_name: The name of the overall model.
        
        Raises:
            ValueError: If the number of qubits is greater than the number of layers + 1.
        """
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
        self.seed = seed

        if num_qubits > num_layers + 1:
            raise ValueError("The number of qubits cannot be greater than the number of layers + 1")

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params_per_layer = params_per_layer
        self.interface = interface
        self.params, self.projection_angles = self.initialize_parameters(params_per_layer, interface)

        self.trained = False
        self.training_info = None
        self.dev = qml.device("default.qubit", wires=num_qubits)

        if n_workers != 1:
            warn("Currently the parallel execution is not faster than sequential")
            n_workers = 1
        self.n_workers = n_workers
        self.name = 'To be defined'
        self.model_name = 'To be defined'

    def qnn(self, params, x, dm):
        if self.interface == 'jax':
            qnode = qml.QNode(self._qnn, self.dev, interface='jax')
        elif self.interface == 'pennylane':
            qnode = qml.QNode(self._qnn, self.dev)
        return qnode(params, x, dm)

    def _qnn(self, params, x, dm) -> ExpectationMP:
        """
        This is the base method. It calls the base circuit and calculates the expected value of the projector.
        The `params` argument contains the parameters for each qubit, without including the projection parameters.

        Args:
        params: A list of length 2 * n_qubits - 1 containing dictionaries.
                - Qubit 0 has parameters in params[0].
                - Qubit k (k != 0) has non-entangling parameters in params[2k-1] and
                  entangling parameters in params[2k].
        x: The (single) data point.
        dm: The (single) projector of the corresponding data label. For example, if self.projection_angles = [0, 0]:
              - dm == 0 -> |0><0|.
              - dm == 1 -> |1><1|.
            These projectors are rotated according to the projection parameter.
        Returns:
        The expected value of the projector.
        """
        self._base_circuit(x, params)
        return qml.expval(pennypulse.Hermitian(dm, wires=[0]))

    def forward(self, point_set: Sequence, up_to_qubit: int = None):
        """
        Perform forward pass of the quantum neural network on a set of input points. To do that, it supposes all points 
        are mapped to label 0 and then correct labels (fid < 0.5) afterwards
    
        This method applies the quantum circuit to the input points and returns predictions based on the fidelities of 
        the output states.
    
        Args:
            point_set (Sequence): A sequence of input points to be processed by the quantum circuit.
            up_to_qubit (int, optional): The index of the last qubit to consider in the computation.
                                         If None, all qubits are used. Defaults to None.
    
        Returns:
            jnp.ndarray: An array of binary predictions (0 or 1) for each input point.
        """
        # 
        if self.interface == 'jax':
            p = self.params if up_to_qubit is None \
                else jax.lax.slice(self.params,  # similar to self.params[:2*up_to_qubit+1,:,:] in jax
                                   start_indices=(0, 0, 0),
                                   limit_indices=(2 * up_to_qubit + 1, self.params.shape[1], self.params.shape[2])
                                   )
            y = jnp.zeros(len(point_set), dtype=int)
        else:
            p = self.params if up_to_qubit is None else self.params[:2 * up_to_qubit + 1, :, :]
            y = np.zeros(len(point_set), dtype=int)

        fidelities = self._get_fidelities([p, self.projection_angles], point_set, y)
        predictions = jnp.where(fidelities > 0.5, 0, 1)
        return predictions

    def create_dms(self, theta, phi):
        """
        creates the density matrix of the states that are used for represent the classification
        PROJECTION PARAMS:
        These angles reprensent the anlge representing the state which with respect perform the measurement projections
        to measure the fidelity. This is, we project with respect to the state
        |psi> = (cos(θ/2) |0> + e^{iϕ}sin(θ/2)|1>) and
        |psi⟂> = (-sin(θ/2) |0> + e^{iϕ}cos(θ/2)|1>) and
        And the corresponding projectors are:
        P =  [[cos^2(θ/2)                  e^{-iϕ}cos(θ/2)sin(θ/2)],
              [e^{iϕ}cos(θ/2)sin(θ/2)      sin^2(θ/2)]]
        P⟂ =[[sin^2(θ/2)                   -e^{-iϕ}cos(θ/2)sin(θ/2)],
              [-e^{iϕ}cos(θ/2)sin(θ/2)     cos^2(θ/2)]]

        Notet that sin(x)cos(x) = sin(2x)/2
        """
        matrix1 = jnp.array([[jnp.cos(theta / 2) ** 2, jnp.exp(-1j * phi) * jnp.sin(theta) / 2],
                             [jnp.exp(1j * phi) * jnp.sin(theta) / 2, jnp.sin(theta / 2) ** 2]],
                            dtype=complex)
        matrix2 = jnp.array([[jnp.sin(theta / 2) ** 2, -jnp.exp(-1j * phi) * jnp.cos(theta / 2) * jnp.sin(theta / 2)],
                             [-jnp.exp(1j * phi) * jnp.sin(theta) / 2, jnp.cos(theta / 2) ** 2]],
                            dtype=complex)
        return jnp.array([matrix1, matrix2], dtype=complex)

    def initialize_parameters(self, params_per_layer, interface: str):
        """
        NORMAL PARAMS:
        The parameters created are stored in a large matrix with shape (2 * n_qubits - 1, n_layers, params_per_layer)
        - The first index (0, :, :) corresponds to qubit 0.
        - The second (1, :, :) and third indices (2, :, :) correspond to qubit 1. The first of these contains
        non-entangling parameters, while the third contains the entangling parameters.
        ...
        - For qubit k != 0, the parameters at 2k-1 are non-entangling, and those at 2k are entangling.

        The second slicing index indicates the number of layers, and the third slicing index depends on the architecture.
        If it is for the GateQNN, then there are 3 parameters that represent rotations with no physical significance.
        If it is for the PulsedQNN, there are 4 parameters that do have physical meaning:
                                [ang_rot_z1, amplitude pulse y, phase y, ang_rot_z2]
        PROJECTION PARAMS:
        These angles reprensent the anlge representing the state which with respect perform the measurement projections
        to measure the fidelity. This is, we project with respect to the state
        |psi> = (cos(θ/2) |0> + e^{iϕ}sin(θ/2)|1>) and
        |psi⟂> = (-sin(θ/2) |0> + e^{iϕ}cos(θ/2)|1>) and
        And the corresponding projectors are:
        P =  [[cos^2(θ/2)                  e^{-iϕ}cos(θ/2)sin(θ/2)],
              [e^{iϕ}cos(θ/2)sin(θ/2)      sin^2(θ/2)]]
        P⟂ =[[sin^2(θ/2)                   -e^{-iϕ}cos(θ/2)sin(θ/2)],
              [-e^{iϕ}cos(θ/2)sin(θ/2)     cos^2(θ/2)]]
        """
        if interface == 'jax':
            if self.seed is None:
                seed = random.randint(0, 100000)
            else:
                seed = self.seed
            key = jax.random.PRNGKey(seed)
            params = jnp.zeros((2 * self.num_qubits - 1, self.num_layers, params_per_layer), dtype=jnp.float32)
            params = params.at[0, :, :].set(jax.random.uniform(key, shape=(self.num_layers, params_per_layer)))
            projection_angles = jnp.zeros(2)
        else:  # elif interface == 'pennylane':
            params = np.zeros(shape=(2 * self.num_qubits - 1, self.num_layers, params_per_layer), requires_grad=True)
            params[0, :, :] = np.random.uniform(size=(self.num_layers, params_per_layer),
                                                requires_grad=True)
            projection_angles = np.zeros(2, requires_grad=True)
        return params, projection_angles

    def _get_fidelities(self, cost_params, x_set, y_set):
        """
        Calculate the fidelities for a given set of input data points and labels.
    
        Args:
            cost_params (list): A list containing two elements:
                - params: The parameters for the quantum circuit.
                - projection_angles: The angles used for creating density matrices.
            x_set (array-like): The input dataset to be processed.
            y_set (array-like): The labels corresponding to the input dataset.
    
        Returns:
            array-like: The computed fidelities for each input data point.
    
        Note: The input x_set is increased to 3 dimensions before processing.
        """
        x_set_ = increase_dimensions(x_set, 3)
        params, projection_angles = cost_params
        dm_labels = self.create_dms(*projection_angles)

        # vectorize function
        if self.interface == 'jax':
            qnn_vectorized = jax.jit(jax.vmap(self.qnn, in_axes=(None, 0, 0)))
            qnn_args = [params, x_set_, dm_labels[y_set]]
        else:
            qnn_vectorized = np.vectorize(lambda x_, y_: self.qnn(params, x_, y_),
                                          signature='(n), (m,m) -> ()')
            qnn_args = [x_set_, dm_labels[y_set]]

        # Run circuit
        if self.n_workers != 1:
            fidelities = (Parallel(n_jobs=self.n_workers, backend='multiprocessing')
                          (delayed(qnn_vectorized)(*qnn_args)))
        else:
            fidelities = qnn_vectorized(*qnn_args)

        return fidelities

    def cost(self, cost_params, Xbatch: jnp.ndarray, Ybatch: jnp.ndarray):
        """
        Calculate the cost (loss) for a batch of data points and their corresponding labels.
        This function computes the fidelities for the given batch and calculates the mean squared error loss.

        Args:
            cost_params (list): A list containing two elements:
                - params: The parameters for the quantum circuit.
                - projection_angles: The angles used for creating density matrices.
            Xbatch (jnp.ndarray): A batch of input data points.
            Ybatch (jnp.ndarray): A batch of corresponding labels for the input data points.

        Returns:
            float: The computed loss value, which is the mean squared error of (1 - fidelities).
        """
        fidelities = self._get_fidelities(cost_params, Xbatch, Ybatch)
        loss = ((1 - fidelities) ** 2).sum() / len(fidelities)
        return loss

    def train(self, data_points_train,
              data_labels_train,
              data_points_test=None,
              data_labels_test=None,
              n_epochs: Union[int, dict] = DEFAULT_N_EPOCHS,
              batch_size: Union[int, dict] = DEFAULT_BATCH_SIZE,
              optimizer: str = 'rms',
              optimizer_parameters: dict = None,
              early_stopping: dict = None,
              save_stats: bool = True,
              silent: bool = False):
        """
        Train the quantum neural network using the provided data.

        This function performs the training of the quantum neural network using the specified parameters and
        optimization strategy. It supports training on a per-qubit basis and can handle both training and test datasets.

        Args:
            data_points_train (array-like): Training data points.
            data_labels_train (array-like): Labels corresponding to the training data points.
            data_points_test (array-like, optional): Test data points. Defaults to None.
            data_labels_test (array-like, optional): Labels corresponding to the test data points. Defaults to None.
            n_epochs (Union[int, dict]): Number of epochs for training. Can be a single integer for all qubits
                or a dictionary specifying epochs for specific qubits. Defaults to 30.
                Example: n_epochs = 4 or n_epochs = {0: 3, 1: 3, -1: 2}
            batch_size (Union[int, dict]): Batch size for training. Can be a single integer for all epochs
                or a dictionary specifying batch sizes for specific epochs. Defaults to 24.
                Example: batch_size = 40 or batch_size = {0: 30, 1: 40, -1: 20}
            optimizer (str): The name of the optimizer to be used. Defaults to 'rms'.
            optimizer_parameters (dict, optional): Parameters for the optimizer. Defaults to None.
                Examples:
                 {'lr': [0.1, 0.05], 'lr_boundaries': [2], 'beta1': 0.9},
                    or {0: {'lr': 0.1}, '-1': {'lr': [0.01, 0.1], 'lr_boundaries': [1]}},
                        or {0: {'lr': 0.1}}
            early_stopping (dict, optional): Early stopping criteria. Can be None, True/False, or a dictionary
                with 'min_delta' and 'patience' keys. Defaults to None.
            save_stats (bool): Whether to save training statistics (loss and accuracy) through epochs.
                If False, only the last epoch's stats are saved. Defaults to True.
            silent (bool): Whether to suppress printing of training information. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing training statistics, including loss and accuracy
            for the final epoch of each qubit.
        """
        # Check silent + save_stats compatibility
        if not silent and not save_stats:  # if silent is False and save_stats is False, change save_stats
            save_stats = True

        range_qubit = range(self.num_qubits)
        if silent:  # Keep track of the progress bar in case do not display qnn training stats
            range_qubit = tqdm(range_qubit, desc='Training qubit', leave=False)

        # Stopping conditions
        if early_stopping:  # If early stopping is 'True'
            early_stopping = DEFAULT_EARLY_STOPPING
        elif isinstance(early_stopping, dict):
            for k, v in DEFAULT_EARLY_STOPPING.items():
                if k not in early_stopping:
                    early_stopping[k] = v
        elif early_stopping is None or not early_stopping:
            early_stopping = False
        else:
            raise ValueError('early_stopping must be either a True/False, a dictionary or None')

        # Start training
        if save_stats:
            cost_params = [self.params[:1], self.projection_angles]  # Take params from qubit 0
            res = [self.cost(cost_params, data_points_train, data_labels_train),
                   self.get_accuracy(data_points_train, data_labels_train, up_to_qubit=0),
                   self.get_accuracy(data_points_test, data_labels_test, up_to_qubit=0) if data_points_test is not None
                   else -1
                   ]
            stats = [[-1] + res]
            if not silent:
                print("\nStart training:\n"
                      "Epoch: {:3d} | Loss: {:3f} | Train accuracy: {:2.2f} | Test accuracy: {:2.2f}".format(0, *res))

        if optimizer_parameters is None:
            optimizer_parameters = DEFAULT_OPT_PARAMS

        for qubit_ in range_qubit:
            if not silent: print(f'%%% Training qubit {qubit_} %%%')
            
            if early_stopping is not False:
                best_loss = float('inf')
                patience_counter = 0

            ##### Get the number of epochs to train this qubit #####
            if isinstance(n_epochs, int):
                n_epochs_ = n_epochs
            elif isinstance(n_epochs, dict):
                if qubit_ in n_epochs:
                    n_epochs_ = n_epochs[qubit_]
                elif -1 in n_epochs:
                    n_epochs_ = n_epochs[-1]
                else:
                    n_epochs_ = DEFAULT_N_EPOCHS  # default value
            else:
                raise ValueError('n_epochs must be either a int or a dictionary')
            range_epochs = range(n_epochs_)
            if silent:
                range_epochs = tqdm(range_epochs, desc=f'Training epoch', leave=False)

            ##### get the batch size to train the qubit #####
            if isinstance(batch_size, int):
                batch_size_ = batch_size
            elif isinstance(batch_size, dict):
                if qubit_ in batch_size:
                    batch_size_ = batch_size[qubit_]
                elif -1 in batch_size:
                    batch_size_ = batch_size[-1]
                else:
                    batch_size_ = DEFAULT_BATCH_SIZE
            else:
                raise ValueError('batch_size must be either a int or a dictionary')

            if len(data_points_train) < batch_size_:
                batch_size_ = len(data_points_train)
            else:
                batch_size_ = batch_size

            ##### Start training #####
            # Init
            params_up_to_qubit = self.params[:2 * qubit_ + 1]  # Take params from qubit 0 to training qubit
            cost_params = [params_up_to_qubit, self.projection_angles]

            # Get an optimizer (one per qubit training is necessary)
            if qubit_ in optimizer_parameters:  # opt for qubit specified
                opt_params_qubit = optimizer_parameters[qubit_]
            elif -1 in optimizer_parameters:  # opt for every qubit not explicitly specified
                opt_params_qubit = optimizer_parameters[-1]
            elif 'lr' in optimizer_parameters:  # a common for all of them
                opt_params_qubit = optimizer_parameters
            else:  # Use of default optimizer
                opt_params_qubit = DEFAULT_OPT_PARAMS

            for key, value in DEFAULT_OPT_PARAMS.items():
                if key not in opt_params_qubit:
                    opt_params_qubit[key] = value  # default value if not provided by user

            opt_params_qubit['n_epochs'] = n_epochs_
            opt = self._get_optimizer(optimizer, opt_params_qubit)

            if self.interface == 'jax':
                opt_state = opt.init(cost_params)
                grad = jax.jit(jax.grad(self.cost))
            else:
                opt_state = None
                grad = None
            
            for it in range_epochs:
                for Xbatch, ybatch in iterate_minibatches(data_points_train, data_labels_train,
                                                          batch_size=batch_size_):
                    if self.interface == 'jax':
                        grad_value = grad(cost_params, Xbatch, ybatch)
                        updates, opt_state = opt.update(grad_value, opt_state)
                        cost_params = optax.apply_updates(cost_params, updates)

                        # if self.name == 'PulsedQNN':
                        #     cost_params[0] = self._clip_params(cost_params[0])

                    else:  # self.interface == 'pennylane
                        raise NotImplementedError
                        params_up_to_qubit = opt.step(self.cost, cost_params, Xbatch=Xbatch, Ybatch=ybatch)

                # Update params
                params_up_to_qubit = cost_params[0]
                if self.interface == 'jax':
                    self.params = self.params.at[:2 * qubit_ + 1].set(params_up_to_qubit)
                else:
                    self.params[:2 * qubit_ + 1] = params_up_to_qubit
                self.projection_angles = cost_params[1]
                # Get stats
                if save_stats:
                    loss = self.cost(cost_params, data_points_train, data_labels_train)
                    accuracy_train = self.get_accuracy(data_points_train, data_labels_train, up_to_qubit=qubit_)

                    if data_labels_test is not None and data_labels_train is not None:
                        accuracy_test = self.get_accuracy(data_points_test, data_labels_test, up_to_qubit=qubit_)
                    else:
                        accuracy_test = -1

                    res = [it + 1, loss, accuracy_train, accuracy_test]
                    if not silent:
                        print("Epoch: {:3d} | Loss: {:3f} | Train accuracy: {:2.2f} | Test accuracy: {:2.2f}".format(
                            *res))

                # Check for early stopping
                if early_stopping is not False:
                    if not save_stats:  # Then loss has not been computed
                        loss = self.cost(cost_params, data_points_train, data_labels_train)

                    if loss < best_loss - early_stopping['min_delta']:
                        best_loss = loss
                        patience_counter = 0  # Reset patience
                    else:
                        patience_counter += 1

                    # Stop early if patience is exceeded
                    if patience_counter >= early_stopping['patience']:
                        print(f"Early stopping at epoch {it + 1}")
                        break
            if save_stats:
                stats.append([qubit_] + res[1:])  # only save final results

        ##### Finish training #####
        if not save_stats:
            accuracy_train = self.get_accuracy(data_points_train, data_labels_train)

            if data_labels_test is not None and data_labels_train is not None:
                accuracy_test = self.get_accuracy(data_points_test, data_labels_test)
            else:
                accuracy_test = -1

            if not save_stats and not early_stopping:  # then loss has not been recorded
                cost_params = [self.params, self.projection_angles]
                loss = self.cost(cost_params, data_points_train, data_labels_train)

            res = [loss, accuracy_train, accuracy_test]  # loss and it come from the last iteration
            stats = [[self.num_qubits] + res]

        # Save training info
        self.training_info = res[1:]
        self.trained = True

        df = pd.DataFrame(stats)  # Coger solo stats finales
        df.columns = ['qubit', 'loss', 'train_accuracy', 'test_accuracy']
        df.set_index('qubit', inplace=True)
        return df

    def get_accuracy(self, points, labels, up_to_qubit: int = None):
        """
        Calculates the accuracy of the QNN on the given data.
        Args:
          points: The input data point set.
          labels: The true labels corresponding to the input point set.
          up_to_qubit: An optional integer specifying the number of qubits to use for the prediction.
                        If None, all qubits are used. Defaults to None
        Returns:
          The accuracy of the QNN as a float.
        """
        predicted = self.forward(points, up_to_qubit=up_to_qubit)
        accuracy = accuracy_score(predicted.flatten(), labels)
        return accuracy.item(0)

    def save_qnn(self, path: str) -> None:
        """
        Saves the Quantum Neural Network in the selected path using `pickle` package. It can be loaded afterwards with
        the static method `BaseQNN.load_qnn(path)`.
        Args:
            path: path to save the qnn
        Return:
             None
        """
        save_pickle(path, self)

    @staticmethod
    def load_qnn(path: str) -> 'BaseQNN':
        """
        Loads a Quantum Neural Network located at the selected path using `pickle` package.
        Args:
            path: path to load the qnn from
        Return:
             BaseQNN object
        """
        return load_pickle(path)

    def _get_optimizer(self, optimizer: str, opt_params: dict):
        """
        Gets the optimizer based on the chosen interface and optimizer name.
        Args:
            optimizer: The name of the optimizer to use.  Available options: ['gradient', 'adam', 'rmsprop'].
            opt_params: A dictionary containing the optimizer parameters.
                        Required parameters depend on the chosen optimizer.
        Returns:
            An optimizer object.
        Raises:
            ValueError: If an unknown optimizer is specified.
            Exception: If the lengths of 'lr' and 'lr_boundaries' do not match (for 'jax' interface).
        """
        if self.interface == 'pennylane':
            lr = opt_params['lr']

            if optimizer == 'adam':
                beta1 = opt_params['beta1']
                beta2 = opt_params['beta2']
                opt = qml.optimize.AdamOptimizer(stepsize=lr, beta1=beta1, beta2=beta2)
            elif optimizer == 'gd' or optimizer == 'gradient':
                opt = qml.optimize.GradientDescentOptimizer(stepsize=lr)
            elif 'rms' == optimizer.lower()[:3]:
                decay = opt_params.get('decay', 0.9)
                eps = opt_params.get('eps', 1e-8)
                opt = qml.optimize.RMSPropOptimizer(stepsize=lr, decay=decay, eps=eps)
            else:
                raise ValueError('Unknown optimizer. Available options: [gradient, adam, rmsprop]')

        else:  # elif self.interface == 'jax':
            lr = opt_params['lr']
            lr_boundaries = opt_params['lr_boundaries']
            n_epochs = opt_params['n_epochs']

            if isinstance(lr, list):
                create_schedule = len(lr) > 1
                if create_schedule:
                    if lr_boundaries is None:
                        lr_boundaries = jnp.linspace(0, n_epochs, len(lr))[1:]
                    elif isinstance(lr_boundaries, list):
                        if len(lr_boundaries) != len(lr) - 1:
                            raise Exception('lr and lr_boundaries lengths does not match')
                    schedule = optax.join_schedules([optax.constant_schedule(lr) for lr in lr],
                                                    boundaries=lr_boundaries)
                else:
                    schedule = lr[0]
            else:
                schedule = lr
            if optimizer == 'adam':
                beta1 = opt_params.get('beta1', DEFAULT_OPT_PARAMS['beta1'])
                beta2 = opt_params.get('beta2', DEFAULT_OPT_PARAMS['beta2'])
                opt = optax.adam(learning_rate=schedule,
                                 b1=beta1,
                                 b2=beta2)
            elif optimizer.lower()[:3] == 'rms':
                decay = opt_params.get('decay', 0.9)
                opt = optax.rmsprop(learning_rate=schedule,
                                    decay=decay)

            else:
                raise ValueError('Unknown optimizer. Available options: [gradient, adam, rmsprop]')
        return opt

    @abstractmethod
    def _base_circuit(self, x, params) -> None:
        """
        In this method, the child objects will save the paramtrized circuit used in the core method `_qnn` of the parent
        class
        Args:
            x: (array-like) single input point
            params: (array-like) parameters for the circuit
        Returns:
            None
        """
        pass

    @abstractmethod
    def get_n_pulses(self) -> int:
        """
        Gets the number of total pulses in the circuit. To be defined in each child.
        """
        pass

    def get_n_params(self) -> int:
        """
        Gets the number of parameters in the circuit
        """
        return 2 * self.num_qubits - 1 * self.num_layers * self.params_per_layer

    def _clip_params(self, params: Sequence) -> Sequence:
        """
        In this method, the child objects will clip parameters after each training epoch to restrict to the valid
        ranges of parameters (example, amplitude of the pulse).
        Args:
            params: (array-like) parameters for the circuit
        Returns:
            clipped params
        """
        raise NotImplementedError('_clip_params method has not been overridden')

    def __str__(self):
        return self.name
