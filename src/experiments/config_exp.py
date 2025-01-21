from src.utils import get_current_folder
import os
import re
import numpy as np
from jax import numpy as jnp
from icecream import ic
from src.QNN import PulsedQNN, GateQNN
from src.QNN.BaseQNN import BaseQNN
from src.EQK import EQKn
from src.utils import load_pickle
from src.utils import pickle_extension
from src.Sampler.utils import scale_points
import argparse
from src.Sampler import *
import ast

PLOT_RESULT = True
RESULTS_FOLDER = f'{get_current_folder()}/results'
PARTIAL_FOLDER = f'{RESULTS_FOLDER}/partial'
LRBONUDARIES_ERROR_MESSAGE = 'lr and lrBoundaries lengths does not match'
OPTIMIZER_ERROR_MESSAGE = 'optimizer_error'


def parse_dict_old(d: dict):
    """
    Recursively convert all numpy arrays or similar array-like objects
    in a dictionary to Python lists.
    """
    parsed_dict = d.copy()
    for key, value in parsed_dict.items():
        if isinstance(value, dict):
            # Recursively apply the function to nested dictionaries
            parse_dict(value)
        elif isinstance(value, (np.ndarray, list)) and hasattr(value, 'tolist'):
            # Convert numpy arrays (or similar) to lists
            parsed_dict[key] = value.tolist()
    return parsed_dict


def parse_dict(d: dict):
    def convert_to_list(value):
        if isinstance(value, (np.ndarray, jnp.ndarray)) and hasattr(value, 'tolist'):
            return value.tolist()  # Convert numpy array to list
        elif isinstance(value, list):
            return [convert_to_list(v) for v in value]  # Recursively process lists
        return value  # Return other types unchanged

    d_ = d.copy()
    for key, value in d_.items():
        d_[key] = convert_to_list(value)  # Convert the value
    return d_


def get_exp_name(config_dict):
    name = '__'.join(f'{k}_{v}' for k, v in config_dict.items() if k != 'paramBoundaries')
    if 'paramBoundaries' in config_dict:
        name += '__paramBoundaries_' + '_'.join(f'{k}_{v}' for k, v in config_dict['paramBoundaries'].items())
    return name


def get_partial_filename(partial_folder: str = PARTIAL_FOLDER, partial_identifier: str = 'results'):
    pattern = re.compile(rf"^{partial_identifier}_(\d+)\.csv$")
    max_index = -1
    for filename in os.listdir(partial_folder):
        match = pattern.match(filename)
        if match:
            # Convertir el índice encontrado a entero y comparar con el actual máximo
            index = int(match.group(1))
            max_index = max(max_index, index)
    partial_id = max_index + 1
    patial_id_file = f'{partial_identifier}_{partial_id}'
    return patial_id_file


# Setter function to modify the global variable
def set_PLOT_RESULT(value):
    global PLOT_RESULT
    PLOT_RESULT = value


# Getter function to access the global variable
def get_PLOT_RESULT():
    return PLOT_RESULT


def process_arguments():
    """
    Process command-line arguments for configuring experiment parameters.

    This function sets up an argument parser to handle various experiment settings,
    including the number of trials, jobs, dataset size, and model selection.

    Returns:\n
        - n_trials (float): Number of trials for hyperparameter tuning.
        - n_jobs (int): Number of simultaneous trials to run.
        - n_train (int): Number of data points in the training set.
        - n_test (int): Number of data points in the test set.
        - dataset (str): Name of the dataset to be used.
        - model (str): Name of the model to be trained.
        - LOAD_RESULTS (bool): Whether to load existing results.

    The function uses argparse to define and parse the following command-line arguments:\n
        --trials (float): Number of trials for hyperparameter tuning (default: 225).\n
        --n_jobs (int): Number of simultaneous trials (default: 4).\n
        --n_train (int): Number of points for the train set (default: 400).\n
        --n_test (int): Number of points for the test set (default: 200).\n
        --load (str): Whether to load results (default: 'False').\n
        --dataset (str): Dataset to be used (default: 'MNIST').\n
        --model (str): Model to train (default: 'pulsedqnn').
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=float, help='Number of trials to tune hyperparameters',
                        required=False, default=225)
    parser.add_argument('--n_jobs', type=int, help='Number of simultaneous trials', required=False, default=4)
    parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=400)
    parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=200)
    parser.add_argument('--load', type=str, help='whether to load results', required=False, default='False')
    parser.add_argument('--dataset', type=str, help='dataset to be used', required=False, default='MNIST')
    parser.add_argument('--model', type=str, help='model to train', required=False, default='pulsedqnn')

    args = parser.parse_args()

    n_trials = args.trials
    n_jobs = int(args.n_jobs)
    n_train = int(args.n_train)
    n_test = int(args.n_test)
    dataset = args.dataset
    model = args.model
    LOAD_RESULTS = ast.literal_eval(args.load)

    return n_trials, n_jobs, n_train, n_test, dataset, model, LOAD_RESULTS


def get_highest_id(folder_path, pattern: str = 'results', extension: str = 'csv'):
    pattern = re.compile(fr"{pattern}_(\d+)\.{extension}")  # Regex to match 'results_id.csv'
    highest_id = -1

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            file_id = int(match.group(1))
            highest_id = max(highest_id, file_id)

    return highest_id


def get_dataset(dataset: str, n_train: int, n_test: int, interface: str, points_dimension: int, seed: int = None, scale:float = np.pi):
    if 'fashion' in dataset.lower():
        if '_' in dataset:  #like fashion_01 to select the labels
            label1, label2 = dataset[-2:]
        else:
            label1, label2 = 3, 6  # default values
        train_set, train_labels, test_set, test_labels = MNISTSampler.fashion(n_train=n_train,
                                                                              label1=label1, label2=label2,
                                                                              points_dimension=points_dimension,
                                                                              n_test=n_test, seed=seed,
                                                                              interface=interface)
    elif 'digits' in dataset.lower():
        if '_' in dataset:  #like digits_01 to select the labels
            label1, label2 = dataset[-2:]
        else:
            label1, label2 = 8, 0  # default values
        train_set, train_labels, test_set, test_labels = MNISTSampler.digits(n_train=n_train,
                                                                             label1=label1, label2=label2,
                                                                             points_dimension=points_dimension,
                                                                             n_test=n_test, seed=seed,
                                                                             interface=interface)
    elif dataset.lower() == 'circles':
        train_set, train_labels = Sampler.circle(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler.circle(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'spiral':
        train_set, train_labels = Sampler.spiral(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler.spiral(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'corners':
        train_set, train_labels = Sampler.corners(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler.corners(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'sinus':
        train_set, train_labels = Sampler.sinus(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler.sinus(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'corners3d':
        train_set, train_labels = Sampler3D.corners3d(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler3D.corners3d(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'sinus3d':
        train_set, train_labels = Sampler3D.sinus3d(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler3D.sinus3d(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'annulus':
        train_set, train_labels = Sampler.annulus(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler.annulus(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'shell':
        train_set, train_labels = Sampler3D.shell(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler3D.shell(n_points=n_test, seed=seed, interface=interface)
    elif dataset.lower() == 'helix':
        train_set, train_labels = Sampler3D.helix(n_points=n_train, seed=seed, interface=interface, z_speed=0.5)
        test_set, test_labels = Sampler3D.helix(n_points=n_test, seed=seed, interface=interface, z_speed=0.5)
    elif dataset.lower() == 'random_easy':
        train_set, train_labels, test_set, test_labels = RandomSampler.easy_problem(dimension=points_dimension,
                                                                                    n_train=n_train,
                                                                                    n_test=n_test,
                                                                                    seed=seed,
                                                                                    interface=interface)
    elif dataset.lower() == 'random_medium':
        train_set, train_labels, test_set, test_labels = RandomSampler.medium_problem(dimension=points_dimension,
                                                                                      n_train=n_train,
                                                                                      n_test=n_test,
                                                                                      seed=seed,
                                                                                      interface=interface)
    elif dataset.lower() == 'random_hard':
        train_set, train_labels, test_set, test_labels = RandomSampler.hard_problem(dimension=points_dimension,
                                                                                    n_train=n_train,
                                                                                    n_test=n_test,
                                                                                    seed=seed,
                                                                                    interface=interface)
    elif dataset.lower() == 'torus':
        train_set, train_labels = Sampler3D.torus(n_points=n_train, seed=seed, interface=interface)
        test_set, test_labels = Sampler3D.torus(n_points=n_test, seed=seed, interface=interface)
    else:
        raise ValueError(f'Dataset {dataset} not recognized. Available: fashion, digits, annulus and circles')

    # Scale dataset
    train_set = scale_points(train_set, scale_range=(-scale, scale), center=True)
    test_set = scale_points(test_set, scale_range=(-scale, scale), center=True)

    return train_set, train_labels, test_set, test_labels


def get_qnn(model: str, n_qubits: int, n_layers: int, realistic_gates: bool = False, seed: int = None,
            interface: str = 'jax', constant4amplitude: float = 1):
    if 'pulse' in model.lower():
        qnn = PulsedQNN(num_qubits=n_qubits, num_layers=n_layers, interface=interface, seed=seed,
                        pulse_shape='constant', encoding='pulsed', constant4amplitude=constant4amplitude)
    elif 'mixed' in model.lower():
        qnn = PulsedQNN(num_qubits=n_qubits, num_layers=n_layers, interface=interface, seed=seed,
                        pulse_shape='constant', encoding='gate', constant4amplitude=constant4amplitude)
    elif 'gate' in model.lower():
        qnn = GateQNN(num_qubits=n_qubits, num_layers=n_layers, interface=interface, seed=seed,
                      realistic_gates=realistic_gates)
    else:
        raise ValueError(f'Model {model} not recognized. Available: pulsed, mixed and gate')
    return qnn


def get_eqk(qnn: BaseQNN):
    eqk = EQKn(qnn)
    return eqk


def get_optimal_lr(optimizer: str, dataset: str) -> tuple[dict, dict]:
    # Load tuned MODEL-QUBIT-LAYER tuned lr in a dictionary
    try:
        tuned_optimizer_dataset_path = \
            f'data/tuned_parameters/{optimizer}/{dataset}/results_model_qubit_layer.{pickle_extension}'
        if os.path.exists(tuned_optimizer_dataset_path):
            tuned_qubits_layer = load_pickle(tuned_optimizer_dataset_path)
        else:
            tuned_optimizer_dataset_path = tuned_optimizer_dataset_path.replace(f'/{dataset}', '')
            if os.path.exists(tuned_optimizer_dataset_path):
                tuned_qubits_layer = load_pickle(tuned_optimizer_dataset_path)
            else:
                tuned_qubits_layer = None
    except:
        tuned_qubits_layer = None

    # Load tuned MODEL tuned lr in a dictionary (generic)
    try:
        tuned_global_path = f'data/tuned_parameters/{optimizer}/{dataset}/results_model_global.{pickle_extension}'
        if os.path.exists(tuned_global_path):
            tuned_global = load_pickle(tuned_global_path)
        else:
            tuned_global_path = tuned_global_path.replace(f'/{dataset}', '')
            if os.path.exists(tuned_global_path):
                tuned_global = load_pickle(tuned_global_path)
            else:
                tuned_global = None
    except:
        tuned_global = None

    return tuned_qubits_layer, tuned_global


def get_optimal_opt_parameters(tuned_qubits_layer: dict, tuned_global: dict, model: str, n_qubits: int, n_layers: int):
    """
    Retrieve optimal optimization parameters for a given quantum model configuration.

    This function attempts to find the best learning rate (lr) for a specified quantum model
    based on pre-tuned parameters or defaults to predefined values if no tuned parameters are available.

    Parameters:
    tuned_qubits_layer (dict): A dictionary containing tuned parameters for specific model configurations.
                               Keys are tuples of (model, n_qubits, n_layers), values are learning rates.
    tuned_global (dict): A dictionary containing globally tuned parameters for models.
                         Keys are model names, values are learning rates.
    model (str): The name of the quantum model ('gate', 'pulsed', or 'mixed').
    n_qubits (int): The number of qubits in the quantum model.
    n_layers (int): The number of layers in the quantum model.

    Returns:
    dict: A dictionary containing the optimal learning rate for the given configuration.
          The dictionary has a single key 'lr' with the corresponding learning rate value.
    """
    default_values = {'gate': 0.05, 'pulsed': 0.0001, 'mixed': 0.0001}
    optimal_parameters = None

    # Try to retrieve the optimal parameters for model, n_qubits and n_layers
    if tuned_qubits_layer is not None:
        if (model, str(n_qubits), str(n_layers)) in tuned_qubits_layer:
            optimal_parameters = {'lr': tuned_qubits_layer[(model, str(n_qubits), str(n_layers))]}

    # Else try to retrieve the optimal parameters for model
    if tuned_global is not None:
        if optimal_parameters is None:
            if model in tuned_qubits_layer:
                optimal_parameters = {'lr': tuned_qubits_layer[model]}

    # Else default values
    if optimal_parameters is None:
        optimal_parameters = {'lr': default_values[model]}

    return optimal_parameters
