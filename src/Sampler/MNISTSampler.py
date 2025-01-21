import pandas as pd
from sklearn.datasets import fetch_openml
from src.utils import get_current_folder
from .utils import get_random_subset, reduce_dimension
import os
from jax import numpy as jnp
from pennylane import numpy as qnp
import random


def _load_and_filter_data(train_data, test_data, label1, label2):
    # Separar características (X) y etiquetas (y)
    x_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    x_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

    # Filtrar las filas que corresponden a las etiquetas deseadas
    train_filter = y_train.isin([label1, label2])
    test_filter = y_test.isin([label1, label2])

    # Aplicar el filtro
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    # Convertir las etiquetas a 0 y 1
    y_train = y_train.apply(lambda y: 0 if y == label1 else 1)
    y_test = y_test.apply(lambda y: 0 if y == label1 else 1)

    # Normalizar los píxeles entre 0 y 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convertir a listas si es necesario
    x_train, y_train = x_train.values.tolist(), y_train.values.tolist()
    x_test, y_test = x_test.values.tolist(), y_test.values.tolist()

    return x_train, y_train, x_test, y_test


def process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface):

    # Reduce x sets to 4 dimensions
    x_train = reduce_dimension(x_train, new_dim=points_dimension, feature_range=(-1, 1))
    x_test = reduce_dimension(x_test, new_dim=points_dimension, feature_range=(-1, 1))

    if interface == 'jax':
        x_train = jnp.array(x_train)
        y_train = jnp.array(y_train)
        x_test = jnp.array(x_test)
        y_test = jnp.array(y_test)
    elif interface == 'pennylane':
        x_train = qnp.array(x_train, requires_grad=False)
        y_train = qnp.array(y_train, requires_grad=False)
        x_test = qnp.array(x_test, requires_grad=False)
        y_test = qnp.array(y_test, requires_grad=False)
    else:  # interface = normal numpy
        pass

    return x_train, y_train, x_test, y_test


class MNISTSampler:

    @staticmethod
    def fashion(n_train: int = 2000, n_test: int = 1000, points_dimension: int = 4,
                label1: int = 3, label2: int = 6, folder: str = None,
                seed: int = None, interface: str = 'jax') -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Retrieves and processes a subset of the MNIST dataset for binary classification between two specified labels.

        Args:
        n_train (int): Number of training samples to retrieve. Defaults to 2000.
        n_test (int): Number of test samples to retrieve. Defaults to 1000.
        label1 (int): The first label for binary classification. Defaults to 3 (# dress)
        label2 (int): The second label for binary classification. Defaults to 6 (# shirt)
        folder (str, optional): The folder path where the MNIST data files are located. If None, defaults to a 'mnist'
        folder in the current directory.
        interface (str, optional): The interface (of numpy) to get the data. Defaults to 'jax'. Available = 'jax',
        'pennylane' or 'numpy'

        Returns:
        x_train,
        y_train,
        x_test,
        y_test
        """

        if interface not in ['pennylane', 'jax', 'numpy']:
            raise ValueError(f"Invalid interface: {interface}. Available interfaces are 'jax', 'pennylane' or 'numpy'.")

        if seed is not None:
            random.seed(seed)

        if folder is None:
            folder = os.path.join(get_current_folder(), 'mnist')

        train_file = os.path.join(folder, 'fashion-mnist_train.csv')
        test_file = os.path.join(folder, 'fashion-mnist_test.csv')

        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        x_train, y_train, x_test, y_test = _load_and_filter_data(train_data, test_data, label1, label2)

        # Random subsets
        x_train, y_train = get_random_subset(x_train, y_train, n_train, seed=seed)
        x_test, y_test = get_random_subset(x_test, y_test, n_test, seed=seed)

        return process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface)

    @staticmethod
    def digits(n_train: int = 2000, n_test: int = 1000, points_dimension: int = 4,
               label1: int = 8, label2: int = 0,
               seed: int = None, interface: str = 'jax') -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Load the MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        mask = (y == label1) | (y == label2)
        X_filtered, y_filtered = X[mask], y[mask]
        y_filtered = jnp.where(y_filtered == label1, 0, 1)

        # Random subsets
        x_train, y_train = get_random_subset(X_filtered, y_filtered, n_train, seed=seed)
        x_test, y_test = get_random_subset(X_filtered, y_filtered, n_test, seed=seed)

        return process_dataset(x_train, y_train, x_test, y_test, points_dimension, interface)