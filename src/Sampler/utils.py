import pickle
import json
import os
from icecream import ic
import numpy as np
import jax
from jax import numpy as jnp
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from typing import Sequence


def generate_random_points(n_points, spread, point_size: int, interface: str = 'jax', seed: int = None):
    if interface == 'jax':
        if seed is None:
            seed = np.random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)
        return spread * (2 * jax.random.uniform(key, shape=(n_points, point_size)) - 1)
    elif interface == 'pennylane':
        return spread * (2 * np.random.rand(n_points, point_size) - 1)


def get_random_subset(x_data, y_data, n_samples, seed=None):
    """
    Toma un conjunto de datos y etiquetas, y devuelve un subconjunto aleatorio de tamaño n_samples.

    Args:
    - x_data (list): Lista de características (imágenes normalizadas).
    - y_data (list): Lista de etiquetas correspondientes.
    - n_samples (int): Número de muestras deseadas en el subconjunto.

    Returns:
    - (list, list): Subconjunto de características y etiquetas.
    """
    if n_samples > len(x_data):
        raise ValueError(f"El tamaño solicitado ({n_samples}) excede el número total de muestras ({len(x_data)}).")

    if seed is not None:
        random.seed(seed)

    # Obtener índices aleatorios
    subset_indices = random.sample(range(len(x_data)), n_samples)

    # Seleccionar las muestras correspondientes
    x_subset = [x_data[i] for i in subset_indices]
    y_subset = [y_data[i] for i in subset_indices]

    return x_subset, y_subset

def reduce_dimension(data, new_dim: int = 4, feature_range: tuple = None) -> Sequence:
    """
    Reduce the dimensionality of the input data using PCA and optionally scale the features.

    This function applies a pipeline of data preprocessing steps:
    1. Standardization of the input data.
    2. Principal Component Analysis (PCA) for dimensionality reduction.
    3. Optional scaling of the reduced data to a specified feature range.

    Parameters:
    data (array-like): The input data to be reduced in dimensionality.
    new_dim (int, optional): The number of dimensions to reduce the data to. Defaults to 4.
    feature_range (tuple, optional): The desired range of transformed data.
                                     If provided, the data will be scaled to this range.
                                     Defaults to None (no scaling).

    Returns:
    numpy.ndarray: The transformed data with reduced dimensionality and optional scaling.
    """
    steps = [
        ('normalize', StandardScaler()),
        ('pca', PCA(n_components=new_dim, svd_solver='full')),
    ]
    if feature_range is not None:
        steps += [('scaler', MinMaxScaler(feature_range=feature_range))]  # for instance (-1, 1)
    pipeline = Pipeline(steps)
    data_reduced = pipeline.fit_transform(data)
    return data_reduced

def scale_points(dataset: np.ndarray, scale_range: tuple = None, center: bool=False) -> np.ndarray:
    """
    Given a dataset, it scales the points so they are in the scale_range (-1, 1). This is done to create
    a meaningful encoder -> Rot(pi * x)
    :param dataset:
    :return: scaled dataset
    """
    if scale_range is None:
        scale_range = (-1, 1)
    scaler = MinMaxScaler(feature_range=scale_range)
    scaled_data = scaler.fit_transform(dataset)
    if center:
        scaler = StandardScaler(with_mean=True, with_std=False)
        scaled_data = scaler.fit_transform(scaled_data)

    return scaled_data



# Print Utils
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

# Files utils
def get_root_path(project_name):

    # Get the absolute path of the current file

    # Get the current working directory
    current_working_dir = os.getcwd()

    # Split the path into parts
    path_parts = current_working_dir.split(os.sep)

    # Find the index of the target directory
    try:
        target_index = path_parts.index(project_name)
    except ValueError:
        raise ValueError(f"Directory '{project_name}' not found in the current working directory path.")

    # Join the path up to and including the target directory
    source_directory = os.sep.join(path_parts[:target_index + 1])
    source_directory = source_directory.replace('\\', '/')

    return source_directory


def _process_extension(path, extension='.json'):
    # Check if the path has the specified extension
    if not path.endswith(extension):
        # If path has no extension, add the specified extension
        if '.' not in path:
            return path + extension
        # If path has an extension but it does not match the specified one, raise an error
        else:
            raise ValueError(f"Expected file extension {extension}, but got a different extension.")
    return path

def save_dict_to_json(dict:dict, path:str):
    path = _process_extension(path, extension='.json')
    with open(path, 'w') as file:
        json.dump(dict, file, indent=4)

def load_json_to_dict(path:str):
    path = _process_extension(path, extension='.json')
    with open(path, 'r') as file:
        return json.load(file)

def save_pickle(obj, filename):
    filename = _process_extension(filename, extension='.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(filename):
    filename = _process_extension(filename, extension='.pkl')
    with open(filename, 'rb') as file:
        return pickle.load(file)