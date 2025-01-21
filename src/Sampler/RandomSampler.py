from typing import Sequence
from icecream import ic
from src.Sampler.utils import reduce_dimension
from sklearn.datasets import make_classification
from random import randint
from sklearn.model_selection import train_test_split
from jax import numpy as jnp


class RandomSampler:

    @staticmethod
    def get_data(dimension: int = 3, n_train: int = 1000, n_test: int = 1000, seed: int = None,
                 n_features: int = 3, n_informative: int = 3, n_redundant: int = 0, n_cluster_per_class: int = 1,
                 flip_y: float = 0.01, class_sep: float = 1.0, interface:str = 'jax',
                 feature_range: tuple =(-1, 1)) -> tuple[Sequence, Sequence, Sequence, Sequence]:

        if seed is None:
            seed = randint(0, 1000)

        # Create synthetic model
        x, y = make_classification(n_samples=n_train + n_test,
                                   n_classes=2,
                                   n_features=n_features,
                                   n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_clusters_per_class=n_cluster_per_class,
                                   flip_y=flip_y,
                                   class_sep=class_sep,
                                   random_state=seed)

        if x.shape[1] < dimension:
            raise ValueError("Dimension of data must be greater than or equal to the number of features.")
        elif x.shape[1] > dimension:
            x = reduce_dimension(x, dimension, feature_range=feature_range)

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=n_train, test_size=n_test,
                                                            random_state=seed)

        # Parse to jax
        if interface == 'jax':
            x_train = jnp.array(x_train.tolist())
            x_test = jnp.array(x_test.tolist())
            y_train = jnp.array(y_train.tolist())
            y_test = jnp.array(y_test.tolist())
        else:
            raise ValueError("Invalid interface. Supported interfaces are 'jax' and 'numpy'.")

        return x_train, y_train, x_test, y_test

    @staticmethod
    def easy_problem(dimension: int = 3, n_train: int = 1000, n_test: int = 1000, seed: int = None,
                     interface:str = 'jax') -> tuple[Sequence, Sequence, Sequence, Sequence]:
        x_train, y_train, x_test, y_test = RandomSampler.get_data(dimension=dimension,
                                                                  n_train=n_train,
                                                                  n_test=n_test,
                                                                  seed=seed,
                                                                  n_features=dimension,
                                                                  n_informative=dimension,
                                                                  n_redundant=0,
                                                                  n_cluster_per_class=1,
                                                                  class_sep=1.5,
                                                                  flip_y=0,
                                                                  interface=interface)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def medium_problem(dimension: int = 3, n_train: int = 1000, n_test: int = 1000, seed: int = None,
                       interface:str = 'jax') -> tuple[Sequence, Sequence, Sequence, Sequence]:
        x_train, y_train, x_test, y_test = RandomSampler.get_data(dimension=dimension,
                                                                  n_train=n_train,
                                                                  n_test=n_test,
                                                                  seed=seed,
                                                                  n_features=dimension * 3,
                                                                  n_informative=dimension * 2,
                                                                  n_redundant=dimension // 2,
                                                                  n_cluster_per_class=3,
                                                                  flip_y=0.01,
                                                                  interface=interface)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def hard_problem(dimension: int = 3, n_train: int = 1000, n_test: int = 1000, seed: int = None,
                     interface:str = 'jax') -> tuple[Sequence, Sequence, Sequence, Sequence]:
        x_train, y_train, x_test, y_test = RandomSampler.get_data(dimension=dimension,
                                                                  n_train=n_train,
                                                                  n_test=n_test,
                                                                  seed=seed,
                                                                  n_features=dimension * 5,
                                                                  n_informative=dimension * 2,
                                                                  n_redundant=dimension,
                                                                  n_cluster_per_class=5,
                                                                  flip_y=0.03,
                                                                  interface=interface)
        return x_train, y_train, x_test, y_test
