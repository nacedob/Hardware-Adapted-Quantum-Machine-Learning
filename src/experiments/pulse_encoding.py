"""
Aqui estudio como hacer el encoding para puntos 2D y 3D usando pulsos aplicados a un solo pulso
"""
from .config_exp import get_dataset
import pennylane as qml
from math import pi
import jax
from src.Sampler.utils import scale_points
from math import pi
from jax import numpy as jnp
try:
    import pennypulse
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "src/Pennypulse"])
    import pennypulse
from icecream import ic
from src.visualization import dark_violet, light_violet, pink_dataset
from src.visualization import plot_bloch_points, plot_3d_dataset
from src.visualization.bloch_sphere import plot_bloch_sphere
from src.Sampler import MNISTSampler, Sampler3D
import matplotlib.pyplot as plt
import argparse
import ast
import os
import random

FONTSIZE = 23

exp_folder = 'data/results/encoding'
os.makedirs(exp_folder, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--show', type=str, help='whether to show results', required=False, default='False')
parser.add_argument('--seed', type=int, help='random seed', required=False, default=42)
parser.add_argument('--n_points', type=int, help='whether n_points results', required=False, default=1000)
parser.add_argument('--dataset', type=str, help='dataset to plot', required=False, default=None)
parser.add_argument('--random_init', type=str, help='Start the encoding in a random state', required=False, default='False')
args = parser.parse_args()
SHOW = ast.literal_eval(args.show)
random_init = ast.literal_eval(args.random_init)
seed = args.seed
dataset = args.dataset
n_points = args.n_points
random.seed(seed)


def get_encoding_by_labels(points, labels, encoding_func: callable, ax=None, marker=None):
    points = scale_points(points, (-1, 1))
    points_0 = points[labels == 0]
    points_1 = points[labels == 1]

    gate_encoding_vect = jax.jit(jax.vmap(encoding_func))
    enc_states_0 = gate_encoding_vect(points_0)
    enc_states_1 = gate_encoding_vect(points_1)

    ax = plot_bloch_sphere(ax=ax, alpha=0.3, fontsize=17)
    ax = plot_bloch_points(enc_states_0, point_color=dark_violet, marker=None, show=False, ax=ax)
    ax = plot_bloch_points(enc_states_1, point_color=pink_dataset, marker='s', show=False, ax=ax)
    ax.view_init(elev=10, azim=40)  # angles in degrees
    return ax


################################
# Encoding with gates
################################
random_params = [random.random(), random.random()]
random_state = jnp.array([jnp.cos(random_params[0]), jnp.exp(1j * random_params[1]) * jnp.sin(random_params[0])])


@qml.qnode(qml.device('default.qubit', wires=range(1)), interface='jax')
def gate_encoding(x):
    if random_init:
        qml.QubitStateVector(random_state, wires=range(1))
    if len(x) == 2:
        x = jnp.array([*x, 0])
    qml.Rot(*(x * pi), wires=0)
    return qml.state()


################################
# Encoding with pulses
################################
freq = 1
duration = 10


@qml.qnode(qml.device('default.qubit', wires=range(1)), interface='jax')
def pulse_encoding(x):
    if random_init:
        qml.QubitStateVector(random_state, wires=range(1))

    norm = jnp.linalg.norm(x)
    azimutal = jnp.arctan2(x[1], x[0])
    polar = jnp.where(norm != 0, jnp.arccos(x[2] / norm), 0.0)

    amp = norm / jnp.sqrt(3) * jnp.pi / duration  # maximum angle is 2 pi
    amplitude_function = pennypulse.shapes.constant(amp)

    pennypulse.pulse1q(q_freq=freq,
                       drive_freq=freq,
                       amplitude_func=amplitude_function,
                       drive_phase=polar,
                       duration=duration,
                       wire=0)
    qml.RZ(azimutal, wires=0)
    return qml.state()


if __name__ == '__main__':

    # datasets
    if dataset is None:
        points_fashion, labels_fashion, _, _ = MNISTSampler.fashion(n_train=n_points, n_test=10, points_dimension=3,
                                                                    label1=3, label2=6, seed=seed)
        points_digits, labels_digits, _, _ = MNISTSampler.digits(n_train=n_points, n_test=10, points_dimension=3,
                                                                 label1=0, label2=8, seed=seed)
        points_shell, labels_shell = Sampler3D.shell(n_points=n_points, seed=seed)
        points_helix, labels_helix = Sampler3D.helix(n_points=n_points, seed=seed, z_speed=0.5)
        points_sinus3d, labels_sinus3d = Sampler3D.sinus3d(n_points=n_points, seed=seed)
        points_corners3d, labels_corners3d = Sampler3D.corners3d(n_points=n_points, seed=seed)
        datasets = [
            [points_fashion, labels_fashion, 'fashion'],
            [points_digits, labels_digits, 'digits'],
            [points_shell, labels_shell, 'shell'],
            [points_corners3d, labels_corners3d, 'corners3d'],
            [points_helix, labels_helix, 'helix'],
            [points_sinus3d, labels_sinus3d, 'sinus3d'],
        ]
    else:
        points, labels, _, _ = get_dataset(dataset, n_points, 10, 'jax', 3, seed, scale=pi)
        datasets = [[points, labels, dataset]]

    figs = []
    for dataset in datasets:
        points, labels, dataset_name = dataset
        fig = plt.figure(figsize=(20, 6), constrained_layout=True)
        ax = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, 4)]

        # Plot dataset
        plot_3d_dataset(points, labels, label_0=0, label_1=1, color0=dark_violet, color1=pink_dataset, axis=ax[0], show=False)
        ax[0].set_aspect("equal")
        ax[0].set_title('Dataset', fontsize=FONTSIZE)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=FONTSIZE-10, rotation=45)
        ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=FONTSIZE-10, rotation=-15)
        ax[0].set_zticklabels(ax[0].get_zticklabels(), fontsize=FONTSIZE-10)
        ax[0].legend(loc='center left', bbox_to_anchor=(-0.25, 0.5), ncol=1, fontsize=FONTSIZE-6)
        # Plot encoding points
        get_encoding_by_labels(points, labels, encoding_func=pulse_encoding, ax=ax[1])
        ax[1].set_title('Pulse encoding', fontsize=FONTSIZE)
        get_encoding_by_labels(points, labels, encoding_func=gate_encoding, ax=ax[2])
        ax[2].set_title('Gate encoding', fontsize=FONTSIZE)
        if '3d' in dataset_name:
            dataset_name_ = dataset_name[:-2]
        else:
            dataset_name_ = dataset_name
        fig.suptitle(f'Dataset: {dataset_name_}', fontsize=FONTSIZE)
        # fig.tight_layout()
        fig.savefig(f'{exp_folder}/{dataset_name}.png')

    if SHOW:
        plt.show()
