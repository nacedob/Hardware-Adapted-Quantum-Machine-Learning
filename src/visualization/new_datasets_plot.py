from src.experiments.config_exp import get_dataset
from src.visualization import plot_dataset
import matplotlib.pyplot as plt
from src.visualization import dark_violet, pink_dataset
from src.utils import get_root_path
import os
import numpy as np
from icecream import ic
import argparse

parser = argparse.ArgumentParser(description='Plot datasets for final experiment.')
parser.add_argument('--dataset', type=str, default='all', help='Dataset for which to plot results.')
parser.add_argument('--distribution', type=str, default='', help='Distribution for the plots')
args = parser.parse_args()
dataset_arg = args.dataset
distribution = args.distribution

if distribution not in ['', '23']:
    raise ValueError("Invalid distribution. Choose either '' or '23'.")

# distribution = 23   # 2 x 3 distribution
# distribution = ''   # da igual, las pone como 1 x 6

if dataset_arg == 'all':
    datasets = ['fashion', 'digits', 'sinus3D', 'helix', 'shell', 'corners3d']
elif dataset_arg == 'tfm':
    datasets = ['digits', 'fashion', 'shell', 'sinus3D']

if distribution == '23':
    fig = plt.figure(figsize=(len(datasets)//2 * 5, 6 * 2))
    ax = np.array([[fig.add_subplot(2, len(datasets) // 2, i + 1 + j * (len(datasets) // 2), projection='3d')
                    for i in range(len(datasets) // 2)] for j in range(2)]).flatten()
else:
    fig = plt.figure(figsize=(len(datasets) * 5, 6))
    ax = np.array([fig.add_subplot(1, len(datasets), i + 1, projection='3d') for i in range(len(datasets))]).flatten()


fontsize = 28

seed = 0
root = get_root_path()
if distribution == '23':
    path = f'{root}/data/datasets/datasets__{dataset_arg}2x3.png'
else:
    path = f'{root}/data/datasets/datasets_{dataset_arg}.png'
os.makedirs(os.path.dirname(path), exist_ok=True)


for i, dataset in enumerate(datasets):
    if dataset in ['fashion', 'digits']:
        n_points = 1000
    elif dataset in ['helix', 'shell']:
        n_points = 1500
    else:
        n_points = 2500

    data, labels, _, _ = get_dataset(dataset, n_train=n_points, n_test=6, points_dimension=3, seed=seed, interface='jax')
    ax[i] = plot_dataset(data, labels, color0=dark_violet, color1=pink_dataset,
                         axis=ax[i], show=False, show_legend=False, fontsize=fontsize)

    if dataset == 'fashion':
        title = 'MNIST Fashion'
    elif dataset == 'digits':
        title = 'MNIST Digits'
    elif dataset == 'shell':
        title = 'Spherical Shell'
    elif dataset == 'corners3d':
        title = 'Corners'
    elif dataset == 'sinus3D':
        title = 'Sinus'
    elif dataset == 'helix':
        title = 'Helix'
    else:
        title = dataset.capitalize()
    ax[i].set_title(title, fontsize=fontsize, fontfamily='Arial')

ax_lines, ax_labels = ax[-1].get_legend_handles_labels()
ax[-1].legend(ax_lines, ax_labels, loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=fontsize-7)

fig.tight_layout(pad=2)
fig.savefig(path)
plt.close(fig)
