from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from warnings import warn
from src.Sampler.utils import reduce_dimension
from src.visualization.utils import darken_color, lighten_color
from .colors import dark_violet, light_violet, pink_dataset, pink

# Obtener colores por defecto
def_color_0 = light_violet
def_color_1 = dark_violet


def plot_3d_dataset(data, labels, label_0: int = 0, label_1: int = 1, color0=def_color_0, color1=def_color_1,
                    axis=None, title: str = None, show_legend: bool = False, show: bool = True, pointsize:float = 100,
                    fontsize: int = 14):
    """
    Plot a 3D dataset with two classes of points.

    This function creates a 3D scatter plot of a dataset, separating the points into two classes
    based on their labels. It's designed for visualizing binary classification problems in 3D space.

    Parameters:
    -----------
    data : numpy.ndarray
        The input dataset. It should be a 2D array where each row represents a data point
        and each column represents a feature. The array must have exactly 3 columns.
    labels : numpy.ndarray
        An array of labels corresponding to each data point. It should have the same length as data.
    label_0 : int, optional
        The label value for the first class (default is 0).
    label_1 : int, optional
        The label value for the second class (default is 1).
    color0 : str, optional
        The color to use for plotting points of the first class (default is def_color_0).
    color1 : str, optional
        The color to use for plotting points of the second class (default is def_color_1).
    title : str, optional
        The title of the plot. If None, no title is displayed (default is None).
    show_legend : bool, optional
        Whether to show the legend on the plot (default is False).

    Raises:
    -------
    ValueError
        If the input data does not have exactly 3 columns or if the length of data and labels are not equal.

    Returns:
    --------
    None
        This function does not return any value. It displays the plot using plt.show().
    """

    if pointsize is None:
        pointsize = 100

    if data.shape[1] != 3:
        raise ValueError("Dataset shape[1] != 3")

    if len(data) != len(labels):
        raise ValueError("Length of data and labels must be equal")

    # Separar puntos por etiquetas
    data_0 = data[labels == label_0]
    data_1 = data[labels == label_1]

    # Crear figura 3D
    if axis is None:
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection='3d')

    # Plotear los puntos
    edge_color0 = darken_color(color0)
    edge_color1 = darken_color(color1)
    axis.scatter(data_0[:, 0], data_0[:, 1], data_0[:, 2], c=color0, edgecolors=edge_color0, s=pointsize, linewidth=1.5,
                 alpha=0.4, label=f'{label_0}')
    axis.scatter(data_1[:, 0], data_1[:, 1], data_1[:, 2], c=color1, edgecolors=edge_color1, s=pointsize, linewidth=1.5,
                 alpha=0.4, label=f'{label_1}', marker='s')

    def pi_formatter(value, tick_number):
        return f'{value * np.pi:.1f}'

    # Apply the formatter to the x-axis
    axis.locator_params(axis='x', nbins=5)
    axis.locator_params(axis='y', nbins=5)
    axis.locator_params(axis='z', nbins=5)
    axis.tick_params(axis='x', labelsize=fontsize-12)
    axis.tick_params(axis='y', labelsize=fontsize-12)
    axis.tick_params(axis='z', labelsize=fontsize-12)
    axis.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    axis.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    axis.zaxis.set_major_formatter(FuncFormatter(pi_formatter))
    # axis.set_xticklabels(axis.get_xticks() * np.pi)
    # axis.set_yticklabels(axis.get_yticks() * np.pi)
    # axis.set_zticklabels(axis.get_zticks() * np.pi)

    if show_legend:
        axis.legend()
    if title is not None:
        plt.title(title)
    if show:
        plt.tight_layout()
        plt.show()

    axis.view_init(elev=22, azim=-60)

    return axis


def plot_2d_dataset(data, labels, label_0: int = 0, label_1: int = 1, color0=def_color_0, color1=def_color_1,
                    axis=None, title: str = None, show_legend: bool = False, show: bool = True):
    """
    Plot a 2D dataset with two classes of points.

    This function creates a 2D scatter plot of a dataset, separating the points into two classes
    based on their labels. It's designed for visualizing binary classification problems in 2D space.

    Parameters:
    -----------
    data : numpy.ndarray
        The input dataset. It should be a 2D array where each row represents a data point
        and each column represents a feature. The array must have exactly 2 columns.
    labels : numpy.ndarray
        An array of labels corresponding to each data point. It should have the same length as data.
    label_0 : int, optional
        The label value for the first class (default is 0).
    label_1 : int, optional
        The label value for the second class (default is 1).
    color0 : str, optional
        The color to use for plotting points of the first class (default is def_color_0).
    color : str, optional
        The color to use for plotting points of the second class (default is def_color_1).
    title : str, optional
        The title of the plot. If None, no title is displayed (default is None).
    show_legend : bool, optional
        Whether to show the legend on the plot (default is False).

    Raises:
    -------
    ValueError
        If the input data does not have exactly 2 columns or if the length of data and labels are not equal.

    Returns:
    --------
    None
        This function does not return any value. It displays the plot using plt.show().
    """

    if data.shape[1] != 2:
        raise ValueError("Dataset shape[1] != 2")

    if len(data) != len(labels):
        raise ValueError("Length of data and labels must be equal")

    # Separar puntos por etiquetas
    data_0 = data[labels == label_0]
    data_1 = data[labels == label_1]

    # Crear figura 2D
    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(8, 6))

    # Plotear los puntos
    edge_color0 = darken_color(color0)
    edge_color1 = darken_color(color1)
    axis.scatter(data_0[:, 0], data_0[:, 1], c=color0, edgecolors=edge_color0, s=100, linewidth=1.5, label=f'{label_0}')
    axis.scatter(data_1[:, 0], data_1[:, 1], c=color1, edgecolors=edge_color1, s=100, linewidth=1.5, label=f'{label_1}',
                 marker='s')

    if show_legend:
        axis.legend()
    if title is not None:
        axis.set_title(title)
    if show:
        plt.tight_layout()
        plt.show()
    return axis


def plot_dataset(data, labels, label_0: int = 0, label_1: int = 1, color0=def_color_0, color1=def_color_1,
                    axis=None, title: str = None, show_legend: bool = False, show: bool = True, pointsize=None,
                 fontsize=14):
    if data.shape[1] == 3:
        axis = plot_3d_dataset(data, labels, label_0, label_1, color0, color1, axis, title, show_legend, show, pointsize, fontsize)
    else:
        axis = plot_2d_dataset(data, labels, label_0, label_1, color0, color1, axis, title, show_legend, show)
    return axis

def visualize_predictions_2d(model: callable, X: np.ndarray, y: np.ndarray,
                             ax=None, title: str = None, save_path: str = None, show: bool = True,
                             fontsize: int = 14, grid_points: int = 25, mapcolor0=def_color_0, mapcolor1=def_color_1,
                             pointcolor0=def_color_0, pointcolor1=def_color_1,
                             legend:bool = False):
    assert hasattr(model, 'forward'), 'model must have a method "forward"'

    if X.shape[1] != 2:
        warn('Dataset provided is not 2D but 3d. Using `reduce_dimension` function. Results may be not correct')
        X_ = reduce_dimension(X, 2)
    else:
        X_ = X

    violets = LinearSegmentedColormap.from_list('violets',
                                                [mapcolor0, mapcolor1])
    x0_min = np.min(X_[:, 0])
    x0_max = np.max(X_[:, 0])
    x1_min = np.min(X_[:, 1])
    x1_max = np.max(X_[:, 1])
    x0x0, x1x1 = np.meshgrid(np.linspace(x0_min, x0_max, 25), np.linspace(x1_min, x1_max, grid_points))
    grid = np.c_[x0x0.ravel(), x1x1.ravel()]
    grid_predictions = model.forward(grid)
    grid_predictions = grid_predictions.reshape(x0x0.shape)
    # grid_predictions = np.array([circuit(params, x) for x in grid])

    if ax is None:
        fig, ax = plt.subplots(1)

    ax.contourf(x0x0, x1x1, grid_predictions, alpha=0.6, cmap=violets)
    ax.set_xlabel("x1", fontsize=fontsize)
    ax.set_ylabel("x2", fontsize=fontsize)
    if title == 'accuracy':
        acc = model.get_accuracy(X, y)
        title = f'Accuracy: {acc}'

    plot_2d_dataset(data=X_, labels=y, show_legend=legend, axis=ax, title=title, show=show,
                    color0=pointcolor0, color1=pointcolor1)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


