from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_json_to_dict, get_current_folder
from src.visualization.utils import darken_color, lighten_color
from .colors import dark_violet, light_violet

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)



def plot_bloch_sphere(color=light_violet, alpha=0.35, ax=None, fontsize=15, plot_states:bool=True):
    """
    Initialize and plot a Bloch sphere with a solid color and customizable transparency.
    Adds labels for |0⟩ and |1⟩ at the north and south poles of the sphere.

    Parameters:
    - color (str): The solid color of the Bloch sphere surface (e.g., 'blue', '#FF5733').
    - alpha (float): Transparency of the Bloch sphere surface.

    Returns:
    - ax (mpl_toolkits.mplot3d.Axes3D): A Matplotlib 3D axis to add points/vectors.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Generate the sphere surface
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x_sphere = np.sin(v) * np.cos(u)
    y_sphere = np.sin(v) * np.sin(u)
    z_sphere = np.cos(v)

    # Plot the sphere surface with a solid color
    ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=5, cstride=5, color=color, alpha=alpha, zorder=3)

    # Add |0⟩, |1⟩ |+⟩, |-⟩, |+i⟩, and |-i⟩ labels at the north and south poles
    if plot_states:
        ax.text(0, 0, 1.13, r'$|0\rangle$', color='black', fontsize=fontsize, ha='center', va='center', zorder=2)
        ax.text(0, 0, -1.13, r'$|1\rangle$', color='black', fontsize=fontsize, ha='center', va='center', zorder=2)
        ax.text(1.13, 0, 0, r'$|+\rangle$', color='black', fontsize=fontsize, ha='center', va='center', zorder=2)
        ax.text(-1.13, 0, 0, r'$|-\rangle$', color='black', fontsize=fontsize, ha='center', va='center', zorder=2)
        ax.text(0, 1.13, 0, r'$|+i\rangle$', color='black', fontsize=fontsize, ha='center', va='center', zorder=2)
        ax.text(0, -1.13, 0, r'$|-i\rangle$', color='black', fontsize=fontsize, ha='center', va='center', zorder=2)
    # Set axis limits and labels
    ax.view_init(elev=10, azim=20)  # angles in degrees
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.grid(False)  # Turn off default grid for a clean look
    ax.set_axis_off()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    return ax


def plot_point(state, ax, color: str = dark_violet, color_edge: str = None, size: int = 70, offset=1.0, marker=None):
    """
    Plot a single quantum state as a point and vector on the Bloch sphere.

    Parameters:
    - state (np.ndarray): A normalized quantum state vector (2D complex array).
    - ax (mpl_toolkits.mplot3d.Axes3D): A Matplotlib 3D axis to plot on.
    """
    # Calculate the Bloch sphere coordinates
    rho = np.outer(state, np.conj(state))  # Density matrix
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    # Apply the offset to ensure the point is slightly outside the sphere
    x, y, z = offset * x, offset * y, offset * z
    # Plot the vector and point
    if color_edge is None:
        color_edge = darken_color(color)
    if marker is None:
        marker = 'o'  # Default marker shape
    ax.scatter(x, y, z, color=color, s=size, edgecolors=color_edge, zorder=1, marker=marker)  # Endpoint marker


def plot_bloch_points(states, sphere_color=light_violet, point_color: [str, list[str]] = dark_violet, point_edgecolor: str = None,
                      pointsize: int = 70, show: bool = True, ax=None, marker=None):
    # Initialize the Bloch sphere with a solid color and transparency
    if ax is None:
        ax = plot_bloch_sphere(color=sphere_color, alpha=0.35)

    if isinstance(point_color, str):
        point_color = [point_color] * len(states)
    # Plot multiple points on the same sphere
    for state, color in zip(states, point_color):
        plot_point(state, ax, color, point_edgecolor, pointsize, marker=marker)


    if show:
        plt.tight_layout()
        plt.show()

    return ax


def bloch_vector_from_density_matrix(rho):
    """
    Calculates the Bloch vector from the density matrix.

    Parameters:
    - rho: A 2x2 numpy array representing the density matrix.

    Returns:
    - bloch_vector: A 3D numpy array representing the Bloch vector.
    """
    # Compute the Bloch vector components
    r_x = np.real(np.trace(rho @ X))
    r_y = np.real(np.trace(rho @ Y))
    r_z = np.real(np.trace(rho @ Z))

    return np.array([r_x, r_y, r_z])


def plot_density_matrix_on_bloch(rho, ax, color: str = dark_violet, color_edge: str = None, size: int = 70,
                                 offset=1.02):
    """
    Plot a single density matrix on the Bloch sphere by calculating its Bloch vector.

    Parameters:
    - rho (np.ndarray): A 2x2 numpy array representing the density matrix.
    - ax (mpl_toolkits.mplot3d.Axes3D): A Matplotlib 3D axis to plot on.
    """
    # Compute the Bloch vector from the density matrix
    bloch_vector = bloch_vector_from_density_matrix(rho)

    # Apply the offset to ensure the point is slightly outside the sphere
    x, y, z = offset * bloch_vector

    # Plot the vector and point
    if color_edge is None:
        color_edge = darken_color(color)  # Default edge color
    ax.scatter(x, y, z, color=color, s=size, edgecolors=color_edge, zorder=1)  # Endpoint marker


def plot_density_matrices_on_bloch(density_matrices, sphere_color=light_violet, point_color=dark_violet,
                                   point_edgecolor=None,
                                   pointsize=70, show=True, ax=None):
    """
    Plot multiple density matrices as points on the Bloch sphere.

    Parameters:
    - density_matrices (list): A list of 2x2 numpy arrays representing density matrices.
    - sphere_color (str): Color of the Bloch sphere.
    - point_color (str or list): Color of the points representing the density matrices.
    - point_edgecolor (str): Edge color of the points.
    - pointsize (int): Size of the points.
    - show (bool): Whether to display the plot.
    """

    # Initialize the Bloch sphere with a solid color and transparency
    if ax is None:
        ax = plot_bloch_sphere(color=sphere_color, alpha=0.5)

    # If point_color is a string, create a list for consistency
    if isinstance(point_color, str):
        point_color = [point_color] * len(density_matrices)

    # Plot multiple density matrices on the same sphere
    for rho, color in zip(density_matrices, point_color):
        plot_density_matrix_on_bloch(rho, ax, color, point_edgecolor, pointsize)

    # Show the plot
    if show:
        plt.tight_layout()
        plt.show()
    return ax