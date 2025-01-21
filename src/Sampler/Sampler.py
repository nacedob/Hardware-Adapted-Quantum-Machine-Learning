import random
from jax import numpy as jnp
from pennylane import numpy as np
import jax
from random import randint
from .utils import generate_random_points



class Sampler:

    @staticmethod
    def circle(n_points=10, radius: float = 0.5, center: list = [0, 0], spread: float = 1, interface='jax',
               seed: int = None):
        """
        Generates 2D points and classifies whether they are inside or outside a given circle.

        Parameters:
            n_points (int): Number of points to generate.
            radius (float): Radius of the circle.
            center (list): Center of the circle [x, y].
            spread (float): Spread parameter for the generated points.
            interface (str): Interface for handling arrays ('jax', 'numpy', etc.).
            seed (int): Seed for reproducibility.

        Returns:
            tuple: (points, labels) where:
                - points is an array of shape (n_points, 2).
                - labels is a binary array indicating whether each point is inside (1) or outside (0) the circle.
        """
        # Unpack center coordinates
        center_x, center_y = center

        # Generate random points
        points = generate_random_points(n_points, spread, 2, interface, seed)

        # Label points based on their position relative to the circle
        labels = jnp.array(
            [((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2) for x, y in points],
            dtype=int)

        return points, labels

    @staticmethod
    def stripes(n_points=10, stripe_width: float = 0.1, stripe_angle: float = 0.0, spread: float = 1,
                separation: float = 0.2, n_stripes=None, center=None, interface='jax', seed: int = None):
        """
        Generates 2D points and classifies whether they are inside or outside given stripes.

        Parameters:
            n_points (int): Number of points to generate.
            stripe_width (float): Width of each stripe.
            stripe_angle (float): Angle of the stripes in radians with respect to the x-axis.
            spread (float): Spread parameter for the generated points. Defines the range of random points.
            separation (float): Separation between consecutive stripes.
            n_stripes (int, optional): Number of stripes to generate. If None, the function will generate as many stripes 
                                      as fit with the given separation.
            center (tuple, optional): Center of the stripes. If None, the center of the (n-1)//2-th stripe is placed at (0, 0).
            interface (str): Interface for handling arrays ('jax', 'numpy', etc.).
            seed (int): Seed for reproducibility.

        Returns:
            tuple: (points, labels) where:
                - points is an array of shape (n_points, 2).
                - labels is a binary array indicating whether each point is inside (1) or outside (0) any stripe.
        """
        # Generate random points
        points = generate_random_points(n_points, spread, 2, interface, seed)

        # Calculate the rotation matrix for the given stripe angle
        cos_theta = jnp.cos(stripe_angle)
        sin_theta = jnp.sin(stripe_angle)

        # Apply the rotation to the points to align with the stripe direction
        rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_points = jnp.dot(points, rotation_matrix.T)

        # Determine how many stripes can fit within the range of [-spread, spread]
        total_range = 2 * spread  # Range of the points
        max_stripes = int(1.5 * total_range / (stripe_width + separation))  # Number of stripes that fit
        if n_stripes is None:
            n_stripes = max_stripes  # Use as many stripes as possible with the given spread and stripe width

        # Determine the starting position for stripes
        if center is None:
            # If no center is provided, place the center of the (n-1)//2-th stripe at (0, 0)
            center = (0, 0)  # The center of the middle stripe should be at (0, 0)

        # Ensure the stripes fit symmetrically around the center
        start_position = center[1] - ((n_stripes - 1) // 2) * (stripe_width + separation)

        # Generate labels based on multiple stripes
        labels = jnp.zeros(n_points, dtype=int)
        for i in range(n_stripes):
            # Calculate the center of the current stripe (based on separation)
            stripe_center = start_position + i * (stripe_width + separation)

            # If the stripe center exceeds the valid range, stop adding stripes
            # if stripe_center > spread or stripe_center < -spread:
            #     break

            # Check if the point is within the stripe's width
            labels = jnp.logical_or(labels, jnp.abs(rotated_points[:, 1] - stripe_center) <= stripe_width / 2)

        return points, labels.astype(int)


    @staticmethod
    def annulus(n_points: int = 10, seed: int = None,
                inner_radius: float = 0.5 * (2 / jnp.pi) ** 0.5,
                outer_radius: float = (2 / jnp.pi) ** 0.5,
                center: list = [0, 0],
                scale: float = 1,
                interface: str = 'jax'):
        """
        Generates 2D points and classifies whether they are inside or outside a given annulus.

        Parameters:
            n_points (int): Number of points to generate.
            seed (int): Seed for reproducibility.
            inner_radius (float): Inner radius of the annulus.
            outer_radius (float): Outer radius of the annulus.
            center (list): Center of the annulus [x, y].
            scale (float): Scaling factor for the generated points.
            interface (str): Interface for handling arrays ('jax', 'numpy', etc.).

        Returns:
            tuple: (points, labels) where:
                - points is an array of shape (n_points, 2).
                - labels is a binary array indicating whether each point is inside (1) or outside (0) the annulus.
        """
        # Unpack center coordinates
        center_x, center_y = center

        # Generate random points
        points = generate_random_points(n_points, scale, 2, interface, seed)

        # Compute squared distances from the center
        squared_distances = jnp.array([(x - center_x) ** 2 + (y - center_y) ** 2 for x, y in points])

        # Label points based on their position relative to the annulus
        labels = jnp.array(
            [(inner_radius ** 2 <= d <= outer_radius ** 2) for d in squared_distances],
            dtype=int)

        return points, labels

    @staticmethod
    def multi_circle(n_points: int = 10, centers: list = [[0, 0]],
                     radii: list = [1.0], spread: float = 1,
                     interface: str = 'jax', seed: int = None):
        """
        Generates 2D points and classifies whether they are inside any of the given circles.

        Parameters:
            n_points (int): Number of points to generate.
            centers (list): List of circle centers, where each center is [x, y].
            radii (list): List of radii for the circles.
            spread (float): Spread parameter for the generated points.
            interface (str): Interface for handling arrays ('jax', 'numpy', etc.).
            seed (int): Seed for reproducibility.

        Returns:
            tuple: (points, labels) where:
                - points is an array of shape (n_points, 2).
                - labels is a binary array indicating whether each point is inside any circle (1) or not (0).
        """
        # Generate random 2D points
        points = generate_random_points(n_points, spread, 2, interface, seed)

        # Initialize labels as zero
        labels = jnp.zeros(n_points, dtype=int)

        # Loop through each circle and update labels
        for (center_x, center_y), radius in zip(centers, radii):
            squared_radius = radius ** 2
            for i, (x, y) in enumerate(points):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= squared_radius:
                    labels = labels.at[i].set(1)

        return points, labels

    @staticmethod
    def sinus(n_points, amplitude: float = 1, freq: float = np.pi, offset_phase: float = 0, offset_sin: float = 0, spread: float = 1, seed:int = None, interface: str = 'jax'):

        """
        Generates a dataset of points with 1/0 labels
        depending on whether they are above or below the sine function
        Args:
            n_points (int): number of samples to generate

        Returns:
            Xvals (array[tuple]): coordinates of points
            yvals (array[int]): classification labels
        """
        points = generate_random_points(n_points, spread, 2, interface, seed)
        x1, x2 = points[:, 0], points[:, 1]
        boundary = - amplitude * np.sin(freq * x1 + offset_phase) + offset_sin
        y = jnp.where(x2 > boundary, 1, 0)
        return points, y


    @staticmethod
    def corners(n_points, spread: float = 1, seed:int = None, interface:str = 'jax'):
        radii = [0.75] * 4
        centers = [[-1,- 1], [-1, 1], [1,- 1], [1, 1]]
        return Sampler.multi_circle(n_points=n_points, radii=radii, centers=centers,
                                    spread=spread, seed=seed, interface=interface)

    @staticmethod
    def spiral(n_points, seed:int = None, interface:str = 'jax'):
        if interface != 'jax':
            raise ValueError("Only JAX interface is supported for this function")

        if seed is None:
            seed = randint(0, 10000)
        key = jax.random.PRNGKey(seed)

        n_points= n_points // 2
        theta = jnp.sqrt(jax.random.uniform(key, shape=(n_points,))) * 2 * jnp.pi # jnp.linspace(0,2*pi,100)

        r_a = 2*theta + jnp.pi
        data_a = jnp.array([jnp.cos(theta)*r_a, jnp.sin(theta)*r_a]).T
        data_a += jax.random.uniform(key, shape=(n_points, 2)) / 10   # width
        x_a = (data_a + jax.random.uniform(key, shape=(n_points, 2)))/20

        r_b = -2*theta - jnp.pi
        data_b = jnp.array([jnp.cos(theta)*r_b, jnp.sin(theta)*r_b]).T
        data_b += jax.random.uniform(key, shape=(n_points, 2)) / 10  # width
        x_b = (data_b + jax.random.uniform(key, shape=(n_points, 2)))/20

        xvals = jnp.append(x_a, x_b, axis=0)
        yvals = jnp.concatenate([jnp.ones(n_points), jnp.zeros(n_points)], axis=0)

        data = list(zip(jnp.array(xvals), jnp.array(yvals)))
        random.shuffle(data)

        equis, yes = zip(*data)

        equis, yes = jnp.array(equis), jnp.array(yes, dtype=int)

        return equis, yes

    @staticmethod
    def rectangle(width: float = 1 / 3, height: float = 1 / 2,
                  center: list = [1 / 2, 1 / 2], n_points: int = 10,
                  seed: int = None, spread: float = 1,
                  interface: str = 'jax'):
        """
        Generates 2D points and classifies whether they are inside a rectangle.

        Parameters:
            width (float): Width of the rectangle.
            height (float): Height of the rectangle.
            center (list): Center of the rectangle [x, y].
            n_points (int): Number of points to generate.
            seed (int): Seed for reproducibility.
            spread (float): Spread parameter for the generated points.
            interface (str): Interface for handling arrays ('jax', 'numpy', etc.).

        Returns:
            tuple: (points, labels) where:
                - points is an array of shape (n_points, 2).
                - labels is a binary array indicating whether each point is inside (1) or outside (0) the rectangle.
        """
        # Calculate rectangle bounds
        center_x, center_y = center
        half_width = width / 2
        half_height = height / 2

        # Generate random 2D points
        points = generate_random_points(n_points, spread, 2, interface, seed)

        # Label points based on their position relative to the rectangle
        labels = jnp.array(
            [(center_x - half_width <= x <= center_x + half_width) and
             (center_y - half_height <= y <= center_y + half_height)
             for x, y in points],
            dtype=int)

        return points, labels
