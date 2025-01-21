import random
import jax
from jax import numpy as jnp
from pennylane import numpy as np
from .utils import generate_random_points

DEFAULT_N_POINTS = 100
DEFAULT_SPREAD = 1


class Sampler3D:

    @staticmethod
    def torus(n_points: int = DEFAULT_N_POINTS, inner_radius: float = 0.25, outer_radius: float = 0.75,
              spread: float = DEFAULT_SPREAD,
              interface: str = 'jax', seed: int = None):
        assert inner_radius < outer_radius

        def is_inside_torus(point):
            x, y, z = point
            dist_from_center = jnp.sqrt(x ** 2 + y ** 2)
            return (dist_from_center - outer_radius) ** 2 + z ** 2 < inner_radius ** 2

        # Function to generate points on a torus
        points = generate_random_points(n_points, spread, 3, interface, seed)
        labels = jnp.array([is_inside_torus(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def sphere(n_points: int = DEFAULT_N_POINTS, radius: float = 1.0, center: tuple = (0, 0, 0),
               spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        def is_inside_sphere(point):
            x, y, z = point
            cx, cy, cz = center
            return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < radius ** 2

        # Generate random points
        points = generate_random_points(n_points, spread, 3, interface, seed)
        # Classify points
        labels = jnp.array([is_inside_sphere(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def shell(n_points: int = DEFAULT_N_POINTS, inner_radius: float = 0.5, outer_radius: float = 0.9874873050084467,   # tuned to be equally probable classes
              center: tuple = (0, 0, 0), spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        def is_inside_shell(point):
            x, y, z = point
            cx, cy, cz = center
            dist1 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
            dist2 = dist1  # assume spheres share the same center
            return dist1 > inner_radius ** 2 and dist2 < outer_radius ** 2

        # Generate random points
        points = generate_random_points(n_points, spread, 3, interface, seed)

        # Classify points
        labels = jnp.array([is_inside_shell(p) for p in points], dtype=int)

        return points, labels

    @staticmethod
    def spiral(n_points: int = DEFAULT_N_POINTS, turns: int = 5, spread: float = DEFAULT_SPREAD,
               center: tuple = (0, 0, 0), height: float = 1.0, seed: int = None):
        cx, cy, cz = center
        rng = np.random.default_rng(seed)
        t = rng.uniform(0, 2 * np.pi * turns, n_points)
        r = spread * t / (2 * np.pi * turns)  # radius increases with t
        z = height * t / (2 * np.pi * turns)  # height increases with t
        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t)
        points = jnp.array([x, y, z]).T
        labels = jnp.zeros(n_points, dtype=int)  # All points labeled 0 (single class)
        return points, labels

    @staticmethod
    def cube(n_points: int = 100, side_length: float = 1, spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        half_side = side_length / 2
        points = generate_random_points(n_points, spread, 3, interface, seed)
        labels = jnp.array(
            [(abs(x) <= half_side and abs(y) <= half_side and abs(z) <= half_side) for x, y, z in points],
            dtype=int)
        return points, labels

    @staticmethod
    def multi_spheres(n_points: int = DEFAULT_N_POINTS, centers: list = [(0, -0.4, 0.2), (0.1, 0.2, 0)],
                      radii: list = [0.3, 0.5],
                      spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        def is_inside_any_sphere(point):
            return any(jnp.linalg.norm(point - np.array(center)) < r for center, r in zip(centers, radii))

        points = generate_random_points(n_points, spread, 3, interface, seed)
        labels = jnp.array([is_inside_any_sphere(p) for p in points], dtype=int)
        return points, labels
    
    @staticmethod
    def corners3d(n_points: int = DEFAULT_N_POINTS, spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        radii = [0.75] * 8
        centers = [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
        return Sampler3D.multi_spheres(n_points=n_points, radii=radii, centers=centers,
                                    spread=spread, seed=seed, interface=interface)

    @staticmethod
    def cylinder(n_points: int = DEFAULT_N_POINTS, radius: float = 0.8, height: float = 1.5,
                 spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        def is_inside_cylinder(point):
            x, y, z = point
            return (x ** 2 + y ** 2 < radius ** 2)

        points = generate_random_points(n_points, spread, 3, interface, seed)
        labels = jnp.array([is_inside_cylinder(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def ellipsoid(n_points: int = DEFAULT_N_POINTS, a: float = 1, b: float = 2, c: float = 0.2,
                  spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        def is_inside_ellipsoid(point):
            x, y, z = point
            return (x ** 2 / a ** 2 + y ** 2 / b ** 2 + z ** 2 / c ** 2) < 1

        points = generate_random_points(n_points, spread, 3, interface, seed)
        labels = jnp.array([is_inside_ellipsoid(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def pyramid(n_points: int = DEFAULT_N_POINTS, base_size: float = 0.7, height: float = 1.2,
                spread: float = DEFAULT_SPREAD, interface: str = 'jax', seed: int = None):
        def is_inside_pyramid(point):
            x, y, z = point
            return (z >= 0) and (z <= height) and (abs(x) <= (1 - z / height) * base_size / 2) and (
                    abs(y) <= (1 - z / height) * base_size / 2)

        points = generate_random_points(n_points, spread, 3, interface, seed)
        labels = jnp.array([is_inside_pyramid(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def helix(n_points, radius: float = 1, z_speed:float = None, ang_speed: float = 4, noise: bool = True,
              seed: int = None, interface: str = 'jax'):
        if interface != 'jax':
            raise ValueError("Only JAX interface is supported for this function")

        if seed is None:
            seed = random.randint(0, 10000)
            random.seed(seed)

        key = jax.random.PRNGKey(seed)

        if z_speed is None:
            z_speed = ang_speed / np.pi   # a perfect circle

        n_points = n_points // 2
        # Generate theta values
        theta = jnp.sqrt(jax.random.uniform(key, shape=(n_points,))) * 2 / z_speed

        # Helix 1
        x_a = jnp.cos(ang_speed * theta) * radius
        y_a = jnp.sin(ang_speed * theta) * radius
        z_a = -1 + theta * z_speed
        data_a = jnp.array([x_a, y_a, z_a]).T
        data_a += jax.random.uniform(key, shape=(n_points, 3)) / 10   # width
        if noise:
            data_a += jax.random.uniform(key, shape=(n_points, 3)) / 20

        # Helix 2 (opposite direction)
        x_b = jnp.cos(- ang_speed * theta + np.pi) * radius
        y_b = jnp.sin(- ang_speed * theta) * radius
        z_b = -1 + theta * z_speed
        data_b = jnp.array([x_b, y_b, z_b]).T
        data_b += jax.random.uniform(key, shape=(n_points, 3)) / 10  # width
        if noise:
            data_b += jax.random.uniform(key, shape=(n_points, 3)) / 20

        # Combine points
        xvals = jnp.append(data_a, data_b, axis=0)
        yvals = jnp.concatenate([jnp.ones(n_points), jnp.zeros(n_points)], axis=0)

        # Shuffle and split into features and labels
        data = list(zip(jnp.array(xvals), jnp.array(yvals)))
        random.shuffle(data)

        equis, yes = zip(*data)
        equis, yes = jnp.array(equis), jnp.array(yes, dtype=int)

        return equis, yes
    
    @staticmethod
    def butterfly(n_points, speed:float = 0.1, noise: bool = True,
              seed: int = None, interface: str = 'jax'):
        if interface != 'jax':
            raise ValueError("Only JAX interface is supported for this function")

        if seed is None:
            seed = random.randint(0, 10000)
            random.seed(seed)

        key = jax.random.PRNGKey(seed)


        n_points = n_points // 2
        # Generate theta values
        theta = jnp.sqrt(jax.random.uniform(key, shape=(n_points,))) * 2 / speed

        # Helix 1
        x_a = jnp.cos(theta) * (-1 + theta * speed)
        y_a = jnp.sin(theta) * (-1 + theta * speed)
        z_a = -1 + theta * speed
        data_a = jnp.array([x_a, y_a, z_a]).T
        data_a += jax.random.uniform(key, shape=(n_points, 3)) / 10   # width
        if noise:
            data_a += jax.random.uniform(key, shape=(n_points, 3)) / 20

        # Helix 2 (opposite direction)
        x_b = jnp.cos(-theta + np.pi) * (-1 + theta * speed)
        y_b = jnp.sin(-theta) * (-1 + theta * speed)
        z_b = -1 + theta * speed
        data_b = jnp.array([x_b, y_b, z_b]).T
        data_b += jax.random.uniform(key, shape=(n_points, 3)) / 10  # width
        if noise:
            data_b += jax.random.uniform(key, shape=(n_points, 3)) / 20

        # Combine points
        xvals = jnp.append(data_a, data_b, axis=0)
        yvals = jnp.concatenate([jnp.ones(n_points), jnp.zeros(n_points)], axis=0)

        # Shuffle and split into features and labels
        data = list(zip(jnp.array(xvals), jnp.array(yvals)))
        random.shuffle(data)

        equis, yes = zip(*data)
        equis, yes = jnp.array(equis), jnp.array(yes, dtype=int)

        return equis, yes

    @staticmethod
    def sinus3d(n_points, amplitude: float = 1, freq: float = np.pi,
             offset_phase: float = 0, offset_sin: float = 0, 
             spread: float = 1, seed: int = None, direction: int = 1 ,
             interface: str = 'jax'):
        """
        Generates a 3D dataset of points with 1/0 labels
        depending on whether they are above or below a sine-based boundary
        in 3D space.
        
        Args:
            n_points (int): number of samples to generate
            amplitude (float): amplitude of the sine wave
            freq (float): frequency of the sine wave
            offset_phase (float): phase offset for the sine wave
            offset_sin (float): vertical offset for the sine wave
            spread (float): spread of random points
            seed (int): random seed for reproducibility
            interface (str): computation interface ('jax' or 'numpy')
            
        Returns:
            Xvals (array[tuple]): coordinates of points (3D)
            yvals (array[int]): classification labels
        """
        if direction not in range(1, 4):
            raise ValueError("direction must be an integer between 1 and 3")
        # Generate random 3D points
        points = generate_random_points(n_points, spread, 3, interface, seed)
        x1, x2, x3 = points[:, 0], points[:, 1], points[:, 2]

        # Define the boundary surface
        if direction == 1:
            boundary = - amplitude * np.sin(freq * x1 + offset_phase) + offset_sin
        elif direction == 2:
            boundary = amplitude * np.sin(freq * x2 + offset_phase) + offset_sin
        else:  # direction == 3
            boundary = - amplitude * np.cos(freq * x3 + offset_phase) + offset_sin

        # Label points above or below the boundary
        if interface == 'jax':
            y = jnp.where(x3 > boundary, 1, 0)
        else:
            y = np.where(x3 > boundary, 1, 0)

        return points, y
        # boundary = - amplitude * np.sin(freq * x1 + offset_phase) + offset_sin
        #
        # # Label points above or below the boundary
        # if interface == 'jax':
        #     y = jnp.where(x2 > boundary, 1, 0)
        # else:
        #     y = np.where(x2 > boundary, 1, 0)
        #
        # return points, y
