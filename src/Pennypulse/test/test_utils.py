from jax import numpy as jnp
import unittest
from src.Pennypulse.src.pennypulse.utils.integration import integrate_ranges

class TestIntegration(unittest.TestCase):

    def test_integrate(self):
        n_points = 25
        t = jnp.linspace(0, 10, n_points)
        y = lambda t: 16 * t**2 - jnp.cos(3 * t)

        result = integrate_ranges(y, t)

        self.assertEqual(len(result), n_points - 1)
