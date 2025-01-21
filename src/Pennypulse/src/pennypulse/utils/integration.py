import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

def integrate_ranges(func, ranges):
    """
    Perform numerical integration of a given function over specified ranges using JAX.

    Parameters:
    func (callable): A function with signature func(t: float) -> float.
    ranges (jnp.ndarray): A 1D array of floats defining the integration boundaries [t0, t1, t2, ...].

    Returns:
    jnp.ndarray: An array of integration results for each range [ti, ti+1].
    """
    if len(ranges) < 2:
        raise ValueError("The ranges array must have at least two elements to define integration intervals.")

    def integrate_segment(t0, t1):
        # Generate fine-grained points between t0 and t1 for numerical integration
        t = jnp.linspace(t0, t1, 100)
        y = func(t)
        return trapezoid(y, t)

    results = jax.vmap(integrate_segment)(ranges[:-1], ranges[1:])
    return results