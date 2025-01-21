from typing import Callable
from jax import numpy as jnp


def gaussian(amplitude, sigma, duration, *args, **kwargs) -> Callable:
    return lambda t: amplitude * jnp.exp(t - duration / 2) / sigma ** 2


def constant(amplitude, *args, **kwargs) -> Callable:
    return lambda t: amplitude


def sin(amplitude, freq, phase, *args, **kwargs) -> Callable:
    return lambda t: amplitude * jnp.sin(freq * t + phase)
