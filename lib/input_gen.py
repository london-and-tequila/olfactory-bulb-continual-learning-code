import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial

from .base import Xs_Generator

# class Uniform_Correlated(Xs_Generator):
#     """generate mulitple pairs of patterns following uniform distribution with certain correlation. Not sure if this is mathematically sound."""
#     def __init__(self, nE: int):
#         super().__init__()
#         self.nE = nE # input dimension

#     def _generate(self, rng: jax.Array, nPair: int, pairCorrelation: float = 0.9):
#         mean = jnp.array([0., 0.])
#         cov = jnp.array([[1., pairCorrelation],
#                          [pairCorrelation, 1.]])
#         xs = jax.random.multivariate_normal(rng, mean, cov, (nPair, self.nE)) # (nPair, self.nE, 2)
#         xs = jax.scipy.stats.norm.cdf(xs) # map to [0, 1] interval # not sure if the correlation is still preserved after this transformation
#         x1s = xs[..., 0] # (nPair, self.nE)
#         x2s = xs[..., 1] # (nPair, self.nE)
#         x1_lens = jnp.linalg.norm(x1s, axis=1)
#         x2_lens = jnp.linalg.norm(x2s, axis=1)
#         avg_len = (jnp.mean(x1_lens) + jnp.mean(x2_lens)) / 2.0
#         x1s = x1s / x1_lens[:, None] * avg_len
#         x2s = x2s / x2_lens[:, None] * avg_len
#         x1s = jnp.clip(x1s, 0.0, 1.0)
#         x2s = jnp.clip(x2s, 0.0, 1.0)
#         return x1s, x2s # each of shape (nPair, nE)
    
class Uniform_Correlated(Xs_Generator):
    """generate mulitple pairs of patterns following uniform distribution with certain correlation. Not sure if this is mathematically sound."""
    def __init__(self, nE: int):
        super().__init__()
        self.nE = nE # input dimension

    def _generate(self, rng: jax.Array, nPair: int, pairCorrelation: float = 0.9):
        rng1, rng2, rng = jax.random.split(rng, 3)
        z1 = jax.random.normal(rng1, (nPair, self.nE))
        epsilon = jax.random.normal(rng2, (nPair, self.nE))
        r = 2 * jnp.sin(jnp.pi * pairCorrelation / 6) 
        z2 = r * z1 + jnp.sqrt(1 - r ** 2) * epsilon
        x1s = jax.scipy.stats.norm.cdf(z1) # map to [0, 1] interval
        x2s = jax.scipy.stats.norm.cdf(z2) # map to [0, 1] interval
        return x1s, x2s # each of shape (nPair, nE)