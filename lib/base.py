from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import jax

Hyper = TypeVar('Hyper')
HyperDyn = TypeVar('HyperDynamic') # Hyperparameters that are fixed within one simulation, but can change between simulations. They are dynamic, and changing them won't trigger jax.jit recompilation.
State = TypeVar('State')

class Network(ABC, Generic[Hyper, State, HyperDyn]):
    def __init__(self, hyperparameters: Hyper):
        self.hyper = hyperparameters
        self.forward = self._forward
        self.update = self._update

    @abstractmethod
    def init_state(self, rng: jax.Array, hyperD: HyperDyn, *args, **kwargs) -> State:
        pass

    @abstractmethod
    def _forward(self, hyperD: HyperDyn, state: State, x, *args, **kwargs):
        """Forward pass of the network."""
        pass

    @abstractmethod
    def _update(self, hyperD: HyperDyn, state: State, x1: jax.Array, x2: jax.Array, *args, **kwargs) -> State:
        """Update the weights and other parameters of the network."""
        pass
        

class Xs_Generator(ABC):
    def __init__(self):
        self.generate = self._generate

    @abstractmethod
    def _generate(self, rng: jax.Array, nPair: int, *args, **kwargs):
        pass