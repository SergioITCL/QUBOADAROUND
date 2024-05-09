from copy import deepcopy
from functools import reduce
import sys
from typing import Callable, List, Tuple
import numpy as np
from simanneal import Annealer
from random import randrange
from itcl_quantizer.equalizers.adaround.iround_optimizer import IRoundOptimizer


class RoundingAnnealer(Annealer, IRoundOptimizer):
    """Rounding Annealer"""

    def __init__(
        self,
        t_min: float = 0.1,
        t_max: float = 25000,
        updates: int = 100,
        steps: int = 5000,
        max_retries: int = 100,
    ):
        """

        Args:
            t_min (float, optional): Minimum Temperature. Defaults to 0.1.
            t_max (int, optional): Maximum Temperature. Defaults to 25000.
            updates (int, optional): Number of updates to show the user. Defaults to 100.
            steps (int, optional): Number of iterations. Defaults to 5000.
        """
        self.state: List[np.ndarray]
        # self.__initial_state = initial_w
        # self.energy_fn = energy_fn
        self.updates = updates
        self.Tmax = t_max
        self.Tmin = t_min
        self.steps = steps
        self._past_permutations: set[tuple[int, ...]] = set()
        self._max_retires = max_retries

    def set_cost_fn(self, fn: Callable[[List[np.ndarray]], float]):
        """Sets the cost function

        Args:
            fn (Callable[[List[np.ndarray]], float]): _description_
        """
        self.energy_fn = fn
        return self

    def set_initial_neigh(self, neigh: List[np.ndarray]):
        """Sets the initial binary neightborhood to optimize

        Args:
            neigh: List[np.ndarray]: A list of binary arrays that will be bit shifted
        """
        self.__initial_state = deepcopy(neigh)
        self.state = deepcopy(neigh)
        max_permutations = 0

        for n in neigh:
            max_permutations += reduce(lambda x, y: x * y, n.shape, 0)
        self._max_permutations = max_permutations
        return self

    def move(self):
        """
        Function to be called in each iteration of SA.
        """
        new_state = []

        while True:
            new_permutation = (*[randrange(start=0, stop=len(t)) for t in self.state],)
            if not new_permutation in self._past_permutations:
                break

            if len(self._past_permutations) >= self._max_permutations:
                return

        for tensor, idx in zip(self.state, new_permutation):

            shape = tensor.shape
            flatten = tensor.flatten()

            # idx = np.random.choice(
            #    np.arange(0, max(len(flatten) - 1, 1)), replace=False, size=(1)
            # )
            flatten[idx] = flatten[idx] ^ 1  # ^ is an XOR operation (bit shift)
            tensor = np.reshape(flatten, shape)
            new_state.append(tensor)
        self.state = new_state

    def _to_minima(self, initial_neigh: List[np.ndarray], initial_energy: float):
        """Function that tries to find local minima of the initial neighborhood

        Args:
            initial_neigh (List[np.ndarray]): _description_
            initial_energy (float): _description_

        Returns:
            _type_: _description_
        """
        best_energy = initial_energy
        best_neigh = initial_neigh
        stuck = 0  # Times the algorithm has been stuck within the same best_neigh

        while True:
            self.move()

            if (energy := self.energy()) < best_energy:
                best_neigh = list(self.state)
                best_energy = energy
                print(f"Improved after {stuck} times with {energy} energy")
                self._past_permutations = set()
                stuck = 0
            else:
                self.state = list(best_neigh)
                stuck += 1
            if stuck > self._max_retires:
                break
        return best_neigh, best_energy

    def anneal(self) -> Tuple[List[np.ndarray], float]:
        """simmaneal main function.
        Anneals the class and reaches a local minima by calling to_minima

        This functions should be private.

        Returns:
            Tuple[List[np.ndarray], float]: _description_
        """
        neigh, energy = super().anneal()
        return self._to_minima(neigh, energy or sys.maxsize)

    def energy(self) -> float:
        """Function that calculates the energy or loss

        Returns:
            float: loss
        """

        energy: float = self.energy_fn(self.state)
        return energy

    def optimize(self) -> Tuple[List[np.ndarray], float]:
        """Base IOptimizer Optimization class

        Returns:
            Tuple[List[np.ndarray], float]: A tuple with the improved neighborhood
            and the final cost
        """
        return self.anneal()
