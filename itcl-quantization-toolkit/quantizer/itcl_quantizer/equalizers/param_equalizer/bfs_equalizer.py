from copy import deepcopy
from itcl_quantizer.equalizers.param_equalizer.abstract_param_optimizer import (
    AbstractParamOptimizer,
)
from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase


class BFSEqualizer(AbstractParamOptimizer):
    """Best First Search

    This optimizer only moves to better neighborhoods.

    Args:
        AbstractParamOptimizer
    """

    def __init__(self, max_retries: int = 200):
        """_summary_

        Args:
            max_retries (int, optional): Max number of retries until the optimization is halted. Defaults to 200.
        """
        self._max_retries = max_retries

    def optimize(self) -> tuple[list[NodeTensorBase], float]:

        best_neigh = deepcopy(self._tensors)
        best_cost = self._cost_fn(best_neigh)

        current_retries = 0
        while True:
            self.move()
            cost = self._cost_fn(self.state)
            current_retries += 1
            if (cost := self._cost_fn(self.state)) < best_cost:
                best_cost = cost
                self.copy(best_neigh, self.state, requantize=False)
                print(f"BFS_EQ improved with {cost} after {current_retries} iterations")
                current_retries = 0
            else:
                self.copy(self.state, best_neigh, requantize=False)

            if current_retries > self._max_retries:
                break

        self.copy(self._tensors, best_neigh, requantize=True)

        return self._tensors, best_cost
