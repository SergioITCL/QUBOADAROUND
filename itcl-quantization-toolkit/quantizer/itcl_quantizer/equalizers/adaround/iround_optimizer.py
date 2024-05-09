from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np

if TYPE_CHECKING:
    # circular import
    from itcl_quantizer.tensor_extractor.abstract_layer import (
        AbstractLayer,
        QuantizationResult,
    )


class IRoundOptimizer(metaclass=ABCMeta):

    """Interface that declares all the methods a Rounding Optimizer should include."""

    def set_cost_fn(self, fn: Callable[[List[np.ndarray]], float]) -> "IRoundOptimizer":
        """Updates the cost function, this function receives the updated rounding policies.

        Args:
            fn (Callable[[List[np.ndarray]], float]): The new cost function


        Returns:
            IRoundOptimizer: Self class
        """
        return self

    def set_initial_neigh(self, neigh: List[np.ndarray]) -> "IRoundOptimizer":
        """Initializes the rounding policy neighborhood to optimize.

        Args:
            neigh (List[np.ndarray]): A list of binary numpy ndarrays.
            The arrays can have different shapes.

        Returns:
            IRoundOptimizer: Self class
        """
        return self

    @abstractmethod
    def optimize(self) -> Tuple[List[np.ndarray], float]:
        """Optimization Method

        Returns:
            Tuple[List[np.ndarray], float]: Returns the optimized rounding policy and
             the final loss/cost
        """

    def set_layer(self, layer: AbstractLayer) -> "IRoundOptimizer":
        """Sets the layer to optimize

        Args:
            layer (AbstractLayer): The layer to optimize

        Returns:
            IRoundOptimizer: Self class
        """
        return self

    def set_quant_results(self, results: QuantizationResult) -> "IRoundOptimizer":
        """Sets the quantization results

        Args:
            results (QuantizationResult): Quantization results

        Returns:
            IRoundOptimizer: Self class
        """
        return self

    def set_input_data(self, data: np.ndarray) -> IRoundOptimizer:
        """Sets the input data to optimize

        Args:
            data (np.ndarray): The input data

        Returns:
            IRoundOptimizer: Self class
        """
        return self
