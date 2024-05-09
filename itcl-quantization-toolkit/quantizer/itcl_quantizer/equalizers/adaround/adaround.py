from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Sequence

import numpy as np
from itcl_inference_engine.network.sequential import Network as NetworkIE

from itcl_quantizer.equalizers.adaround.iround_optimizer import IRoundOptimizer
from itcl_quantizer.equalizers.adaround.irounding_policy import IRoundingPolicy
from itcl_quantizer.util.network import Network

if TYPE_CHECKING:
    # circular import
    from itcl_quantizer.tensor_extractor.abstract_layer import (
        AbstractLayer,
        QuantizationResult,
    )


class AdaRound:
    """
    Node Tensor AdaRound
    """

    def __init__(
        self,
        optimizer: IRoundOptimizer,
        cost_fn: Callable[[], float],
        layer: AbstractLayer,
        qresults: QuantizationResult,
        float_input: np.ndarray | None,
    ) -> None:
        self._out_cost_fn = cost_fn
        self._optimizer = optimizer
        self._optimizer.set_cost_fn(self._cost_fn)
        self._optimizer.set_layer(layer)
        self._optimizer.set_input_data(float_input)
        if qresults is not None:
            self._optimizer.set_quant_results(qresults)
        if float_input is not None:
            self._optimizer.set_input_data(float_input)

    def _initialize_neighborhood(self, tensors: Sequence[IRoundingPolicy]):
        """
        Initializes the neigh with a list of one and zeroes of the current rounding policy.
        """
        return [t.rounding_policy for t in tensors]

    def _cost_fn(self, neigh: List[np.ndarray]) -> float:
        """
        The function `_cost_fn` is called by the optimizer to evaluate the cost function

        Updates the rounding policy by reference, so it only calls the base cost function.

        :param neigh: List[np.ndarray]: New Rounding Neighborhood
        :return: The cost function is being returned.
        """

        self._update_weights(self._operators, neigh)
        return self._out_cost_fn()

    def _update_weights(
        self, tensors: Sequence[IRoundingPolicy], neighborhoods: List[np.ndarray]
    ):
        """Updates each tensor rounding policy with the neighborhood by reference

        Args:
            tensors (List[IRoundingPolicy]): _description_
            neighborhoods (List[np.ndarray]): _description_

        Returns:
            the same input list by convenience
        """
        for tensor, neigh in zip(tensors, neighborhoods):
            tensor.rounding_policy = neigh
        return tensors

    def round(self, operators: Sequence[IRoundingPolicy]) -> Sequence[IRoundingPolicy]:
        """
        Runs the rounding policy optimization.
        Args:
            operators (List[IRoundingPolicy]): NodeTensors to optimize

        Returns:
            List[IRoundingPolicy]: The exact same input list, as the rounding_policy is updated by reference
        """
        self._operators = operators
        neigh = self._initialize_neighborhood(operators)
        self._optimizer.set_initial_neigh(neigh)
        best_neigh, cost = self._optimizer.optimize()
        return self._update_weights(operators, best_neigh)


class AdaroundNet:
    """Network Aware Adaround

    Optimizes each tensor rounding policy by taking into account the network loss.

    """

    def __init__(
        self,
        net: Network,
        loss_fn: Callable[[NetworkIE], float],
        optimizer_factory: Callable[[], IRoundOptimizer],
    ) -> None:
        """

        Args:
            net (Network): A Quantized Network
            loss_fn (Callable[[NetworkIE], float]): Loss Functions that infers a new network a returns a loss value to minimize
            optimizer_factory (Callable[[], IRoundOptimizer]): Functions that creates a new IRoundOptimizer per layer.
        """
        self._loss_fn = loss_fn
        self._net = net
        self._layer_results = net.as_quant_results()
        self._optimizer_factory = optimizer_factory

    def round(
        self,
    ):
        """
        AdaRound Function, updates the rounding policy of each layer of the net:Network by reference by taking into account
        the loss_fn.
        """

        for result in self._layer_results:
            layer = result.layer
            layer.adaround(
                self._optimizer_factory,
                result,
                lambda: self._loss_fn(self._net.as_sequential_network()),
            )
