from itertools import islice
from typing import Callable

from itcl_inference_engine.network.sequential import Network as NetworkIE

from itcl_quantizer.equalizers.param_equalizer.abstract_param_optimizer import (
    AbstractParamOptimizer,
)
from itcl_quantizer.equalizers.eq_bundler import NodeBundler
from itcl_quantizer.util.network import Network


class ParamEqualizerNet:
    """Parameter Equalizer Orchestrator
    Equalizes the Scales and Zero Points of an entire network to improve the network overall loss.
    This class updates the scales and zero points of the network nodes by reference.
    """

    def __init__(
        self,
        net: Network,
        loss_fn: Callable[[NetworkIE], float],
        optimizer_factory: Callable[[], AbstractParamOptimizer],
    ):
        """
        Param Equalizer Constructor


        Args:
            net (Network): Full Network (T)
            loss_fn (Callable[[NetworkIE], float]): _description_
            optimizer_factory (Callable[[], AbstractParamOptimizer]): _description_
        """
        self._loss_fn = loss_fn
        self._net = net
        self._layer_results = net.as_quant_results()
        self._optimizer_factory = optimizer_factory

    def equalize(self):
        """
        Main Function. Equalizes each layer individually.
        """

        results = self._layer_results

        for result in reversed(results):

            result.layer.param_equalizer(
                self._optimizer_factory,
                result,
                lambda: self._loss_fn(self._net.as_sequential_network()),
            )


class ParamEqualizerFullNet:
    def __init__(
        self,
        net: Network,
        loss_fn: Callable[[NetworkIE], float],
        optimizer_factory: Callable[[], AbstractParamOptimizer],
    ):
        self._net = net
        self._loss_fn = loss_fn
        self._layer_results = net.as_quant_results()
        self._optimizer_factory = optimizer_factory

    def equalize(self):
        bundler = NodeBundler()
        for result in self._layer_results:
            result.layer.param_equalizer_bundle(bundler, result)

        optimizer = self._optimizer_factory()
        optimizer.set_cost_fn(lambda: self._loss_fn(self._net.as_sequential_network()))
        optimizer.set_initial_neigh(list(bundler.nodes))
        optimizer.optimize()
