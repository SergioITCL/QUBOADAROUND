from abc import ABCMeta, abstractmethod
from copy import deepcopy

from math import floor, log10
from typing import Callable, Tuple
from itcl_quantization import Quantization
from random import random

from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase

_M = 1  # Scale Tweak Multiplier


class AbstractParamOptimizer(metaclass=ABCMeta):
    """This Optimizer updates the scale and zero point of the state attribute by reference.

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.

    Returns:
        _type_: _description_
    """

    state: list[NodeTensorBase]

    @abstractmethod
    def optimize(self) -> Tuple[list[NodeTensorBase], float]:
        """Optimization Method

        Returns:
            Tuple[List[np.ndarray], float]: Returns the optimized tensors with updated scales and zero points and the final loss/cost
        """
        pass

    def set_cost_fn(self, fn: Callable[[], float]) -> "AbstractParamOptimizer":
        """Updates the cost function, this function receives the updated rounding policies.

        Args:
            fn (Callable[[List[np.ndarray]], float]): The new cost function


        Returns:
            IRoundOptimizer: Self class
        """
        self._cost_fn: Callable[[list[NodeTensorBase]], float] = lambda x: fn()
        return self

    def set_initial_neigh(
        self, tensors: list[NodeTensorBase]
    ) -> "AbstractParamOptimizer":
        """Initializes the rounding policy neighborhood to optimize.

        Args:
            neigh (List[np.ndarray]): A list of binary numpy ndarrays. The arrays can have different shapes.

        Returns:
            IRoundOptimizer: Self class
        """

        self.state = tensors
        self._og_deepcopy = deepcopy(tensors)
        self._tensors = tensors
        return self

    def _eq_scale(self, scale: float) -> float:
        """Equalize Scale: Slightly changes the scale to find a better one.

        Args:
            scale (float): A float number smaller than 1

        Returns:
            float: The tweaked/equalized scale
        """
        num_of_zeroes = -floor(log10(scale)) + 1

        rnd = (random() / pow(10, num_of_zeroes)) * (-_M if random() > 0.5 else +_M)
        return scale + rnd

    def _eq_zp(self, zp: int, quant: Quantization) -> int:
        """Equalize Zero Point: Randomly increases or decreases the zero point
        by 1.

        Args:
            zp (int): Zero Point to Equalize
            quant (Quantization): Tensor's Quantization Class to avoid overflow

        Returns:
            int: _description_
        """
        if zp == quant.min_value():
            return zp + 1
        elif zp == quant.max_value():
            return zp - 1
        else:
            return zp + (1 if random() > 0.5 else -1)

    def copy(
        self, to: list[NodeTensorBase], from_: list[NodeTensorBase], requantize: bool
    ):
        """Copy the scale and ZP from one tensor to another

        Args:
            to (list[NodeTensorBase]): Tensors whose values are to be overriden
            from_ (list[NodeTensorBase]): Tensors to be copied.
            requantize (bool): If after updating, the tensor should be requantized.
        """
        for t, f in zip(to, from_):
            t.update_quant_parameters(f.scale, f.zero_point, requantize=requantize)

    def move(self):
        """
        Updates the state by randomly tweaking the scale and zero_point of each tensor.
        """
        for tensor in self.state:
            scale, zp = tensor.scale, tensor.zero_point

            # if randint(0, 1):
            #     scale = self._eq_scale(scale)
            # else:
            #     zp = self._eq_zp(zp, tensor.quant)

            zp = self._eq_zp(zp, tensor.quant)
            scale = self._eq_scale(scale)
            tensor.update_quant_parameters(scale, zp)
