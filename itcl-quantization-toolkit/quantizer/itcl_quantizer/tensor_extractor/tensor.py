from typing import Callable

import numpy as np
from itcl_quantization import Quantization
from itcl_quantization.json.specification import LUT as LUTJson
from itcl_quantization.json.specification import Node as NodeJson
from itcl_quantization.json.specification import Tensor as TensorJson
from itcl_quantization.quantization.lut import ReducedLUT
from typing_extensions import Self

from itcl_quantizer.equalizers.adaround.irounding_policy import (
    IRoundingPolicy,
    get_base_rounding,
)
from itcl_quantizer.interfaces.serializable import ISerializable


class NodeLUT(ISerializable, IRoundingPolicy):
    """Node Look Up Table

    Args:
        ISerializable (): Must Be Serializable

    Returns:
        _type_:
    """

    _reduced: None | ReducedLUT = None

    def __init__(
        self,
        quant: Quantization,
        fn: Callable[[np.ndarray], np.ndarray],
        input_s: float,
        input_zp: int,
        output_s: float,
        output_zp: int,
        reduced_depth: int = 3,
    ) -> None:
        """_summary_

        Args:
            quant (Quantization): Quantization parameters/class of the LUT
            fn (Callable[[np.ndarray], np.ndarray]): Function to transform into a LUT
            input_s (float): Previous Node Scale
            input_zp (int): Previous Node Zero Point
            output_s (float): Output Scale (Calculated from the distribution of the activated data)
            output_zp (int): Output ZP (...)
            reduced_depth (int, optional): Depth of the Reduced LUT. Defaults to 3.
        """
        self._input_s = input_s
        self._input_zp = input_zp
        self._output_s = output_s
        self._output_zp = output_zp
        self._reduced_depth = reduced_depth
        self._quant = quant
        self._fn = fn

        self._rounding_policy = get_base_rounding(
            quant.float_activation(fn, input_s, input_zp), output_s
        )
        self._create()

    def _create(self):
        """Create the LUT"""
        self._LUT = self._quant.create_LUT(
            self._fn,
            self._input_s,
            self._input_zp,
            self._output_s,
            self._output_zp,
            rounding_policy=self._rounding_policy,
        )
        self._offset = -self._quant.min_value()
        self._quant = self._quant

        # TODO: Reduced LUT
        #   if reduced_depth := self._reduced_depth:
        #    self._reduced = ReducedLUT(self._LUT, reduced_depth, 1)

    def _update_in_qparams(self, input_s: float, input_zp: int):
        """Update Input Quantization Parameters

        Args:
            input_s (float): New input scale
            input_zp (int): New input Zero Point
        """
        self._input_s = input_s
        self._input_zp = input_zp
        self._create()

    def _update_out_qparams(self, output_s: float, output_zp: int):
        """Update the out node quantiation parameters

        Args:
            output_s (float): Output Node Scale
            output_zp (int): Ouptut Node Zero Point
        """
        self._output_s = output_s
        self._output_zp = output_zp

    @property
    def offset(
        self,
    ) -> int:
        """Offset of the LUT to index

        expected usage:

        lut[quant_value_to_activate + lut.offset]

        Returns:
            int: The offset of the LUT
        """
        return self._offset

    def __call__(self, value: int) -> int:
        """Gets the activated value of the input value

        usage:
            quant_val = -1
            activated_val = lut(quant_val)

        Args:
            value (int): Value to activate

        Returns:
            int: Activated Value
        """
        return self._LUT[value + self.offset]

    def __getitem__(self, idx: int):
        """Activate the LUT

        Args:
            idx (int): _description_

        Returns:
            _type_: _description_
        """
        return self._LUT[idx]

    def __len__(self) -> int:
        """Len off the

        Returns:
            int: _description_
        """
        return len(self._LUT)

    @property
    def rounding_policy(self) -> np.ndarray:
        return self._rounding_policy

    @rounding_policy.setter
    def rounding_policy(self, policy: np.ndarray):
        self._rounding_policy = policy

    def as_json(self) -> LUTJson:
        """Serializes the LUT class as a serializable dictionary k:str, v: Any

        Returns:
            LUTJson: A TypedDict
        """
        reduced = self._reduced.serialize() if self._reduced is not None else None

        return {
            "LUT": self._LUT,
            "reduced_LUT": reduced,
            "offset": self._offset,
        }


class NodeTensorBase(ISerializable):
    """

    Args:
        ISerializable (_type_):

    """

    _LUT: NodeLUT | None = None
    """Look Up Table
    """
    _exclude_params: bool = False
    """Serializing Flag: Excludes the quantization parameters from the generated DICT
    """
    #
    # Hooks: Functions that are called on an event
    #
    on_requantize: list[Callable[[Self], None]]
    on_quant_param_update: list[Callable[[Self], None]]

    def __init__(self, scale: float, zero_p: int, name: str, quant: Quantization):
        """

        Args:
            scale (float): Node Scale
            zero_p (int): Node Zero Point
            name (str): Node Name
            quant (Quantization): Node Quantization Details
        """
        self._scale = self._og_scale = scale
        self._zero_p = self._og_zero_p = zero_p
        self._name = name
        self._quant = quant
        self.on_requantize = []
        self.on_quant_param_update = []

    @property
    def scale(self) -> float:
        """Scale Getter"""
        return self._scale

    def reset_scale(self):
        """Resets the scale to the original/constructor one."""
        self._scale = self._og_scale

    @property
    def name(self) -> str:
        """Name Getter

        Returns:
            str: Node Name
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Name Setter. Replaces the node name

        Args:
            name (str): New Name
        """
        self._name = name

    @property
    def quant(self) -> Quantization:
        """Quant getter"""
        return self._quant

    @property
    def zero_point(self) -> int:
        """Zero Point Getter

        Returns:
            int: Current Node Zero Point
        """
        return self._zero_p

    def reset_zp(self):
        """Resets the current zero point to the original one"""
        self._zero_p = self._og_zero_p

    def update_quant_parameters(
        self, scale: float, zero_point: int, requantize: bool = True
    ):
        """Function that updates the scale and zero point.

        By default, the node is requantized (Tensors, LUT, etc)

        Args:
            scale (float): New scale
            zero_point (int): New zero Point
            requantize (bool, optional): If requantize. Defaults to True.
        """
        self._scale = scale
        self._zero_p = zero_point
        if requantize:
            self._requantize()
        self._run_hook(self.on_quant_param_update)

    def with_tensor(self, tensor: np.ndarray) -> "NodeTensorTensor":
        """Creates a tensor version of the Node

        Args:
            tensor (np.ndarray): The node tesnor.

        Returns:
            NodeTensorTensor: A new instance of the class with the current parameters
        """
        return NodeTensorTensor(self, tensor)

    def with_lut(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        out_scale: float,
        out_zp: int,
        reduced_depth=3,
    ) -> Self:
        """Adds a LUT to the node

        Args:
            fn (Callable[[np.ndarray], np.ndarray]): Function to create a LUT with
            out_scale (float): Output Node Scale
            out_zp (int): Output Node Zero Point
            reduced_depth (int, optional): Reduced Depth. Defaults to 3, 0 to disable.

        Returns:
            Self: _description_
        """
        self._LUT = NodeLUT(
            self.quant,
            fn,
            self.scale,
            self.zero_point,
            out_scale,
            out_zp,
            reduced_depth,
        )
        return self

    @property
    def LUT(self) -> NodeLUT | None:
        """Gets the NodeLUT

        Returns:
            NodeLUT | None: None if the node does not contain a LUT.
        """
        return self._LUT

    def exclude_quant_params(self, exclude: bool = True) -> "NodeTensorBase":
        self._exclude_params = exclude
        return self

    def as_json(self) -> NodeJson:

        LUT = self.LUT.as_json() if self.LUT else None

        return {
            "scale": self.scale if not self._exclude_params else None,
            "zero_point": self.zero_point if not self._exclude_params else None,
            "tensor": None,
            "LUT": LUT,
        }

    def __getitem__(self, item: str):
        """
            Gets a value of the json. Deprecated method, use the attribute/as_json() instead.

        Args:
            item (str):

        Returns:
            _type_: _description_
        """
        return self.as_json().get(item)

    def _requantize(self):
        """Requantize the node"""
        if LUT := self.LUT:
            LUT._update_in_qparams(self.scale, self.zero_point)
        self._run_hook(self.on_requantize)

    def _run_hook(self, hooks: list[Callable[[Self], None]]):
        """Helper function to run a hook

        Args:
            hooks (list[Callable[[Self], None]]): List of hooks
        """
        for fn in hooks:
            fn(self)


class NodeTensorTensor(IRoundingPolicy, NodeTensorBase):
    """

    Args:
        NodeTensorBase

    Returns:
        _type_: _description_
    """

    # Serialization settings to include/exclude data in the json
    _exclude_batch_dimension = False
    _exclude_tensor = False

    def __init__(
        self,
        node: NodeTensorBase,
        fp_tensor: np.ndarray,
    ) -> None:
        """Constructor

        Args:
            node (NodeTensorBase): A NodeTensorBase
            fp_tensor (np.ndarray): A numpy array with the new tensor
        """
        self.__dict__ = node.__dict__.copy()
        self._fp_tensor = fp_tensor
        self._rounding_policy = self._og_rounding_policy = get_base_rounding(
            fp_tensor, self.scale
        )
        self._quantized = self.quant.quantize(fp_tensor, self.zero_point, self.scale)

    def exclude_batch_dimension(self, exclude: bool = True) -> Self:
        """If the batch dimension should not appear in the json

        Args:
            exclude (bool, optional): Should be excluded or not. Defaults to True.

        Returns:
            Self: The same object
        """
        self._exclude_batch_dimension = exclude
        return self

    def exclude_tensor(self, exclude: bool = True) -> Self:
        """
        If the tensor should be exclusded from the json

        Args:
            exclude (bool, optional):  Should be excluded or not. Defaults to True.

        Returns:
            Self: _description_
        """
        self._exclude_tensor = exclude
        return self

    def reset_rounding(self):
        """Reset the rounding policy to Round To Nearest"""
        self.rounding_policy = self._og_rounding_policy

    @property
    def fp_tensor(self) -> np.ndarray:
        """Getter for the float tensor to be quantized

        Returns:
            np.ndarray: The float tensor
        """
        return self._fp_tensor

    @property
    def rounding_policy(self) -> np.ndarray:
        """Get the current rounding policy

        Returns:
            np.ndarray: The Rounding Policy
        """
        return self._rounding_policy

    @rounding_policy.setter
    def rounding_policy(self, rounding: np.ndarray):
        """Rounding Policy Setter
        On the rounding policy change, the tensor is requantized with the new rounding policy.

        Args:
            rounding (np.ndarray): New rounding policy. Should match the tensor shape.

        Raises:
            ValueError: _description_
        """
        if rounding.shape != self._fp_tensor.shape:
            raise ValueError(
                f"New rounding policy has shape {rounding.shape}. {self._fp_tensor.shape} expected"
            )

        self._rounding_policy = rounding
        self._quantized = self.quant.quantize(
            self._fp_tensor, self.zero_point, self.scale, rounding_policy=rounding
        )

    @property
    def quantized(self) -> np.ndarray:
        """Gets the quantized float tensor with the current rounding policy, scale and ZP

        Returns:
            np.ndarray: The
        """
        return self._quantized

    @property
    def dequantized(self) -> np.ndarray:
        """Gets a float version of the tensor with the quantization error.
        This getter dequantizes the current quantized tensor.

        Returns:
            np.ndarray: A float tensor
        """
        return self.quantized * self.scale + self.zero_point

    def _requantize(self):
        """Requantization Function"""
        self._quantized = self.quant.quantize(
            self._fp_tensor,
            self.zero_point,
            self.scale,
            rounding_policy=self._rounding_policy,
        )
        super()._requantize()

    def _tensor_json(self) -> TensorJson | None:
        """Helper function that serializes the Tensor

        Returns:
            TensorJson | None: _description_
        """

        return (
            {
                "name": self.name,
                "dtype": self.quant.dtype_str(),
                "shape": list(self._fp_tensor.shape)[
                    1 if self._exclude_batch_dimension else None :
                ],  # Remove the batch dimension if necessary
                "tensor": None if self._exclude_tensor else self.quantized.tolist(),
            }
            if self._fp_tensor is not None
            else None
        )

    def as_json(self) -> NodeJson:
        """Serializable Method

        Returns:
            NodeJson: Creates a NodeJson.
        """
        super_json = super().as_json()
        super_json["tensor"] = self._tensor_json()
        return super_json
