import math
from copy import deepcopy
from typing import Callable, List, Mapping, Tuple, Union, cast

import numpy as np
from itcl_quantization import LayerIds, Quantization
from tensorflow import keras

from itcl_quantizer.equalizers.adaround.adaround import AdaRound
from itcl_quantizer.equalizers.adaround.iround_optimizer import IRoundOptimizer
from itcl_quantizer.equalizers.adaround.irounding_policy import IRoundingPolicy
from itcl_quantizer.equalizers.adaround.qubo import QUBOAnnealer
from itcl_quantizer.equalizers.eq_bundler import NodeBundler
from itcl_quantizer.equalizers.param_equalizer.abstract_param_optimizer import (
    AbstractParamOptimizer,
)
from itcl_quantizer.optimizers import MinMaxOptimizer
from itcl_quantizer.optimizers.IOptimizer import IOptimizer
from itcl_quantizer.quantizer.distributions.distribution import Distribution
from itcl_quantizer.tensor_extractor.abstract_layer import (
    AbstractLayer,
    QuantizationResult,
)
from itcl_quantizer.tensor_extractor.keras.layers.activations import (
    quantize_activation_node,
)
from itcl_quantizer.tensor_extractor.operator import Operator
from itcl_quantizer.tensor_extractor.tensor import (
    NodeTensorBase as Node,
)
from itcl_quantizer.tensor_extractor.tensor import (
    NodeTensorTensor,
)

FUSED_RELU_TRIM: Callable[[Distribution], Distribution] = lambda x: x[0:]
FUSED_RELU6_TRIM: Callable[[Distribution], Distribution] = lambda x: x[0:6]


def fused_relu(activation: str):
    """
    If the activation is RELU, return FUSED_RELU_TRIM, else if the activation is RELU6, return
    FUSED_RELU6_TRIM, else return None.

    Args:
      activation (str): The activation function to use.

    Returns:
      the value of the variable FUSED_RELU_TRIM or FUSED_RELU6_TRIM.
    """
    if activation.upper() == "RELU":
        return FUSED_RELU_TRIM
    elif activation.upper() == "RELU6":
        return FUSED_RELU6_TRIM
    else:
        return None


class KerasDense(AbstractLayer):
    def __init__(
        self,
        layer,
        kernel_optimizer: IOptimizer = MinMaxOptimizer(),
        activation_optimizer: IOptimizer = MinMaxOptimizer(),
        bias_add_optimizer: IOptimizer = MinMaxOptimizer(),
        adaround_optimizer: None | IRoundOptimizer = None,
        kernel_dtype: str = "int8",
        bias_dtype: str = "int32",
        activation_dtype: str = "int8",
        bias_add_dtype: str = "int8",
        kernel_symmetric: bool = True,
    ):
        """
        The function takes in a keras layer, and then creates a new layer with the same weights and
        activation function.

        Args:
          layer: The Keras layer to be quantized.
          kernel_optimizer (IOptimizer): IOptimizer = MinMaxOptimizer()
          activation_optimizer (IOptimizer): IOptimizer = MinMaxOptimizer()
          bias_add_optimizer (IOptimizer): IOptimizer = MinMaxOptimizer()
          kernel_dtype (str): str = "int8". Defaults to int8
          bias_dtype (str): str = "int32". Defaults to int32
          activation_dtype (str): str = "int8". Defaults to int8
          bias_add_dtype (str): str = "int8". Defaults to int8
        """

        # Current keras Layer (Dense layer is expected)
        self.__layer = layer
        self.__layer_name = layer.name
        assert isinstance(layer, keras.layers.Dense)

        # Weight array (Kernel + Bias)
        weights = layer.get_weights()

        self.__kernel = np.transpose(weights[0])
        bias = (
            weights[1] if len(weights) > 1 else [0] * len(self.__kernel)
        )  # if there is no bias, create one

        self.__bias = np.array(bias)
        # Dense activation function
        self.__activation_fn = layer.activation
        self.__activation_name = str(self.__activation_fn.__name__)

        # Current Quantizer
        self.__q_kernel = Quantization(kernel_dtype)
        self.__q_activation = Quantization(activation_dtype)
        self.__q_bias_add = Quantization(bias_add_dtype)
        self.__q_bias = Quantization(bias_dtype)
        self.__adaround_optimizer = adaround_optimizer

        self.__kernel_trim = kernel_optimizer.trim
        self.__activation_trim = activation_optimizer.trim
        self.__bias_add_trim = bias_add_optimizer.trim
        self.__bias_add_trim = (
            fused_relu(self.__activation_name) or self.__bias_add_trim
        )
        self.__sym_kernel = kernel_symmetric
        self._last_float_input: np.ndarray | None = None

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns the weights of the layer.

        Returns:
          kernel and bias
        """
        return self.__kernel, self.__bias

    def __quantize_activation(
        self,
        input_data: np.ndarray,
        input_node: NodeTensorTensor,
    ) -> Tuple[Union[Operator, None], np.ndarray]:
        """
        It quantizes the activation function and generates a Look Up Table
        if possible

        Args:
          input_data (np.ndarray): the input data to the activation function
          input_node (Node): Previous / BiasAdd node
        """

        return quantize_activation_node(
            self.__activation_fn,
            self.__activation_name,
            input_data,
            self.__q_activation,
            input_node,
            self.__activation_trim,
        )

    def __quantize_kernel_from_params(
        self, scale: float, zero_p: int, clip: Tuple[float, float] | None = None
    ) -> NodeTensorTensor:
        kernel = self.__kernel
        if clip is not None:
            kernel = np.clip(kernel, clip[0], clip[1])

        return Node(
            scale,
            zero_p,
            f"{self.__layer_name}/Kernel{'Sym' if self.__sym_kernel else ''}",
            self.__q_kernel,
        ).with_tensor(kernel)

    def __quantize_kernel(self) -> NodeTensorTensor:
        """
        The function takes the kernel tensor, trims it, quantizes it, and returns the quantized kernel
        tensor.

        Returns:
          The quantized kernel
        """
        # Create and Trim the kernel distribution
        kernel_dist = Distribution(self.__kernel)
        kernel_dist = self.__kernel_trim(kernel_dist)

        # Get the quantized kernel
        w_s, w_zp = kernel_dist.quantize(
            self.__q_kernel, force_zp=0, symmetric=self.__sym_kernel
        )
        min_max = kernel_dist.get_min(), kernel_dist.get_max()
        return self.__quantize_kernel_from_params(w_s, w_zp, clip=min_max)

    def __quantize_bias_from_params(
        self, scale: float, zero_p: int
    ) -> NodeTensorTensor:
        """
        This function takes in a scale and zero point and returns a quantized bias tensor

        Args:
          scale (float): The scale of the quantized bias.
          zero_p (int): the zero point of the quantized tensor

        Returns:
          A dictionary with the following keys:
        """

        return Node(
            scale, zero_p, f"{self.__layer_name}/Bias", self.__q_bias
        ).with_tensor(self.__bias)

    def __quantize_bias(
        self, input_scale: float, kernel_scale: float
    ) -> NodeTensorTensor:
        """
        The function takes the input scale and kernel scale as input and returns the bias scale and zero
        point as output

        Args:
          input_scale (float): The scale of the input tensor.
          kernel_scale (float): The scale of the kernel tensor.

        Returns:
          A dictionary with the following keys:
        """
        bias_s = input_scale * kernel_scale
        bias_zp = 0  # Constraint

        return self.__quantize_bias_from_params(bias_s, bias_zp)

    def __quantize_bias_add_from_params(
        self, float_input: np.ndarray, scale: float, zp: int
    ) -> NodeTensorTensor:
        return (
            Node(
                scale,
                zp,
                f"{self.__layer_name}/BiasAdd",
                self.__q_bias_add,
            )
            .with_tensor(float_input)
            .exclude_batch_dimension()
            .exclude_tensor()
        )

    def __quantize_bias_add(
        self, float_input: np.ndarray
    ) -> Tuple[NodeTensorTensor, np.ndarray]:
        """
        It takes a float input, runs it through the layer, gets the bias add, trims the distribution,
        quantizes the bias add, and returns the quantized bias add and the float output.

        Args:
          float_input (np.ndarray): The input to the layer.

        Returns:
          The return value is a tuple of two elements. The first element is a dictionary that contains the
        quantization parameters for the bias add operation. The second element is the output of the bias add
        operation.
        """
        layer = deepcopy(self.__layer)
        layer.activation = None
        float_output = layer(float_input)
        # Get and trim the distribution:
        out_dist = Distribution(float_output)

        out_dist = self.__bias_add_trim(out_dist)

        out_s, out_zp = out_dist.quantize(self.__q_bias_add)
        return (
            self.__quantize_bias_add_from_params(float_input, out_s, out_zp),
            float_output,
        )

    def __round_cost_fn(
        self,
        fp_input: np.ndarray,
        tensors: List[NodeTensorTensor],
        expected_output: np.ndarray,
    ) -> float:
        layer_dq = deepcopy(self.__layer)
        deq_weights = []

        for tensor in tensors:
            deq_weights.append(tensor.dequantized.T)

        layer_dq.activation = None
        layer_dq.set_weights(deq_weights)

        dq_output = layer_dq(fp_input)
        diff = expected_output - dq_output

        cost = np.linalg.norm(diff)
        return float(cost)

    def quantize(self, input_result: QuantizationResult) -> QuantizationResult:
        """
        The function takes in a QuantizationResult object, which contains the input data, the output node,
        and the operators. It then quantizes the kernel, bias, and bias add, and then quantizes the
        activation. It then returns a QuantizationResult object, which contains the input data, the output
        node, and the operators (QFullyCon = FullyCon(matmul+biasAdd) + Activation).

        Args:
          input_result (QuantizationResult): QuantizationResult

        Returns:
          The quantization result is being returned.
        """

        float_input = input_result.input_data
        self._last_float_input = float_input
        input_node = input_result.out_node

        if input_node is None:
            raise ValueError(
                "Input quantization is not defined in a layer that should contain an input"
            )

        #
        # Kernel Node
        #
        kernel_node = self.__quantize_kernel()

        #
        # Bias Node
        #
        input_node_s, kernel_node_s = input_node.scale, kernel_node.scale

        if input_node_s is None or kernel_node_s is None:
            raise ValueError(
                "Input and Kernel scales are not defined in a layer that should contain an input"
            )

        bias_node = self.__quantize_bias(input_node_s, kernel_node_s)

        update_bias_scale_fn = lambda _: bias_node.update_quant_parameters(
            kernel_node.scale * input_node.scale, bias_node.zero_point
        )

        # On kernel/input scale update: Update the bias scale and requantize the BIAS
        kernel_node.on_quant_param_update.append(update_bias_scale_fn)
        input_node.on_quant_param_update.append(update_bias_scale_fn)

        #
        # Bias Add Node
        #
        bias_add_node, float_output = self.__quantize_bias_add(float_input)
        fc_layer = deepcopy(self.__layer)
        fc_layer.activation = None
        fully_connected = Operator(
            LayerIds.fully_conn,
            f"fully_connected {self.__layer_name}",
            [input_node, kernel_node, bias_node],
            [bias_add_node],
            fc_layer,
        )

        #
        # Layer Aware Adaround
        #
        if optimizer := self.__adaround_optimizer:
            adaroundable_tensors = [kernel_node, bias_node]
            round_cost_fn: Callable[[], float] = lambda: self.__round_cost_fn(
                float_input,
                adaroundable_tensors,
                float_output,
            )
            raise NotImplementedError("TODO: Implement Adaround for ")
            AdaRound(optimizer, round_cost_fn).round(adaroundable_tensors)

        #
        # Activation Function (Creates a new Operator)
        #
        activation_op, activation_result = self.__quantize_activation(
            float_output, bias_add_node
        )

        nullable_operators: list[Operator | None] = [fully_connected, activation_op]
        # remove Nones from operators but keeping the order
        operators: list[Operator] = [op for op in nullable_operators if op is not None]

        return QuantizationResult(
            input_data=activation_result,
            out_node=operators[-1].outputs[0],
            operators=operators,
            layer=self,
        )

    def adaround(
        self,
        optimizer_factory: Callable[[], IRoundOptimizer],
        results: QuantizationResult,
        cost_fn: Callable[[], float],
    ):
        if not results.operators:
            raise ValueError("Missing Fully Connected Tensor")

        fully_conn = results.operators[0]

        activation = results.operators[1] if len(results.operators) > 1 else None

        tensors = cast(list[IRoundingPolicy], fully_conn.inputs[1:])

        if activation:
            lut = activation.inputs[0]
            if lut.LUT:
                tensors.append(lut.LUT)

        optimizer = optimizer_factory()

        AdaRound(
            optimizer, cost_fn, self, results, float_input=self._last_float_input
        ).round(tensors)

    def _get_nodes_eq(self, results: QuantizationResult) -> List[Node]:
        if not results.operators:
            raise ValueError("Missing Fully Connected Tensor")

        fully_conn = results.operators[0]
        activation = results.operators[1] if len(results.operators) == 2 else None

        tensors_to_eq = [fully_conn.inputs[1], fully_conn.outputs[0]]

        if activation:
            tensors_to_eq.append(activation.outputs[0])

        return tensors_to_eq

    def param_equalizer(
        self,
        optimizer_factory: Callable[[], AbstractParamOptimizer],
        results: QuantizationResult,
        cost_fn: Callable[[], float],
    ):
        if not results.operators:
            raise ValueError("Missing Fully Connected operator")

        fully_conn = results.operators[0]
        activation = results.operators[1] if len(results.operators) == 2 else None

        nodes_to_eq = [fully_conn.inputs[1], fully_conn.outputs[0]]

        if activation:
            nodes_to_eq.append(activation.outputs[0])

        optimizer_factory().set_cost_fn(cost_fn).set_initial_neigh(
            nodes_to_eq
        ).optimize()

        if activation:
            optimizer_factory().set_cost_fn(cost_fn).set_initial_neigh(
                [activation.outputs[0]]
            ).optimize()

    def param_equalizer_bundle(self, bundler: NodeBundler, results: QuantizationResult):
        if not results.operators:
            raise ValueError("Missing Fully Connected operator")

        fully_conn = results.operators[0]

        activation = results.operators[1] if len(results.operators) == 2 else None

        # Kernel & Bias Add
        nodes_to_eq = [fully_conn.inputs[1], fully_conn.outputs[0]]

        if activation:
            nodes_to_eq.append(activation.outputs[0])

        bundler.add_nodes(*nodes_to_eq)

    def calc_collisions(
        self,
        results: QuantizationResult,
        collision_policy: Callable[[np.ndarray, np.ndarray], int],
    ) -> Mapping[str, int]:
        res = {}

        nodes = self._get_nodes_eq(results)

        for node in nodes:
            assert isinstance(node, NodeTensorTensor)

        kernel: NodeTensorTensor = nodes[0]  # type: ignore
        res[kernel.name] = collision_policy(kernel.fp_tensor, kernel.quantized)

        bias_add: NodeTensorTensor = nodes[1]  # type: ignore
        print("bias add c: ", collision_policy(bias_add.fp_tensor, bias_add.quantized))
        res[bias_add.name] = math.ceil(
            collision_policy(bias_add.fp_tensor, bias_add.quantized)
            / bias_add.fp_tensor.shape[0]
        )

        if len(nodes) == 3:
            activation: NodeTensorTensor = nodes[2]  # type: ignore
            res[activation.name] = math.ceil(
                collision_policy(activation.fp_tensor, activation.quantized)
                / activation.fp_tensor.shape[0]
            )

        return res
