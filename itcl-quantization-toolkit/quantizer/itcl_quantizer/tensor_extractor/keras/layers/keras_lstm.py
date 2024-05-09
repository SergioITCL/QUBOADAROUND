from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import tensorflow as tf
from itcl_inference_engine.layers.itclq.lstm import LSTM as LSTMInference
from itcl_quantization import LayerIds, Quantization
from itcl_quantization.ids import LSTMInputs
from itcl_quantization.json.specification import Attribute

from itcl_quantizer.equalizers.adaround.adaround import AdaRound
from itcl_quantizer.equalizers.adaround.iround_optimizer import IRoundOptimizer
from itcl_quantizer.quantizer import Distribution
from itcl_quantizer.tensor_extractor import NodeTensorBase as Node
from itcl_quantizer.tensor_extractor import NodeTensorTensor, Operator
from itcl_quantizer.tensor_extractor.abstract_layer import (
    AbstractLayer,
    QuantizationResult,
)
from itcl_quantizer.util.const import ACTIVATION_NAMES
from itcl_quantizer.util.debug import is_debug_mode
from itcl_quantizer.util.typing import NPFP32

from .activations import quantize_activation_node

LSTM_GATES = ["input", "forget", "cell", "output"]


class KerasLSTM(AbstractLayer):
    """Keras LSTM Quantization Class"""

    def __init__(
        self,
        layer,
        kernel_dtype: str = "int8",
        recurrent_kernel_dtype: str = "int8",
        bias_dtype: str = "int32",
        bias_add_dtype: str = "int32",
        recurrent_add_dtype: str = "int8",
        activation_dtype: str = "int8",
        output_dtype: str = "int8",
        cell_state_dtype: str = "int8",
        hidden_state_dtype: str = "int8",
    ) -> None:
        self._layer = layer
        self._layer_name = layer.name
        self._K: NPFP32 = np.array(layer.get_weights()[0])  # kernel
        self._U: NPFP32 = np.array(layer.get_weights()[1])  # recurrent_kernel
        self._B: NPFP32 = np.array(layer.get_weights()[2])  # bias
        self._n_units: int = layer.units

        self._return_sequences: bool = layer.return_sequences

        self._q_kernel = Quantization(kernel_dtype)
        self._q_recurrent_kernel = Quantization(recurrent_kernel_dtype)
        self._q_hidden_state = Quantization(hidden_state_dtype)
        self._q_cell_state = Quantization(cell_state_dtype)
        self._q_bias = Quantization(bias_dtype)
        self._q_activation = Quantization(activation_dtype)
        self._q_bias_add = Quantization(bias_add_dtype)
        self._q_recurrent_add = Quantization(recurrent_add_dtype)
        self._q_output = Quantization(output_dtype)

    def _get_scale_zp(
        self, tensor: NPFP32, quantize: Quantization, symmetric: bool = False
    ) -> tuple[float, int]:
        """Base function that obtains the scale and zero point of a tensor.

        Args:
            tensor (NPFP32): A nummpy array
            quantize (Quantization): Quantization details of the object
            symmetric (bool, optional): If the tensor should be quantized symmetrically
                The zero point is 0. Defaults to False.

        Returns:
            tuple[float, int]: The scale and zero point of the quantized tensor
        """
        dist = Distribution(tensor)

        sym = 0 if symmetric else None

        scale, zp = dist.quantize(quantize, symmetric=symmetric, force_zp=sym)
        return scale, zp

    #
    # Static Tensor Quantization
    #
    def _quantize_kernel(
        self,
    ) -> list[NodeTensorTensor]:
        """Quantizes the kernel

        The kernel is split and quantized into 4 different parts, one for each gate

        Returns:
            list[NodeTensorTensor]: 4 nodes, one for each gate
        """

        nodes: list[NodeTensorTensor] = []
        for i, gate in enumerate(LSTM_GATES):
            tensor = self._K[:, i * self._n_units : (i + 1) * self._n_units]

            scale, zp = self._get_scale_zp(tensor, self._q_kernel)
            nodes.append(
                Node(
                    scale, zp, f"{self._layer_name}/Kernel/{gate}", self._q_kernel
                ).with_tensor(tensor)
            )

        return nodes

    def _quantize_recurrent_kernel(
        self,
    ) -> list[NodeTensorTensor]:
        """Quantizes the recurrent kernel

        The recurrent kernel is split and quantized into 4 different parts, one for each gate

        Returns:
            list[NodeTensorTensor]: 4 tensor nodes, one for each gate
        """

        nodes: list[NodeTensorTensor] = []

        for i, gate in enumerate(LSTM_GATES):
            tensor = self._U[:, i * self._n_units : (i + 1) * self._n_units]
            scale, zp = self._get_scale_zp(tensor, self._q_recurrent_kernel)
            nodes.append(
                Node(
                    scale,
                    zp,
                    f"{self._layer_name}/RecurrentKernel/{gate}",
                    self._q_recurrent_kernel,
                ).with_tensor(tensor)
            )

        return nodes

    def _quantize_bias(
        self, input_scale: float, kernel_nodes: list[NodeTensorTensor]
    ) -> list[NodeTensorTensor]:
        """Quantizes the bias symmetrically

        The bias scale is calculated as the product of the input scale and the kernel scale.

        As there are 4 kernel nodes, the bias will be quantized into 4 different parts, one for each gate.


        Args:
            input_scale (float): The scale of the Input Tensor
            kernel_nodes (list[NodeTensorTensor]): The 4 quantized kernel nodes

        Returns:
            list[NodeTensorTensor]: 4 bias tensor nodes, one for each gate.
        """
        nodes: list[NodeTensorTensor] = []

        for i, (gate, node) in enumerate(zip(LSTM_GATES, kernel_nodes)):
            tensor = self._B[i * self._n_units : (i + 1) * self._n_units]
            scale = input_scale * node.scale
            zp = 0
            nodes.append(
                Node(
                    scale, zp, f"{self._layer_name}/Bias/{gate}", self._q_bias
                ).with_tensor(tensor)
            )

        return nodes

    def _quantize_bias_add(
        self,
        bias_add: NPFP32,
        recurrent_nodes: list[NodeTensorTensor],
        hidden_scale: float,
    ) -> list[Node]:
        """Symmetric quantization of the intermediate tensor "Bias Add"

        This tensor is the result of adding the bias to the multiplication of W (kernel) with I (input)

        The bias add is quantized into 4 different parts, one for each gate.

        The scale of the bias add is calculated as the product between the scale
            of the Hidden State and the scale of the recurrent kernel.

        Args:
            bias_add (NPFP32): The bias add tensor
            recurrent_nodes (list[NodeTensorTensor]): The 4 quantized recurrent kernel nodes
            hidden_scale (float): Hidden State / Output vector Scale

        Returns:
            list[Node]: 4 bias add nodes, one for each gate.
        """

        nodes: list[Node] = []

        for i, gate, node in zip(range(len(LSTM_GATES)), LSTM_GATES, recurrent_nodes):
            tensor = bias_add[:, :, i * self._n_units : (i + 1) * self._n_units]
            scale = node.scale * hidden_scale
            zp = 0
            nodes.append(
                Node(scale, zp, f"{self._layer_name}/BiasAdd/{gate}", self._q_bias_add)
                .with_tensor(tensor)
                .exclude_batch_dimension()
                .exclude_tensor()
            )

        return nodes

    def _quantize_recurrent_add(self, recurrent_add: NPFP32) -> list[Node]:
        """Quantizes the intermediate tensor "Recurrent Add"

        This tensor is divided into 4 different parts, one for each gate.

        Args:
            recurrent_add (NPFP32): Intermediate tensor to quantize

        Returns:
            list[Node]: _description_
        """
        nodes: list[Node] = []

        for i, gate in enumerate(LSTM_GATES):
            tensor = recurrent_add[:, :, i * self._n_units : (i + 1) * self._n_units]

            scale, zp = self._get_scale_zp(tensor, self._q_recurrent_add)
            nodes.append(
                Node(
                    scale,
                    zp,
                    f"{self._layer_name}/RecurrentAdd/{gate}",
                    self._q_recurrent_add,
                )
                .with_tensor(tensor)
                .exclude_batch_dimension()
                .exclude_tensor()
            )
        return nodes

    def _quantize_cell_state(self, cell_state: NPFP32) -> Node:
        """Quantizes the Cell State

        Args:
            cell_state (NPFP32): All the cell states of the LSTM.

        Returns:
            Node: Quantized Cell State
        """
        scale, zp = self._get_scale_zp(cell_state, self._q_cell_state, symmetric=True)
        return (
            Node(scale, zp, f"{self._layer_name}/CellState", self._q_cell_state)
            .with_tensor(cell_state)
            .exclude_batch_dimension()
            .exclude_tensor()
        )

    def _quantize_hidden_state(self, hidden_state: NPFP32) -> Node:
        """Quantizes the Hidden State and Output Vector of the Network

        Args:
            hidden_state (NPFP32): Accumulated hidden state of the LSTM network

        Returns:
            Node: _description_
        """
        scale, zp = self._get_scale_zp(hidden_state, self._q_hidden_state)
        return (
            Node(scale, zp, f"{self._layer_name}/Output", self._q_hidden_state)
            .with_tensor(hidden_state)
            .exclude_batch_dimension()
            .exclude_tensor()
        )

    def _quantization_noise(
        self, result: QuantizationResult, fp_results: LSTMResult, fp_input: np.ndarray
    ) -> list[tuple[str, list[float]]]:
        """Adds quantization noise to the output of the LSTM layer

        Args:
            result (QuantizationResult): The quantization result of the LSTM layer

        Returns:
            QuantizationResult: The quantization result of the LSTM layer with quantization noise
        """

        if result.operators is None:
            raise ValueError("LSTM: No operators found in the quantization result")

        operator = result.operators[0]

        input_node = operator.inputs[0]

        input_q = input_node.quant

        int_input = input_q.quantize(fp_input, input_node.zero_point, input_node.scale)

        # get the intermediate quantized tensors
        lstm = LSTMInference.from_model(operator.as_json())
        lstm.infer(int_input)
        quant = lstm.intermediate_tensors

        floating_results = [
            *np.split(fp_results.bias_add, 4, axis=-1),
            *np.split(fp_results.recurrent_add, 4, axis=-1),
            fp_results.input_gate,
            fp_results.forget_gate,
            fp_results.cell_gate,
            fp_results.output_gate,
            fp_results.cell_state,
            fp_results.cell_state_activation,
            fp_results.hidden_state,
        ]

        quant_results = [
            *np.array(quant.bias_add).swapaxes(0, 1),
            *np.array(quant.recurrent_add).swapaxes(0, 1),
            quant.input_gate_act,
            quant.forget_gate_act,
            quant.cell_gate_act,
            quant.output_gate_act,
            quant.cell_state,
            quant.cell_state_act,
            quant.hidden_state,
        ]
        input_nodes = operator.inputs
        nodes = [
            input_nodes[LSTMInputs.BIAS_ADD_INPUT],
            input_nodes[LSTMInputs.BIAS_ADD_FORGET],
            input_nodes[LSTMInputs.BIAS_ADD_CELL],
            input_nodes[LSTMInputs.BIAS_ADD_OUTPUT],
            input_nodes[LSTMInputs.RECURRENT_ADD_INPUT],
            input_nodes[LSTMInputs.RECURRENT_ADD_FORGET],
            input_nodes[LSTMInputs.RECURRENT_ADD_CELL],
            input_nodes[LSTMInputs.RECURRENT_ADD_OUTPUT],
            input_nodes[LSTMInputs.INPUT_GATE_OUT],
            input_nodes[LSTMInputs.FORGET_GATE_OUT],
            input_nodes[LSTMInputs.CELL_GATE_OUT],
            input_nodes[LSTMInputs.OUTPUT_GATE_OUT],
            input_nodes[LSTMInputs.CELL_STATE],
            input_nodes[LSTMInputs.CELL_STATE_ACT_OUT],
            operator.outputs[0],
        ]

        error_nodes: list[tuple[str, list[float]]] = []

        def mse(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.mean((a - b) ** 2))

        for node, q_res, fp_res in zip(nodes, quant_results, floating_results):
            fpq_res = node.quant.dequantize(q_res, node.zero_point, node.scale)

            if fpq_res.shape != fp_res.shape:
                raise ValueError(
                    f"Shape mismatch ({node.name}), {fpq_res.shape=} != {fp_res.shape=}"
                )
            # fpq_res = np.swapaxes(fpq_res, 0, 1)
            # fp_res = np.swapaxes(fp_res, 0, 1)

            per_step_mse = [mse(a, b) for a, b in zip(fpq_res, fp_res)]
            error_nodes.append((node.name, per_step_mse))
        import pandas as pd

        df = pd.DataFrame(
            np.array([error[1] for error in error_nodes]).T,
            columns=[node[0] for node in error_nodes],
        )

        return error_nodes

    def quantize(self, input_result: QuantizationResult) -> QuantizationResult:
        """Quantizes the LSTM layer

        Args:
            input_result (QuantizationResult): The quantization result of the previous layer

        Raises:
            ValueError: Incorrect usage

        Returns:
            QuantizationResult: The quantization result of the LSTM layer
        """
        input_data = input_result.input_data
        input_node = input_result.out_node

        if input_node is None:
            raise ValueError(
                "LSTM: Input Node is None, the layer expects previous data"
            )

        # Calculate all the LSTM intermediate states
        lstm_states = LSTM(self._layer).call(input_data)

        # LSTM states
        recurrent_add = lstm_states.recurrent_add
        bias_add = lstm_states.bias_add

        # Quantize the static weight Nodes:
        kernel_nodes = self._quantize_kernel()
        recurrent_kernel_nodes = self._quantize_recurrent_kernel()
        bias_nodes = self._quantize_bias(input_node.scale, kernel_nodes)

        # Quantize the intermediate Nodes
        hidden_node = self._quantize_hidden_state(lstm_states.hidden_state)

        bias_add_nodes = self._quantize_bias_add(
            bias_add, recurrent_kernel_nodes, hidden_node.scale
        )
        recurrent_add_nodes = self._quantize_recurrent_add(recurrent_add)
        cell_node = self._quantize_cell_state(lstm_states.cell_state)

        #
        # Quantize the forget gates
        # Each gate has an individual recurrent_add quantized node as input
        #
        input_gate = deepcopy(recurrent_add_nodes[0])
        input_gate.name = f"{self._layer_name}/InputGate"
        input_gate_op, _ = quantize_activation_node(
            tf.math.sigmoid,
            ACTIVATION_NAMES.SIGMOID,
            recurrent_add[: self._n_units],
            self._q_activation,
            input_gate,
        )

        forget_gate = deepcopy(recurrent_add_nodes[1])
        forget_gate.name = f"{self._layer_name}/ForgetInputGate"
        forget_gate_op, _ = quantize_activation_node(
            tf.math.sigmoid,
            ACTIVATION_NAMES.SIGMOID,
            recurrent_add[:, :, self._n_units : 2 * self._n_units],
            self._q_activation,
            forget_gate,
        )
        cell_gate = deepcopy(recurrent_add_nodes[2])
        cell_gate.name = f"{self._layer_name}/CellGate"
        cell_gate_op, _ = quantize_activation_node(
            np.tanh,
            ACTIVATION_NAMES.TANH,
            recurrent_add[:, :, 2 * self._n_units : 3 * self._n_units],
            self._q_activation,
            cell_gate,
        )
        output_gate = deepcopy(recurrent_add_nodes[3])
        output_gate.name = f"{self._layer_name}/OutputGate"
        output_gate_op, _ = quantize_activation_node(
            tf.math.sigmoid,
            ACTIVATION_NAMES.SIGMOID,
            recurrent_add[:, :, 3 * self._n_units :],
            self._q_activation,
            output_gate,
        )

        #
        # Quantize the Cell State activation function
        #
        cell_state_node_lut = deepcopy(cell_node)
        cell_state_node_lut.name = f"{self._layer_name}/CellStateLUT"
        cell_state_activation, _ = quantize_activation_node(
            np.tanh,
            ACTIVATION_NAMES.TANH,
            lstm_states.cell_state,
            self._q_cell_state,
            cell_state_node_lut,
        )

        # Join all the activation operators
        activation_ops_none: list[None | Operator] = [
            input_gate_op,
            forget_gate_op,
            cell_gate_op,
            output_gate_op,
            cell_state_activation,
        ]
        if None in activation_ops_none:
            raise ValueError(
                "LSTM: Unexpected None in activation ops. None of the activations should be skipped"
            )

        # Convert the activation OPs into Activation Nodes
        activation_ops = cast(list[Operator], activation_ops_none)

        # Get the activation nodes
        activation_nodes = []
        for op in activation_ops:
            # Add the input LUT and the ouput per activation function
            activation_nodes.extend([op.inputs[0], op.outputs[0]])

        #
        # Create the LSTM Operator
        #
        lstm_attributes: dict[str, Attribute] = {
            "return_sequences": Attribute(value=[self._return_sequences], dtype="bool"),
            "n_cells": Attribute(value=[self._n_units], dtype="int"),
        }

        lstm_op = Operator(
            op_type=LayerIds.lstm,
            name=f"lstm_quant {self._layer_name}",
            inputs=[
                input_node,
                *kernel_nodes,
                *recurrent_kernel_nodes,
                *bias_nodes,
                *bias_add_nodes,
                *recurrent_add_nodes,
                cell_node,
                *activation_nodes,
            ],
            outputs=[hidden_node],
            layer=self._layer,
            attributes=lstm_attributes,
        )

        quantization_result = QuantizationResult(
            input_data=lstm_states.result,
            out_node=hidden_node,
            operators=[lstm_op],
            layer=self._layer,
        )

        if is_debug_mode():
            print(
                self._quantization_noise(
                    result=quantization_result,
                    fp_results=lstm_states,
                    fp_input=input_data,
                )
            )

        return quantization_result

    def adaround(
        self,
        optimizer_factory: Callable[[], IRoundOptimizer],
        results: QuantizationResult,
        cost_fn: Callable[[], float],
    ):
        if results.operators is None:
            raise ValueError("LSTM: Unexpected None in activation ops")

        op = results.operators[0]

        inp = op.inputs
        tensors = cast(
            list[NodeTensorTensor],
            [
                inp[LSTMInputs.BIAS_CELL],
                inp[LSTMInputs.BIAS_FORGET],
                inp[LSTMInputs.BIAS_INPUT],
                inp[LSTMInputs.BIAS_OUTPUT],
                inp[LSTMInputs.KERNEL_CELL],
                inp[LSTMInputs.KERNEL_FORGET],
                inp[LSTMInputs.KERNEL_INPUT],
                inp[LSTMInputs.KERNEL_OUTPUT],
                inp[LSTMInputs.RECURRENT_CELL],
                inp[LSTMInputs.RECURRENT_FORGET],
                inp[LSTMInputs.RECURRENT_INPUT],
                inp[LSTMInputs.RECURRENT_OUTPUT],
            ],
        )

        AdaRound(optimizer_factory(), cost_fn, self, results).round(tensors)


class MinMaxList(list):
    """Class that stores just the min and max value of a list of values or numpy arrays as tuples

    Args:
        list (_type_): _description_
    """

    def __init__(self, min_max: bool):
        super().__init__()
        self._min_max = min_max

    def append(self, __object) -> None:
        o = __object
        if self._min_max:
            o = np.min(__object), np.max(__object)  # type: ignore
        return super().append(o)


@dataclass
class LSTMResult:
    hidden_state: NPFP32
    cell_state: NPFP32
    bias_add: NPFP32
    recurrent_add: NPFP32
    input_gate: NPFP32
    forget_gate: NPFP32
    cell_gate: NPFP32
    output_gate: NPFP32
    cell_state_activation: NPFP32
    result: NPFP32


class LSTM:
    def __init__(self, layer: tf.keras.layers.LSTM, min_max_mode: bool = False) -> None:
        self._layer = layer
        self._W = np.array(layer.get_weights()[0])
        self._U = np.array(layer.get_weights()[1])
        self._b = np.array(layer.get_weights()[2])
        self._n_units = layer.units

        self._return_sequences = layer.return_sequences
        self._min_max_mode = min_max_mode  #

    def call(self, window_batch: NPFP32) -> LSTMResult:
        """Forward pass of the LSTM

        Args:
            window_batch (npa32): (batch_size, window_size, n_features), the input to the LSTM

        Returns:
            npa32: (batch_size, window_size, n_units), the output of the LSTM
        """
        return self(window_batch)

    def __call__(self, window_batch: NPFP32) -> LSTMResult:
        """Forward pass of the LSTM

        Args:
            window_batch (npa32): (batch_size, window_size, n_features), the input to the LSTM

        Returns:
            npa32: (batch_size, window_size, n_units), the output of the LSTM
        """
        W = self._W
        U = self._U
        B = self._b
        n_units = self._n_units
        batch_size = window_batch.shape[0]
        c = np.zeros((1, self._n_units), dtype=np.float32)

        h = [
            np.zeros((batch_size, self._n_units), dtype=np.float32)
        ]  # [timestep1 (batch, units), ]

        batched_timesteps = np.swapaxes(
            window_batch, 0, 1
        )  # (batch, timestep, features) -> (timestep, batch, features)

        previous_bias_add = []
        previous_recurrent_add = []
        previous_c = MinMaxList(self._min_max_mode)
        previous_i = MinMaxList(self._min_max_mode)
        previous_f = MinMaxList(self._min_max_mode)
        previous_o = MinMaxList(self._min_max_mode)
        previous_c_gate = MinMaxList(self._min_max_mode)
        previous_c = MinMaxList(self._min_max_mode)
        previous_c_act = MinMaxList(self._min_max_mode)
        for timestep_data in batched_timesteps:
            # timestep_data.shape -> (batch_size, n_features)
            Y_ = np.dot(timestep_data, W) + B
            Y = np.dot(h[-1], U) + Y_

            previous_bias_add.append(Y_)
            previous_recurrent_add.append(Y)

            i = tf.math.sigmoid(Y[:, :n_units])
            previous_i.append(i)

            f = tf.math.sigmoid(Y[:, n_units : 2 * n_units])
            previous_f.append(f)

            _c = np.tanh(Y[:, 2 * n_units : 3 * n_units])
            previous_c_gate.append(_c)

            o = tf.math.sigmoid(Y[:, 3 * n_units :])
            previous_o.append(o)

            c = f * c + i * _c

            previous_c.append(c)

            _c_act = np.tanh(c)
            previous_c_act.append(_c_act)
            _o = o * _c_act

            h.append(_o)

        result: np.ndarray = (
            np.swapaxes(h[1:], 0, 1) if self._return_sequences else h[-1]
        )

        return LSTMResult(
            hidden_state=np.array(h[1:]),  # Remove the initial scale
            cell_state=np.array(previous_c),
            bias_add=np.array(previous_recurrent_add),
            recurrent_add=np.array(previous_recurrent_add),
            input_gate=np.array(previous_i),
            forget_gate=np.array(previous_f),
            cell_gate=np.array(previous_c_gate),
            output_gate=np.array(previous_o),
            cell_state_activation=np.array(previous_c_act),
            result=result,
        )
