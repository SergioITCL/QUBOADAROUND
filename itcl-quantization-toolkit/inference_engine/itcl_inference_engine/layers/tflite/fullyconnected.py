import math

import numpy as np
from itcl_quantization import Quantization
from itcl_quantization.json.specification import Operator

from itcl_inference_engine.layers.common.layer import ILayer
from itcl_inference_engine.util.quantization import (int16_multiplication,
                                                     int32_addition,
                                                     int32_multiplication,
                                                     max_int32, min_int32)


class FullyConnected(ILayer):
    """Dense Layer

    FULLY_CONNECTED

    Input 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
    Input 1 (Weight):
        data_type  : int8
        range      : [-127, 127]
        granularity: per-tensor
        restriction: zero_point = 0
    Input 2 (Bias):
        data_type  : int32
        range      : [int32_min, int32_max]
        granularity: per-tensor
        restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
    Output 0:
        data_type  : int8
        range      : [-128, 127]
        granularity: per-tensor
    """

    def __init__(
        self,
        input_zp,
        weights,
        bias,
        quant_mult_params: "FullyConnectedQuantizedMultiplier",
        output_zp,
        input_dtype: str = "int8",
        output_dtype: str = "int8",
        weights_dtype: str = "int8",
        bias_dtype: str = "int32",
    ) -> None:
        """
        The function takes in the input_zp, weights, bias, quant_mult_params, and output_zp.

        The function then initializes the super class, and sets the input_zp, output_zp, weights, bias,
        quantized_multiplier, shift, and round.

        The function then returns None.

        :param input_zp: the zero point of the input tensor
        :param weights: the weights of the fully connected layer
        :param bias: The bias vector
        :param quant_mult_params: "FullyConnectedQuantizedMultiplier"
        :type quant_mult_params: "FullyConnectedQuantizedMultiplier"
        :param output_zp: The zero point of the output tensor
        """
        super().__init__()
        self.__input_zp = input_zp
        self.__output_zp = output_zp

        self.__weights: np.ndarray = weights
        self.__bias: np.ndarray = np.array(
            bias
        )  # np.expand_dims(bias, axis=1) # expand for batch matmul

        self.__quantized_multiplier = quant_mult_params.quantized_multiplier
        self.__shift = quant_mult_params.shift
        self.__round = quant_mult_params.round

        self.__q_output = Quantization(output_dtype)

    @staticmethod
    def build(
        input_s,
        input_zp,
        weight_s,
        weight_zp,
        weights,
        bias_s,
        bias_zp,
        bias,
        output_s,
        output_zp,
        output_dtype="int8",
    ):
        """
        It takes in the input, weight, and bias tensors, and the quantization parameters for each, and
        returns a fully connected layer.

        :param input_s: the scale of the input
        :param input_zp: the zero point of the input tensor
        :param weight_s: the scale of the weights
        :param weight_zp: the zero point of the weights
        :param weights: the weights of the fully connected layer
        :param bias_s: bias scale
        :param bias_zp: bias zero point
        :param bias: the bias vector
        :param output_s: output scale
        :param output_zp: The zero point of the output
        :return: A fully connected layer with the given parameters.
        """
        quant_mult_params = FullyConnectedQuantizedMultiplier(
            input_s, weight_s, output_s
        )

        weights = weights - weight_zp

        return FullyConnected(
            input_zp,
            weights,
            bias,
            quant_mult_params,
            output_zp,
            output_dtype=output_dtype,
        )

    @staticmethod
    def from_model(operator: Operator):
        """Layer Builder from a Operator

        This operator has 3 input nodes and 1 output node
        INPUT:
            - Input Node
            - Weight Node (Has a Tensor with the dense weights0)
            - Bias Node (Has a Tensor with the dense bias )

        Args:
            operator (Operator): Json Operator with all the layer nodes (inputs and outputs)

        Returns:
            A FullyConnected Layer Instance
        """

        input_s = operator["inputs"][0]["scale"]
        input_zp = operator["inputs"][0]["zero_point"]

        weight_s = operator["inputs"][1]["scale"]
        weight_zp = operator["inputs"][1]["zero_point"]
        weights = np.array(operator["inputs"][1]["tensor"]["tensor"] or [])  # type: ignore

        bias_s = operator["inputs"][2]["scale"]
        bias_zp = operator["inputs"][2]["zero_point"]
        bias = np.array(operator["inputs"][2]["tensor"]["tensor"] or [])  # type: ignore

        out_s = operator["outputs"][0]["scale"]
        out_zp = operator["outputs"][0]["zero_point"]
        out_dtype = operator["outputs"][0]["tensor"]["dtype"]  # type: ignore
        return FullyConnected.build(
            input_s,
            input_zp,
            weight_s,
            weight_zp,
            weights,
            bias_s,
            bias_zp,
            bias,
            out_s,
            out_zp,
            output_dtype=out_dtype,
        )

    def __calculate_accumulator(self, input_: np.ndarray):
        """Helper Method that calculates the multiplication of the Kernel * Inpunt + Bias

        Args:
            input (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        weights_matrix = self.__weights.astype(np.int64)

        bias_vector = self.__bias.astype(np.int64)
        z1 = self.__input_zp

        input_ = input_.astype(np.int64) - np.int64(z1)

        # int64 (as the bias is int32 and the result of the multiplication int16)

        if input_.ndim == 2:
            W = np.expand_dims(weights_matrix.T, 2)
            W = weights_matrix.T
            mmul_res = np.matmul(input_, W)
        else:
            mmul_res = np.matmul(weights_matrix, input_)
        return mmul_res + self.__bias

        accumulator = []
        for weight_vector, bias in zip(weights_matrix, bias_vector):
            # Initialized the accumulator with the bias vector
            accumulator_value = bias

            for input_value, weight in zip(input_, weight_vector):
                multiplication_value = int16_multiplication(
                    input_value, weight
                )  # int16 tensor

                # Accumulator (bias) + (Input * Weight)
                accumulator_value = int32_addition(
                    accumulator_value, multiplication_value
                )  # int16 tensor

            accumulator.append(accumulator_value)

        accumulator = np.array(accumulator, dtype="int64")

        return accumulator

    def __multiply_by_quantized_multiplier(self, accumulator: np.ndarray):
        """Helper Method that scales down the result of the matrix multiplication.

        Args:
            accumulator (int64): Output of the calculate accumulator method

        Returns:
            int32: Result of the multiplication
        """
        accumulator = accumulator.astype(np.int64)
        quantized_multiplier = self.__quantized_multiplier.astype(np.int64)
        rounded = self.__round.astype(np.int64)
        res = accumulator * quantized_multiplier + rounded

        return res >> self.__shift
        result = []

        for accumulator_value in accumulator:
            result_value = (
                int32_multiplication(accumulator_value, self.__quantized_multiplier)
                + self.__round
            )
            result_value = np.int32(result_value >> self.__shift)

            assert result_value >= min_int32
            assert result_value <= max_int32

            result.append(result_value)

        """
            MNIST TANH ACC (depending on result variable Dtype):
            int32: 91.15% (Original C++ Dtype)
            int16: 91.15%
            int8 (Direct Cast):  90.97% 
            int8 (Clip/Clamp/Round): 91.15% -> extra operations
        """
        return np.array(
            result, dtype="int16"
        )  # The result should be a number near int8

    def __add_output_zero_point(self, multiplied_accumulator: np.ndarray):
        """
        Helper Method that adds the output zero point to the result of the matrix multiplication.

        Args:
            multiplied_accumulator (int16): Result of the matrix multiplication
        """

        # int8 (as int16 without the clamp op) + int8
        return multiplied_accumulator + np.int64(self.__output_zp)

    def infer(self, input_: np.ndarray):

        assert input_.ndim <= 2, "Input Dimension is greater than 2"

        accumulator = self.__calculate_accumulator(input_)
        multiplied_accumulator = self.__multiply_by_quantized_multiplier(accumulator)
        result = self.__add_output_zero_point(multiplied_accumulator)
        # Cast back to INT8
        output = np.clip(
            result, self.__q_output.min_value(), self.__q_output.max_value()
        )
        return output


class FullyConnectedQuantizedMultiplier:
    """Helper Class that calculates the Quantized Multiplier and the Shift for the Fully Connected Layer"""

    def __init__(
        self,
        input_scale: np.float32,
        weight_scale: np.float32,
        output_scale: np.float32,
    ):
        """Layer Constructor

        Args:
            input_scale (np.float32): Input Scale
            weight_scale (np.float32): Weight Scale
            output_scale (np.float32): Output (Bias_Add result or the input to the activation layer) scale
        """
        params = self.__build(input_scale, weight_scale, output_scale)

        self.quantized_multiplier = params[0]
        self.shift = params[1]
        self.round = params[2]

    def __adjust_mult(self, quantized_multiplier: np.float32, shift: np.int64):
        """Method to adjust the Quantized Multiplier and the Shift

        Args:
            quantized_multiplier (np.float32): Calculated Quantized Multiplier
            shift (np.int64): Current Shift

        Returns:
            int32 tuple: Quantized Multplier (M), shift and Round
        """
        assert quantized_multiplier >= 0
        assert shift >= -31 and shift <= 30

        shift = int32_addition(31, -shift)  # type: ignore

        round_ = np.int64(1 << (shift - 1))

        return quantized_multiplier, shift, round_

    def __build(
        self,
        input_scale: np.float32,
        weight_scale: np.float32,
        output_scale: np.float32,
    ):
        """Builds and adjust the float multiplier.

        This multiplier is a  large int32 number (EG: 82147483648)

        Returns:
            int32 tuple: Quantized Multplier (M), shift and Round
        """

        float_multiplier = input_scale * weight_scale / output_scale

        if float_multiplier < 1e-6:
            return self.__adjust_mult(np.float32(0), np.int64(0))

        q, shift = math.frexp(float_multiplier)
        quantized_multiplier = np.int64(round(q * (1 << 31)))
        assert quantized_multiplier <= (1 << 31)

        if quantized_multiplier == (1 << 31):
            quantized_multiplier /= 2
            shift += 1

        assert quantized_multiplier <= max_int32

        if shift < -31:
            shift = 0
            quantized_multiplier = 0
        return self.__adjust_mult(quantized_multiplier, shift)  # type: ignore
