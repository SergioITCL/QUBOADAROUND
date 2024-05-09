from __future__ import annotations

import numpy as np
from itcl_quantization import Quantization
from itcl_quantization.json.specification import Node, Operator

from itcl_inference_engine.layers.common.layer import ILayer
from itcl_inference_engine.layers.common.lut import LUTActivation
from itcl_inference_engine.util.floating import FloatingIntegerApprox


class LSTMIntermediateTensors:
    def __init__(self) -> None:
        self.bias_add: list[list[np.ndarray]] = []
        self.recurrent_add: list[list[np.ndarray]] = []
        self.input_gate_act: list[np.ndarray] = []
        self.forget_gate_act: list[np.ndarray] = []
        self.cell_gate_act: list[np.ndarray] = []
        self.output_gate_act: list[np.ndarray] = []
        self.cell_state_act: list[np.ndarray] = []
        self.cell_state: list[np.ndarray] = []
        self.hidden_state: np.ndarray


class LSTM(ILayer):
    """Quantized LSTM layer"""

    def __init__(
        self,
        /,
        kernel: list[np.ndarray],
        recurrent_kernel: list[np.ndarray],
        bias: list[np.ndarray],
        M1: list[float],
        M2: list[float],
        M3: float,
        M4: float,
        M5: float,
        input_gate_lut: LUTActivation,
        forget_gate_lut: LUTActivation,
        cell_gate_lut: LUTActivation,
        output_gate_lut: LUTActivation,
        cell_state_lut: LUTActivation,
        input_zp: int,
        bias_add_zp: int,
        recurrent_add_zp: list[int],
        hidden_state_zp: int,
        input_gate_zp: int,
        forget_gate_zp: int,
        cell_gate_zp: int,
        output_gate_zp: int,
        cell_state_zp: int,
        cell_state_activation_zp: int,
        bias_add_dtype: str,
        recurrent_add_dtype: str,
        cell_state_dtype: str,
        hidden_state_dtype: str,
        return_sequences: bool,
        fp_integer_only: bool = False,
    ) -> None:
        """LSTM layer.

        Args:
            kernels (list[np.ndarray]): Kernel tensor split into 4 equal parts
            recurrent_kernel (list[np.ndarray]): Recurrent kernel tensor split into 4 parts
            bias (list[np.ndarray]):  Bias tensor split into 4 parts
            M1 (list[float]): 4 precalculated M1 values, to scale each of the 4 bias_add
            M2 (list[float]): 4 precalculated M2 values, to scale each of the 4 recurrent kernels
            M3 (float): To calculate the cell state
            M4 (float): To calculate the cell state
            M5 (float): Scales down the output
            input_gate_lut (LUTActivation): Look Up Table of the gate
            forget_gate_lut (LUTActivation): Look Up Table of the gate
            cell_gate_lut (LUTActivation): Look Up Table of the gate
            output_gate_lut (LUTActivation): Look Up Table of the gate
            cell_state_lut (LUTActivation): Look Up Table of the gate
            input_zp (int): Input Zero Point
            bias_add_zp (int): Bias Add Zero Point
            recurrent_add_zp (list[int]): Recurrent Add Zero Point
            hidden_state_zp (int): _description_
            input_gate_zp (int): _description_
            forget_gate_zp (int): _description_
            cell_gate_zp (int): _description_
            output_gate_zp (int): _description_
            cell_state_zp (int): _description_
            cell_state_activation_zp (int): _description_
            bias_add_dtype (str): _description_
            recurrent_add_dtype (str):
            cell_state_dtype (str):
            hidden_state_dtype (str):
            return_sequences (bool):
            fp_integer_only (bool, optional): If the inference should be done without any floating point operations. Defaults to False.
        """
        self._kernel = kernel
        self._recurrent_kernel = recurrent_kernel
        self._bias = bias
        self._M1 = M1  # FloatingIntegerApprox(M1, integer_only=fp_integer_only)
        self._M2 = M2  # FloatingIntegerApprox(M2, integer_only=fp_integer_only)
        self._M3 = FloatingIntegerApprox(M3, integer_only=fp_integer_only)
        self._M4 = FloatingIntegerApprox(M4, integer_only=fp_integer_only)
        self._M5 = FloatingIntegerApprox(M5, integer_only=fp_integer_only)
        self._input_gate_lut = input_gate_lut
        self._forget_gate_lut = forget_gate_lut
        self._cell_gate_lut = cell_gate_lut
        self._output_gate_lut = output_gate_lut
        self._cell_state_lut = cell_state_lut

        self._input_zp = input_zp
        self._bias_add_zp = bias_add_zp
        self._recurrent_add_zp = recurrent_add_zp
        self._hidden_state_zp = hidden_state_zp
        self._input_gate_zp = input_gate_zp
        self._forget_gate_zp = forget_gate_zp
        self._cell_gate_zp = cell_gate_zp
        self._output_gate_zp = output_gate_zp
        self._cell_state_zp = cell_state_zp
        self._cell_state_activation_zp = cell_state_activation_zp

        self._q_bias_add = Quantization(bias_add_dtype)
        self._q_recurrent_add = Quantization(recurrent_add_dtype)
        self._q_cell_state = Quantization(cell_state_dtype)
        self._q_hidden_state = Quantization(hidden_state_dtype)

        self._n_cells = self._kernel[0].shape[-1]
        self._n_features = self._kernel[0].shape[0]

        self._return_sequences = return_sequences

        self.intermediate_tensors = LSTMIntermediateTensors()

    @staticmethod
    def from_model(operator: Operator):

        input_ = operator["inputs"][0]

        kernel = operator["inputs"][1:5]
        recurrent = operator["inputs"][5:9]
        bias = operator["inputs"][9:13]
        bias_add = operator["inputs"][13:17]
        recurrent_add = operator["inputs"][17:21]
        cell_state = operator["inputs"][21]
        input_gate_in, input_gate_out = operator["inputs"][22:24]
        forget_gate_in, forget_gate_out = operator["inputs"][24:26]
        cell_gate_in, cell_gate_out = operator["inputs"][26:28]
        output_gate_in, output_gate_out = operator["inputs"][28:30]
        cell_state_act_in, cell_state_act_out = operator["inputs"][30:32]

        hidden_state = operator["outputs"][0]
        attr = operator["attributes"] or {}

        return LSTM.build(
            input_=input_,
            kernel_nodes=kernel,
            recurrent_kernel_nodes=recurrent,
            bias_nodes=bias,
            bias_add_nodes=bias_add,
            cell_state=cell_state,
            recurrent_add_nodes=recurrent_add,
            hidden_state=hidden_state,
            input_gate_in=input_gate_in,
            input_gate_out=input_gate_out,
            forget_gate_in=forget_gate_in,
            forget_gate_out=forget_gate_out,
            cell_gate_in=cell_gate_in,
            cell_gate_out=cell_gate_out,
            output_gate_in=output_gate_in,
            output_gate_out=output_gate_out,
            cell_state_in=cell_state_act_in,
            cell_state_out=cell_state_act_out,
            return_sequences=attr["return_sequences"]["value"][0],
        )

    @staticmethod
    def build(
        input_: Node,
        kernel_nodes: list[Node],
        recurrent_kernel_nodes: list[Node],
        bias_nodes: list[Node],
        bias_add_nodes: list[Node],
        recurrent_add_nodes: list[Node],
        cell_state: Node,
        hidden_state: Node,
        input_gate_in: Node,
        input_gate_out: Node,
        forget_gate_in: Node,
        forget_gate_out: Node,
        cell_gate_in: Node,  # Look Up Tables
        cell_gate_out: Node,  # Look Up Tables
        output_gate_in: Node,  # Look Up Tables
        output_gate_out: Node,  # Look Up Tables
        cell_state_in: Node,
        cell_state_out: Node,
        return_sequences: bool,
    ) -> LSTM:

        assert cell_state_out["tensor"] is not None
        assert hidden_state["tensor"] is not None

        # Kernel
        kernel_arr = [np.array(node["tensor"]["tensor"]) - node["zero_point"] for node in kernel_nodes]  # type: ignore

        # Recurrent Kernel
        # q_rkernel = Quantization(recurrent_kernel_nodes["tensor"]["dtype"])
        recurrent_kernel_arr = [np.array(node["tensor"]["tensor"]) - node["zero_point"] for node in recurrent_kernel_nodes]  # type: ignore

        # Bias
        # q_bias = Quantization(bias["tensor"]["dtype"])
        bias_arr = [np.array(node["tensor"]["tensor"]) - node["zero_point"] for node in bias_nodes]  # type: ignore

        M1: list[float] = []
        for kernel_node, recurrent_kernel_node in zip(
            kernel_nodes, recurrent_kernel_nodes
        ):
            M1.append(kernel_node["scale"] * input_["scale"] / (hidden_state["scale"] * recurrent_kernel_node["scale"]))  # type: ignore

        M2: list[float] = []
        for recurrent_kernel_node, recurrent_add in zip(
            recurrent_kernel_nodes, recurrent_add_nodes
        ):
            M2.append(recurrent_kernel_node["scale"] * hidden_state["scale"] / recurrent_add["scale"])  # type: ignore
        # M1: float = kernel_nodes["scale"] * input_["scale"] / (hidden_state["scale"] * recurrent_kernel_nodes["scale"])  # type: ignore
        # M2: float = recurrent_kernel_nodes["scale"] * hidden_state["scale"] / recurrent_add_nodes["scale"]  # type: ignore
        M3: float = forget_gate_out["scale"]  # type: ignore
        M4: float = input_gate_out["scale"] * cell_gate_out["scale"] / cell_state["scale"]  # type: ignore
        M5: float = output_gate_out["scale"] * cell_state_out["scale"] / hidden_state["scale"]  # type: ignore

        input_gate_lut = LUTActivation.from_node(input_gate_in)
        forget_gate_lut = LUTActivation.from_node(forget_gate_in)
        cell_gate_lut = LUTActivation.from_node(cell_gate_in)
        output_gate_lut = LUTActivation.from_node(output_gate_in)
        cell_state_lut = LUTActivation.from_node(cell_state_in)

        return LSTM(
            kernel=kernel_arr,
            recurrent_kernel=recurrent_kernel_arr,
            bias=bias_arr,
            M1=M1,
            M2=M2,
            M3=M3,
            M4=M4,
            M5=M5,
            # Look Up tables:
            input_gate_lut=input_gate_lut,
            forget_gate_lut=forget_gate_lut,
            cell_gate_lut=cell_gate_lut,
            output_gate_lut=output_gate_lut,
            cell_state_lut=cell_state_lut,
            # Zero Points:
            input_zp=input_["zero_point"] or 0,
            bias_add_zp=0,
            recurrent_add_zp=[node["zero_point"] or 0 for node in recurrent_add_nodes],
            hidden_state_zp=hidden_state["zero_point"] or 0,
            input_gate_zp=input_gate_out["zero_point"] or 0,
            forget_gate_zp=forget_gate_out["zero_point"] or 0,
            cell_gate_zp=cell_gate_out["zero_point"] or 0,
            output_gate_zp=output_gate_out["zero_point"] or 0,
            cell_state_zp=cell_state["zero_point"] or 0,
            cell_state_activation_zp=cell_state_out["zero_point"] or 0,
            # dtypes:
            bias_add_dtype=bias_add_nodes[0]["tensor"]["dtype"],  # type: ignore
            recurrent_add_dtype=recurrent_add_nodes[0]["tensor"]["dtype"],  # type: ignore
            cell_state_dtype=cell_state_out["tensor"]["dtype"],
            hidden_state_dtype=hidden_state["tensor"]["dtype"],
            return_sequences=return_sequences,
        )

    def infer(self, input_: np.ndarray) -> np.ndarray:

        # batch, window_len, n_features =  -> w, b , n
        input_norm = (input_ - self._input_zp).swapaxes(0, 1)

        # Create the initial hidden state and cell state
        hidden_state = [np.zeros((1, self._n_cells), dtype=np.int64)]
        cell_state = np.zeros((1, self._n_cells), dtype=np.int64)

        inter = LSTMIntermediateTensors()

        # Iterate each timestep
        for input_timestep in input_norm:

            bias_add: list[np.ndarray] = []  # 4 intermediate arrays

            for i, (kernel, bias, M1) in enumerate(
                zip(self._kernel, self._bias, self._M1)
            ):
                bias_add_large = (input_timestep @ kernel) + bias
                bias_add.append((M1 * bias_add_large).astype(np.int64))

            recurrent_add: list[np.ndarray] = []  # 4 intermediate arrays

            for i, (recurrent_kernel, bias, M2) in enumerate(
                zip(self._recurrent_kernel, bias_add, self._M2)
            ):
                recurrent_add_large = (hidden_state[-1] @ recurrent_kernel) + bias
                recurrent_add.append(
                    (recurrent_add_large * M2 + self._recurrent_add_zp[i])
                    .clip(
                        self._q_recurrent_add.min_value(),
                        self._q_recurrent_add.max_value(),
                    )
                    .astype(np.int64)
                )

            inter.bias_add.append(bias_add)
            inter.recurrent_add.append(recurrent_add)

            # Activate the kernel_add tensors
            activated_input_gate = self._input_gate_lut.infer(recurrent_add[0])
            inter.input_gate_act.append(activated_input_gate)
            activated_forget_gate = self._forget_gate_lut.infer(recurrent_add[1])
            inter.forget_gate_act.append(activated_forget_gate)
            activated_cell_gate = self._cell_gate_lut.infer(recurrent_add[2])
            inter.cell_gate_act.append(activated_cell_gate)
            activated_output_gate = self._output_gate_lut.infer(recurrent_add[3])
            inter.output_gate_act.append(activated_output_gate)

            # Calculate & update the new cell state
            cell_state = (
                (
                    self._cell_state_zp
                    + (
                        self._M3
                        * (activated_forget_gate - self._forget_gate_zp)
                        * (cell_state - self._cell_state_zp)
                    )
                    + self._M4
                    * (
                        (activated_input_gate - self._input_gate_zp)
                        * (activated_cell_gate - self._cell_gate_zp)
                    )
                )
                .astype(np.int64)
                .clip(self._q_cell_state.min_value(), self._q_cell_state.max_value())
            )

            inter.cell_state.append(cell_state)

            activated_cell_state = self._cell_state_lut.infer(
                cell_state
            )  # .clip(-128, 127)
            inter.cell_state_act.append(activated_cell_state)

            new_hidden_state = (
                (
                    self._hidden_state_zp
                    + (
                        self._M5
                        * (
                            (activated_output_gate - self._output_gate_zp)
                            * (activated_cell_state - self._cell_state_activation_zp)
                        )
                    )
                )
                .astype(np.int64)
                .clip(
                    self._q_hidden_state.min_value(), self._q_hidden_state.max_value()
                )
            )

            hidden_state.append(new_hidden_state)

        hs = np.array(hidden_state[1:])  # Remove the initial step (all zeros)
        inter.hidden_state = hs
        self.intermediate_tensors = inter
        if self._return_sequences:
            return hs.swapaxes(0, 1)  # STEP, BATCH, DATA -> B, S, D

        return hs[-1]  # BATCH, DATA
