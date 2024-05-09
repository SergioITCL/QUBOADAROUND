import math
from typing import Dict
import numpy as np
from tflite import Model, Operator
from itcl_tflite2json.operators.BaseOperator import BaseOperator
from itcl_quantization.json.specification import Operator as OperatorJson, LUT as LUT_json
import tensorflow.lite as tfl

from itcl_quantization import Quantization
from itcl_quantization.quantization.lut import ReducedLUT
from itcl_tflite2json.util.settings import settings

# A function that takes in a value (x) and returns the sigmoid of that value.
sigmoid_fn = lambda x: 1 / (1 + math.exp(-x))

class SigmoidLUT(BaseOperator):
    def __init__(
        self, interpreter: tfl.Interpreter, layer_name2idx: Dict[str, int], model: Model
    ):
        """
        Layer constructor
        
        :param interpreter: tfl.Interpreter
        :type interpreter: tfl.Interpreter
        :param layer_name2idx: A dictionary mapping layer names to indices
        :type layer_name2idx: Dict[str, int]
        :param model: The model object that was used to create the interpreter
        :type model: Model
        """
        super().__init__(interpreter, layer_name2idx, model)

    def build(self, operator: Operator) -> OperatorJson:
        """Override method that transposes the weights matrix"""

        self.__quantization = Quantization(np.int8)
        op = super().build(operator)
        self.__input_s = op["inputs"][0]["scale"] or 1.0
        self.__input_zp = op["inputs"][0]["zero_point"] or 0

        self.__output_s = op["outputs"][0]["scale"] or 1.0
        self.__output_zp = op["outputs"][0]["zero_point"] or 0

        op["op_type"] = "SIGMOIDLUT"
        op["name"] = "SIGMOIDLUT"

        LUT = self.__build_LUT()

        
        op["inputs"][0]["LUT"] = LUT
        return op

    def __build_LUT(self) -> LUT_json:
        """
        It takes a sigmoid function, and creates a lookup table for it
        :return: A dictionary with the keys "LUT", "offset", and "reduced_LUT".
        """
        LUT =  self.__quantization.create_LUT(
            sigmoid_fn,
            self.__input_s,
            self.__input_zp,
            self.__output_s,
            self.__output_zp,
        )   
        assert len(LUT) == 256
        lut_settings = settings["settings"]["lut"]

        reduced_lut = ReducedLUT(LUT, lut_settings["lut_depth"], lut_settings["min_removal"])
        return {
            "LUT": LUT, 
            "offset": 128,
            "reduced_LUT": reduced_lut.serialize()
        }
