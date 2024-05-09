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
class TanhLUT(BaseOperator):
    def __init__(
        self, interpreter: tfl.Interpreter, layer_name2idx: Dict[str, int], model: Model
    ):
        super().__init__(interpreter, layer_name2idx, model)

    def build(self, operator: Operator) -> OperatorJson:
        """Override method that transposes the weights matrix"""

        self.__quantization = Quantization(np.int8)
        op = super().build(operator)
        self.__input_s = op["inputs"][0]["scale"] or 1.0
        self.__input_zp = op["inputs"][0]["zero_point"] or 0

        self.__output_s = op["outputs"][0]["scale"] or 1.0
        self.__output_zp = op["outputs"][0]["zero_point"] or 0

        op["op_type"] = "TANHLUT"
        op["name"] = "TANHLUT"

        LUT = self.__build_LUT()

        
        op["inputs"][0]["LUT"] = LUT
        return op

    def __build_LUT(self) -> LUT_json:
        LUT =  self.__quantization.create_LUT(
            math.tanh,
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
