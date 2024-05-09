import math
from random import randint, random
from typing import Any, Callable
import numpy as np
from itcl_quantization import Quantization
from itcl_quantization.quantization.lut import ReducedLUT

class TestReducedLut():
    
    uint8 = Quantization(np.uint8)
    int8 = Quantization(np.int8)

    def assert_eq(self, original, reduced):
        for real, expected in zip(original, reduced):
            assert real == expected, f"{real} != {expected}"

    def test_uint8_base_tanh(self):
        lut = self.uint8.create_LUT(math.tanh, 0.058881718665361404, 151, 0.00784310046583414, 128)
        reduced_lut = ReducedLUT(lut, 3, 0)
        self.assert_eq(lut, reduced_lut)

    
    def test_uint8_small_tanh(self):
        lut = self.uint8.create_LUT(math.tanh, 1, 0, 1, 0)
        reduced_lut = ReducedLUT(lut, 3, 0)

        self.assert_eq(lut, reduced_lut)

    def test_depth_0(self):
        lut = self.uint8.create_LUT(math.tanh, 1, 0, 1, 0)
        reduced_lut = ReducedLUT(lut, 0, 0)

        self.assert_eq(lut, reduced_lut)

    def __random(self, quantization: Quantization, fn: Callable[[Any], Any], min_reduce: int, depth: int):
        for _ in range(20):
            input_s = random()
            input_zp = randint(quantization.min_value(), quantization.max_value())
            output_s = random()
            output_zp = randint(quantization.min_value(), quantization.max_value())

            lut = quantization.create_LUT(fn, input_s, input_zp, output_s, output_zp)
            reduced_lut = ReducedLUT(lut, min_reduce, depth)
            try:

                list(reduced_lut)
                self.assert_eq(lut, reduced_lut)
            except:
                print(f"Failed for fn={fn}, input_s={input_s}, input_zp={input_zp}, output_s={output_s}, output_zp={output_zp}")
                print(reduced_lut.serialize())
                print(f"Expected LUT: {lut}")
                raise
    def test_random_uint_tanh(self):
        self.__random(self.uint8, math.tanh, 3, 0)
    
    def test_random_int_tanh(self):
        self.__random(self.int8, math.tanh,  3, 0)
    
    def test_random_uint_cos(self):
        self.__random(self.uint8, math.cos,  3, 0)
    
    def test_random_int_cos(self):
        self.__random(self.int8, math.cos,  3, 0)


    def test_serialize_deserialize(self):
        lut = self.uint8.create_LUT(math.tanh, 1, 0, 1, 0)
        reduced_lut = ReducedLUT(lut, 3, 0)
        serialized = reduced_lut.serialize()
        deserialized = ReducedLUT.deserialize(serialized)
        self.assert_eq(lut, deserialized)

slice_test = [1, 2, 3]
print(slice_test[0:None])