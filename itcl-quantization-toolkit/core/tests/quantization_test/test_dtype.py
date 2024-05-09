import numpy as np
from itcl_quantization.quantization.operators import Dtype

class TestDtype():

    
    def test_init(self):
        dtype = Dtype(-128, dtype="int8")
        print(dtype)

        assert dtype.type == "int"
        assert dtype.bits == 8

        assert (str(dtype) == "int8")


    def test_invalid_dtype(self):

        try:
            dtype = Dtype(0, dtype="int22")
            assert False
        except Exception:
            assert True

    def test_max_min(self):
        dtype = Dtype(0, dtype="int8")
        assert dtype.min() == -128
        assert dtype.max() == 127

        dtype = Dtype(0, dtype="int16")
        assert dtype.min() == -32768
        assert dtype.max() == 32767

        dtype = Dtype(0, dtype="uint8")
        assert dtype.min() == 0
        assert dtype.max() == 255


    def test_add(self):
        """
        
        """
        dtype1 = Dtype(-128, dtype="int8")
        dtype2 = Dtype(-128, dtype="int8")
        dtype3 = dtype1 + dtype2

        assert dtype3.type == "int"
        assert dtype3.bits == 16

        assert dtype3.value == -128 * 2


    def test_sub(self):
        dtype1 = Dtype(-128, dtype="int8")
        dtype2 = Dtype(128, dtype="int8")
        dtype3 = dtype1 - dtype2

        assert dtype3.type == "int"
        assert dtype3.bits == 16

        assert dtype3.value == -256

    def test_mul(self):
        dtype1 = Dtype(-128, dtype="int8")
        dtype2 = Dtype(-128, dtype="int8")
        dtype3 = dtype1 * dtype2

        assert dtype3.type == "int"
        assert dtype3.bits == 16

        assert dtype3.value == -128 * -128


    def test_int8_int16(self):
        dtype1 = Dtype(-128, dtype="int8")
        dtype2 = Dtype(400, dtype="int16")
        dtype3 = dtype1 * dtype2

        assert dtype3.type == "int"
        assert dtype3.bits == 32

        assert dtype3.value == -128 * 400 

    def test_int8_int32(self):
        dtype1 = Dtype(-128, dtype="int8")
        dtype2 = Dtype(2147483647, dtype="int32")
        dtype3 = dtype1 * dtype2

        assert dtype3.type == "int"
        assert dtype3.bits == 64

        assert dtype3.value == -128 * 2147483647


    def test_int8_no_overflow(self):

        dtype1 = Dtype(-128, dtype="int8")
        dtype2 = Dtype(0, dtype="int8")
        dtype3 = dtype1 - dtype2

        assert dtype3.type == "int"
        assert dtype3.bits == 8

        assert dtype3.value == -128


    def test_arrays_plus_literals(self):
        a = np.array([-128, -128, -128])

        dtype = Dtype(a, dtype="int8")
        res = dtype + 128

        assert res.type == "int"
        assert res.bits == 8
        assert np.array(res.value).sum() == 0
        
