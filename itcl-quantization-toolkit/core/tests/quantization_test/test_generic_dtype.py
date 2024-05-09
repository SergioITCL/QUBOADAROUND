from itcl_quantization.quantization.operators import GenericDtype

class  TestGenericDtype():


    def test_string(self):
        dtype = GenericDtype("int8")

        assert dtype.min() == -128
        assert dtype.max() == 127

        assert str(dtype) == "int8"


    def test_dizzy_string(self):
        dtype = GenericDtype("InT4")
        assert dtype.min() == -8
        assert dtype.max() == 7
        assert str(dtype) == "int4"

    def test_invalid_dtypes(self):

        try:
            dtype = GenericDtype("integer22")
            assert False
        except TypeError:
            assert True
        
        try:
            dtype = GenericDtype("int-22")
            assert False
        except TypeError:
            assert True

        try:
            dtype = GenericDtype("float0")
            assert False

        except TypeError:
            assert True


    def test_uint(self):

        dtype = GenericDtype("uint8")

        assert dtype.min() == 0
        assert dtype.max() == 255

        assert str(dtype) == "uint8"



        