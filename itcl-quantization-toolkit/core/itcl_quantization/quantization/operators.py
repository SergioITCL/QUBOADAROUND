import re
from typing import Any, Callable, Generic, List, TypeVar, Union, Callable
import numpy as np
import logging


numpy_dtypes = Union[
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float64,
    np.float64,
]

q_dtypes = Union[numpy_dtypes, str, Any]

number = Union[int, float]

T = TypeVar("T", int, float)

input_type = Union[List[number], np.ndarray]

LUT_TYPE = TypeVar("LUT_TYPE", number, np.ndarray)


class GenericDtype:
    ALLOWED_STR_TYPES = ["int", "float", "uint"]

    def __init__(self, dtype: q_dtypes):
        if isinstance(dtype, str):
            self.__build_str_dtype(dtype)
        else:
            self.__build_numpy_dtype(dtype)

    def __build_numpy_dtype(self, dtype: numpy_dtypes):
        try:
            self.__min: number = np.iinfo(dtype).min
            self.__max: number = np.iinfo(dtype).max
            self.__repr: str = dtype.__name__  # type: ignore
        except:
            raise TypeError(f"{dtype} is not a valid numpy dtype")

    def __build_str_dtype(self, dtype: str):
        bits: int = int(re.findall(r"\d+", dtype)[0])
        type: str = re.findall(r"[a-zA-Z]+", dtype)[0].lower()
        signed = type == "int"
        if type not in self.ALLOWED_STR_TYPES:
            raise TypeError(f"{type} is not a valid type")

        if bits <= 0:
            raise TypeError(f"{bits} is not a valid number of bits")

        self.__min = -(2 ** (int(bits) - 1)) if signed else 0
        self.__max = 2 ** (int(bits) - 1) - 1 if signed else 2 ** int(bits) - 1
        self.__repr = f"{type}{bits}"

        if self.__repr != dtype.lower():
            raise TypeError(f"{dtype} is not a valid dtype")

    def min(self) -> number:
        return self.__min  # type: ignore

    def max(self) -> number:
        return self.__max

    def __repr__(self):
        return self.__repr

    def __str__(self):
        return self.__repr


class Quantization(Generic[T]):
    """Quantization Helper Class

    This class is used to help with quantization of the model.
    Args:
        Generic: T: int|float, native python type-hint. V: List[Union[int, float]]|np.ndarray, input data dtype
    """

    def __init__(self, dtype: Union[q_dtypes, str]):
        """Dtype of the quantization

        Args:
            dtype (Union[q_dtypes, str]): (int8, int16, int32, uint8, uint16, uint32)
        """

        self.dtype = GenericDtype(dtype)

    def max_value(self) -> T:
        """Max value of the quantization dtype

        Returns:
            T: Native python dtype
        """
        return self.dtype.max()  # type: ignore

    def min_value(self) -> T:
        """Min value of the quantization dtype

        Returns:
            T: Native python dtype
        """
        return self.dtype.min()  # type: ignore

    def quantize(
        self,
        data: input_type,
        zero_point: int,
        scale: float,
        rounding_policy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Quantizes a numpy array of float64 values to the quantization dtype

        Args:
            data (V): A numpy array of python list
            zero_point (int): Zero point of the quantization dtype
            scale (float): Scale of the quantization dtype

        Returns:
            np.ndarray: Quantized numpy array
        """

        float_res = np.array(np.array(data).astype(np.float64) / scale + zero_point)

        if rounding_policy is not None:
            try:
                return self.round(float_res + rounding_policy, extra=0)
            except ValueError:
                logging.warning(
                    "Invalid quantization rounding policy, rounding to nearest"
                )
        return self.round(float_res)

    def dequantize(self, data: input_type, zero_point: int, scale: float) -> np.ndarray:
        """Dequantizes a numpy array of quantized values to float64

        Args:
            data (V): A numpy array of python list
            zero_point (int): Zero point of the quantization dtype
            scale (float): Scale of the quantization dtype

        Returns:
            np.ndarray: Dequantized Numpy Array
        """
        return (np.array(data, dtype=np.int32) - zero_point) * scale

    def round(self, data: input_type, extra=0.5) -> np.ndarray:
        """Rounds a numpy array of float64 quantized values to their nearest int8 value

        Args:
            arr (np.ndarray): Numpy array of float64 quantized values.
        """
        rounded_data = np.floor(np.array(data) + extra)

        return rounded_data.clip(self.dtype.min(), self.dtype.max()).astype("int32")

    def float_activation(
        self,
        activation_fn: Callable[[LUT_TYPE], LUT_TYPE],
        input_s: float,
        input_zp: int,
    ) -> np.ndarray:
        quantized_input = np.array(
            range(int(self.min_value()), int(self.max_value()) + 1)
        )
        float_input = self.dequantize(quantized_input, input_zp, input_s)

        apply_fn = np.vectorize(activation_fn)

        return apply_fn(float_input)

    def create_LUT(
        self,
        activation_fn: Callable[[LUT_TYPE], LUT_TYPE],
        input_s: float,
        input_zp: int,
        output_s: float,
        output_zp: int,
        rounding_policy: np.ndarray | None = None,
    ) -> List[int]:

        activated_float = self.float_activation(activation_fn, input_s, input_zp)

        return self.quantize(
            activated_float, output_zp, output_s, rounding_policy=rounding_policy
        ).tolist()

    def dtype_str(self) -> str:
        """
        If the dtype is a string, return it. Otherwise, try to return the name of the class of the dtype. If
        that fails, return "Unknown"
        :return: The dtype of the object.
        """
        return str(self.dtype)


max_int32 = (1 << 32) - 1
min_int32 = -(1 << 32)


value_dtype = Union[number, np.ndarray]


class Dtype:
    def __init__(self, value: value_dtype, dtype: q_dtypes = None):

        if isinstance(dtype, str):
            self.dtype = np.dtype(dtype.lower())
        else:
            self.dtype = dtype
        if dtype is None:
            dtype = np.array(value).dtype
            self.dtype = dtype
        assert self.dtype is not None

        self.value = value
        self.bits: int = int(re.sub("[^0-9]", "", str(self)) or "32")
        self.type: str = re.sub(r"\d+", "", str(self))

    @classmethod
    def from_dtype(cls, dtype: Union[q_dtypes, str]):
        """
        Create a Quantization object from a dtype
        """
        return cls(0, dtype)

    def min(self) -> number:
        return np.iinfo(self.dtype).min  # type: ignore

    def max(self) -> number:
        return np.iinfo(self.dtype).max

    def __double_dtype(self):
        return np.dtype(f"{self.type}{min(64, self.bits * 2)}")

    @staticmethod
    def get_larger_dtype(a: "Dtype", b: "Dtype"):
        if a.bits > b.bits:
            return a
        else:
            return b

    def __add__(self, other):
        other = self.__adjust_other(other)
        larger = self.get_larger_dtype(self, other)
        next = Dtype.from_dtype(larger.__double_dtype())
        a = np.array(self.value).astype(next.dtype)
        b = np.array(other.value).astype(next.dtype)
        res = np.clip(a + b, next.min(), next.max())
        dtype = larger.dtype

        # If overflow: Upgrade the dtype to the next higher dtype
        if self.__has_overflow(res, self):
            dtype = next.dtype
        return Dtype(res, dtype)

    def __sub__(self, other):
        larger = self.get_larger_dtype(self, other)
        next = Dtype.from_dtype(larger.__double_dtype())

        a = np.array(self.value).astype(next.dtype)
        b = np.array(other.value).astype(next.dtype)
        res = np.clip(
            a - b,
            next.min(),
            next.max(),
        )

        dtype = larger.dtype

        if self.__has_overflow(res, self):
            dtype = next.dtype

        return Dtype(res, dtype)

    @staticmethod
    def __has_overflow(value: Union[np.ndarray, number], larger: "Dtype"):

        if isinstance(value, np.ndarray):
            return any(value > larger.max()) or any(value < larger.min())

        return value > larger.max() or value < larger.min()

    def __mul__(self, other):
        larger = self.get_larger_dtype(self, other)
        next = Dtype.from_dtype(larger.__double_dtype())

        a = np.array(self.value).astype(next.dtype)
        b = np.array(other.value).astype(next.dtype)
        res = np.clip(
            a * b,
            next.min(),
            next.max(),
        )

        dtype = larger.dtype
        if self.__has_overflow(res, self):
            dtype = next.dtype
        return Dtype(res, dtype)

    def __adjust_other(self, other):

        if isinstance(other, Dtype):
            return other
        else:
            return Dtype(other, self.dtype)

    def __str__(self):
        """
        If the dtype is a string, return it. Otherwise, try to return the name of the class of the dtype. If
        that fails, return "Unknown"
        :return: The dtype of the object.
        """
        if isinstance(self.dtype, str):
            return self.dtype

        try:
            return self.dtype.name  # type: ignore
        except Exception:
            return "Unknown"


int8_multiplication: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int16(np.int16(x) * np.int16(y))  # type: ignore
int8_addition: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int16(np.int16(x) + np.int16(y))  # type: ignore
int16_multiplication: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int32(np.int32(x) * np.int32(y))  # type: ignore
int16_addition: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int32(np.int32(x) + np.int32(y))  # type: ignore
int32_multiplication: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int64(np.int64(x) * np.int64(y))  # type: ignore
int32_addition: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.int64(np.int64(x) + np.int64(y))  # type: ignore
