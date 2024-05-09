import tflite
from inspect import getmembers, isroutine


# Dictionary of tflite dtypes codes (INTEGER) to string (only god and i knew how to understand this onliner, just print it)
_dtype_lut = {
    code: string
    for string, code in filter(
        lambda x: "__" not in x[0],
        getmembers(tflite.TensorType, lambda a: not (isroutine(a))),
    )
}


def dtype2str(code: int):
    """Translates a given tflite code to a string.

    Args:
        code (int): Tflite Dtype code

    Returns:
        string: Code as as string, a "?" if it was not found
    """
    return _dtype_lut.get(code, "?")


# Dictionary of tflite layers code to string
_layers_lut = {
    code: string
    for string, code in filter(
        lambda x: "__" not in x[0],
        getmembers(tflite.BuiltinOperator, lambda a: not (isroutine(a))),
    )
}


def layers2str(code: int):
    """Translates a given tflite code to a string.

    Args:
        code (int): Tflite Builtin operator code

    Returns:
        str: Code as as string, a "?" if it was not found
    """
    return _layers_lut.get(code, "?")
