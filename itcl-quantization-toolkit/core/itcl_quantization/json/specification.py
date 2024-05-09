from __future__ import annotations
from typing_extensions import TypedDict # 3.7 backwards comp. 
from typing import Any, Dict, List, Optional, Tuple, Union

TensorW = List[Union[float,int, List[Any], Any]]

class ReducedLutSide(TypedDict):
    """Stores a reduced LUT side in json format
    """
    shift: int
    tuples_len: int
    tuples: List[Tuple[int, int]]
class ReducedLUT(TypedDict):
    """Stores a reduced LUT in json format
    """
    reduced_lut_len: int
    reduced_lut: List[int]
    head: ReducedLutSide
    tail: ReducedLutSide
    depth: int
    min_reduce: int
    asymmetric: bool
    description: str

class LUT(TypedDict):
    """Look Up Table
    """
    LUT: List[int]
    offset: int
    reduced_LUT: Optional[ReducedLUT]

class Tensor(TypedDict):
    """Stores a Tensor (An array or matrix of values)
    """
    tensor: Optional[TensorW]
    shape: List[int]
    dtype: str
    name: Optional[str]

class Node(TypedDict):
    """Stores a node of a Layer / Operator

    Args:
        TypedDict (_type_): _description_
    """
    tensor: Optional[Tensor]
    zero_point: Optional[int | List[int]]
    scale: Optional[float | List[float]]
    LUT: Optional[LUT]

class Attribute(TypedDict):
    """Stores the attributes of a Layer / Operator
    """
    value: List[Any]
    dtype: str

class Operator(TypedDict):
    """
        Stores an Operator (A layer of a network)
        Has input and output Nodes
    """
    op_type: str
    name: str
    inputs: List[Node]
    outputs: List[Node]
    description: Optional[str]
    attributes: Optional[Dict[str, Attribute]]

class JsonNetwork(TypedDict):
    """Stores a Json Network

    Args:
        TypedDict (_type_): _description_
    """
    ir_version: str
    engine: str
    generated_at: Union[None, str]
    operators: int
    graph: List[Operator]
    input: List[Node]
    output: List[Node]

