from typing import Any, Callable, Generic, TypeVar
from typing_extensions import Self
from itcl_quantization.json.specification import Operator as JsonOperator, Attribute
import numpy as np
from itcl_quantizer.interfaces.serializable import ISerializable
from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase

"""Generic Input/Output nodes. They must be or inherit from NodeTensorBase"""
T = TypeVar("T", bound=NodeTensorBase)
E = TypeVar("E", bound=NodeTensorBase)


class Operator(Generic[T, E], ISerializable):
    """
    An operator is a class that contains sequential input and output nodes.

    T is the input nodes type
    E is the output nodes type
    """

    _description: str | None = None
    """Operator Description. Updated with set_description()
    """

    def __init__(
        self,
        op_type: str,
        name: str,
        inputs: list[T],
        outputs: list[E],
        layer: Callable[[np.ndarray], np.ndarray] | None,
        attributes: dict[str, Attribute] = {}
    ) -> None:
        """Operator Constructor

        Args:
            op_type (str): Operator Datatype
            name (str): Operator Name
            inputs (list[T]): List of Nodes
            outputs (list[E]): List of Nodes
            layer (Callable[[np.ndarray], np.ndarray] | None): Original Keras Layer that can be called to infer a float value.
        """
        self._op_type = op_type
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.layer = layer
        self.attributes = attributes

    def set_description(self, desc: str) -> Self:
        """Updates the _description attribute

        Args:
            desc (str): An string that contains the description

        Returns:
            Operator: The self operator
        """
        self._description = desc
        return self

    def as_json(self) -> JsonOperator:
        return {
            "op_type": self._op_type,
            "name": self.name,
            "description": self._description or self.name,
            "inputs": [t.as_json() for t in self.inputs],
            "outputs": [t.as_json() for t in self.outputs],
            "attributes": self.attributes,
        }

    def __getitem__(self, item: str) -> Any:
        """Gets an item from the json

        Compatibility deprecated magic method.

        Args:
            item (str): json key

        Returns:
            _type_: an item from the json
        """

        return self.as_json().get(item)
