from typing import Iterable

from itcl_quantizer.tensor_extractor.operator import NodeTensorBase


class NodeBundler:
    """Class that bundles nodes together."""

    def __init__(self):

        # TODO: Make this generic?
        self._nodes: list[NodeTensorBase] = []

    def add_nodes(self, *nodes: NodeTensorBase):
        self._nodes.extend(nodes)

    @property
    def nodes(self) -> Iterable[NodeTensorBase]:
        return self._nodes
