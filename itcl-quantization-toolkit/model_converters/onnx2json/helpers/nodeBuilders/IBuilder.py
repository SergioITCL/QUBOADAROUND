from typing import List, Optional, Tuple

from itcl_quantization.json.specification import Node


class IBuilder:
    def build(self, node, graph) -> Tuple[List[Node], List[Node], Optional[str]]:
        raise NotImplemented
