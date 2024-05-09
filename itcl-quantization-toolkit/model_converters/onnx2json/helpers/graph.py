from typing import List
from itcl_quantization.json.specification import Operator
from helpers.buildOperator import BuildOperatorCandidate, buildOperator
from util.settings import settings

ADVANCED_OPERATORS = {
    "TanhLUT": ["DequantizeLinear", "Tanh", "QuantizeLinear"],
}

ADVANCED_OPERATORS_SORTED = sorted(
    ADVANCED_OPERATORS.items(), key=lambda x: len(x[1]), reverse=True
)


class OnnxGraph:
    """Class that represents the operator graph
    """
    def __init__(self, model):
        """
        It takes a model as input and initializes the current operator index to 0, the model to the model
        passed in, and the operators to the nodes in the model's graph.
        
        :param model: The model that we want to optimize
        """
        self.__current_op_idx = 0
        self.__model = model
        self.__operators = model.graph.node
        
    def build_graph(self) -> List[Operator]:
        """
        It takes a list of operators and returns a list of operators
        :return: A list of operators
        """

        operators = []

        while self.__current_op_idx < len(self.__operators):
            next_operator_candidate = self.__get_next_ops()
            print(next_operator_candidate.op_type)
            operators.append(buildOperator(next_operator_candidate, self.__model))

        return operators

    def __get_next_ops(

        self,
    ) -> BuildOperatorCandidate:
        """

        Next Ops, gets the next operator candidate

        A candidate can be an ADVANCE_OPERATOR or a normal operator

        An ADVANCE_OPERATOR is an operator that is build by more than one operator. For example TANH_LUT.

        The function takes a list of operators and returns the next operator to be built
        :return: BuildOperatorCandidate Instance. 
        """
        remaining_op = self.__operators[self.__current_op_idx :]

        op_type_seq = ".".join(
            [op.op_type for op in remaining_op]
        )  # op_typ_seq = "QuantizeLinear.Tanh.Softmax"

        # Find the largest Advanced Operator Match
        for advanced_op_name, advanced_op in ADVANCED_OPERATORS_SORTED:
            advanced_op_regex = ".".join(advanced_op)
            # if op_type_seq starts with op_type_seq_regex
            if op_type_seq.startswith(advanced_op_regex):

                candidate = BuildOperatorCandidate(
                    advanced_op_name, remaining_op[: len(advanced_op)]
                )
                self.__current_op_idx += len(advanced_op)
                return candidate

        self.__current_op_idx += 1
        return BuildOperatorCandidate(remaining_op[0].op_type, remaining_op[0])
