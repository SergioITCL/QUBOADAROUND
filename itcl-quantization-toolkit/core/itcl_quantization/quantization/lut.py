import math
from typing import List, Tuple
import numpy as np
from itcl_quantization.quantization.operators import Quantization
from itcl_quantization.json.specification import ReducedLUT as ReducedLUT_JSON


class ReducedLUT:
    """
    Reduced Look Up Table

    This class instantiates a reduced look up table that has the head and tail pruned.
    The pruned sides are replaced by intervals.

    This class has a serializable representation as JSON.
    """

    def __init__(self, lut: List[int], depth: int, min_reduce: int, asymmetric=False):
        """Reduce LUT Constructor + Reducer

        Given a Look Up Table, returns a reduced one which has the head and tail pruned.
        The pruned sides are replaced with intervals.

        Args:
            lut (List[int]): Original LUT
            depth (int): Number of different values to be pruned from each side
            min_reduce (int): Minimun number of values to be reduced
            asymmetric (bool, optional): If the number of intervals on each side can be different. Defaults to False.
        """
        self.lut = list(lut)  # Clone the LUT
        self.depth = depth
        self.asymmetric = asymmetric
        self.min_reduce = min_reduce

        # Head:
        self.left_shift, self.left_tuples = self.__reduce_lut_from_left(
            lut, reverse=False
        )

        # Tail
        self.right_shift, self.right_tuples = self.__reduce_lut_from_left(
            lut[self.left_shift :] if self.asymmetric else lut, reverse=True
        )

        # Reduce the original LUT
        r_shift_slice_idx = -self.right_shift if self.right_shift else None
        self.reduced_lut = lut[self.left_shift : r_shift_slice_idx]

    def __getitem__(self, index: int) -> int:
        """Get the value of the LUT at the given index

        Args:
            index (int): An integer index

        Raises:
            IndexError: The index is out of bounds

        Returns:
            int: The value of the LUT at the given index
        """
        if index < self.left_shift:
            for t in self.left_tuples:
                if index < t[0]:
                    return t[1]

        lut_idx = index - self.left_shift

        if lut_idx < len(self.reduced_lut):
            return self.reduced_lut[lut_idx]

        for t in self.right_tuples:
            if index >= t[0]:
                return t[1]

        raise IndexError(f"LUT Index {index} out of range")

    def __iter__(self):
        """Iterator for the LUT"""
        for i in range(len(self.lut)):
            yield self[i]

    def __reduce_lut_from_left(
        self, lut: List[int], reverse: bool = False
    ) -> Tuple[int, List[Tuple[int, int]]]:
        """Reduce LUT from left

        Args:
            lut (List[int]): _description_
            reverse (bool, optional): False: reduce the head, True: reduce the tail. Defaults to False.

        Returns:
            Tuple[int, List[Tuple[int, int]]]: The shift (number of elements pruned) and tuples with the offsets and values
        """

        if len(lut) == 0:
            return 0, []

        lut = list(lut)  # Clone the LUT

        if reverse:
            lut.reverse()

        counter: List[Tuple[int, int]] = [
            (lut[0], 0)
        ]  # K: lut/fn value, V: n repetitions

        # Count the number of repetitions of each value in the LUT up to DEPTH different numbers
        for element in lut:
            current_val, current_count = counter[-1]

            if element != current_val:
                # If we have reached the max depth:
                if len(counter) == self.depth:
                    break

                else:  # Add the current value to the counter
                    counter.append((element, 0))
                    current_val = element
                    current_count = 0

            # Increment the counter of the current val
            counter[-1] = (current_val, current_count + 1)

        # Get the number of repetitions that will be pruned
        shift = sum([c[1] for c in counter])

        # Create a list of [offset, value] of the repetitions that will be pruned
        intervals: List[Tuple[int, int]] = []  # List of [end_shift, value/fn output]

        for key, value in counter:
            intervals.append(
                (value + intervals[-1][0] if len(intervals) else value, key)
            )

        # Adjust the individual intervals if we are working with the tail
        if reverse:
            # intervals.reverse()
            for i in range(len(intervals)):
                intervals[i] = (len(lut) - intervals[i][0], intervals[i][1])

        return shift, intervals

    def serialize(self) -> ReducedLUT_JSON:
        """Serialize the reduced LUT as a json serializable dictionary indexed by a named string"""
        return {
            "reduced_lut_len": len(self.reduced_lut),
            "reduced_lut": self.reduced_lut,
            "head": {
                "shift": self.left_shift,
                "tuples_len": len(self.left_tuples),
                "tuples": self.left_tuples,
            },
            "tail": {
                "shift": self.right_shift,
                "tuples_len": len(self.right_tuples),
                "tuples": self.right_tuples,
            },
            "depth": self.depth,
            "min_reduce": self.min_reduce,
            "asymmetric": self.asymmetric,
            "description": f"Reduced LUT of depth {self.depth} with {self.min_reduce} minimum reduce",
        }

    @staticmethod
    def deserialize(json_lut: ReducedLUT_JSON) -> "ReducedLUT":
        """Deserialize a ReducedLUT from a json serializable dictionary indexed by a named string"""
        reduced_lut = ReducedLUT(
            [], json_lut["depth"], json_lut["min_reduce"], json_lut["asymmetric"]
        )

        reduced_lut.left_shift = json_lut["head"]["shift"]
        reduced_lut.left_tuples = json_lut["head"]["tuples"]
        reduced_lut.right_shift = json_lut["tail"]["shift"]
        reduced_lut.right_tuples = json_lut["tail"]["tuples"]
        reduced_lut.reduced_lut = json_lut["reduced_lut"]

        return reduced_lut


if __name__ == "__main__":
    q = Quantization(np.uint8)
    lut = q.create_LUT(math.tanh, 0.058881718665361404, 151, 0.00784310046583414, 128)
    lut = q.create_LUT(math.tanh, 1, 0, 1, 0)
    lut = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    print(lut)
    # print(lut)
    # Reduce the LUT
    r_lut = ReducedLUT(lut, 5, 0)
    serialized = r_lut.serialize()
    print("--------------------")
    for k, v in serialized.items():
        print(f"{k}: {v}")

    for i, v in enumerate(lut):
        assert v == r_lut[i], f"{i}: {v} != {r_lut[i]}"
