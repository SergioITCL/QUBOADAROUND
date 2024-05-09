from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from itcl_quantization import Quantization
from scipy.stats import shapiro, ks_2samp, norm


@dataclass
class Bin:
    """Dataclass container that holds a minimum and maximum value, the number of items in the bin
    and a numpy view that contains a sorted vector of the items in the bin.
    """

    min: float
    max: float
    items: int
    items_in_bin: np.ndarray


class Distribution:
    def __init__(
        self,
        a: Union[list, np.ndarray],
        bins: Union[int, List[int]] = 512,
        range=None,
        weights=None,
        include_partial_bins=True,
    ):

        self.__range = range
        self.__weights = weights

        # flatten
        self._sorted_a: np.ndarray = np.sort(np.array(a).flatten())

        # Build the initial histogram
        histogram, edges = np.histogram(
            a, bins=bins, range=range, weights=weights, density=False
        )
        self._bin_edges: List[int] = edges.tolist()
        self._bin_size: int = edges[1] - edges[0]
        self._histogram: List[int] = histogram.tolist()
        self._include_partial_bins = include_partial_bins

    def __update(self, a: Union[list, np.ndarray]):
        """
        Args:
            a (Union[list, np.ndarray]): _description_
        """
        # TODO: Update the sorted_a with a param
        raise NotImplemented("Check the TODO")

        min = np.min(a)
        max = np.max(a)

        # If the data overflows the distribution, extend the distribution
        if min < self._min_bin_val:
            diff = int(abs(self._min_bin_val - min))
            diff = int(int(abs(self._min_bin_val - min) + 1.5) / self._bin_size)

            extra_left_bins = [
                self._min_bin_val - self._bin_size * i for i in range(1, diff)
            ]
            extra_left_bins.reverse()
            # Extend the left side with the new bins and add empty data to the histogram
            self._bin_edges = extra_left_bins + self._bin_edges
            self._histogram = [0] * len(extra_left_bins) + self._histogram
            self._min_bin_val = min

        if max > self._max_bin_val:
            diff = int(abs(self._max_bin_val - max))
            diff = int(int(abs(self._max_bin_val - max) + 1.5) / self._bin_size)

            extra_right_bins = [
                self._max_bin_val + self._bin_size * i for i in range(1, diff)
            ]

            # Extend the right side with the new bins and add empty data to the histogram
            self._bin_edges = self._bin_edges + extra_right_bins
            self._histogram = self._histogram + [0] * len(extra_right_bins)
            self._max_bin_val = max

        histogram, _ = np.histogram(
            a,
            bins=np.array(self._bin_edges),
            range=self.__range,
            weights=self.__weights,
        )

        self._histogram = (np.array(self._histogram) + histogram).tolist()

    def normalize(self) -> np.ndarray:
        """
        The function `normalize` returns a normalized version of the histogram

        Returns:
          The normalized histogram.
        """
        return np.array(self._histogram) / np.sum(self._histogram)

    def get_histogram(self) -> List[int]:
        """
        Returns a list of integers representing the histogram of the image

        Returns:
          A list of integers.
        """
        return self._histogram

    def get_bin_edges(self) -> List[int]:
        """
        This function returns a list of integers that represent the bin edges

        Returns:
          The bin edges
        """
        return self._bin_edges

    def quantize(
        self,
        q: Quantization = Quantization("int8"),
        force_zp: int | None = None,
        force_s: float | None = None,
        symmetric: bool = False,
    ) -> Tuple[float, int]:
        """
        The function takes the minimum and maximum values of the tensor and returns the scale and zero
        point for quantization process

        Args:
          bits (int): the number of bits to use for quantization. Defaults to 8
          signed (boolean): If True, the quantized values will be signed. If False, the quantized values
        will be unsigned. Defaults to True

        Returns:
          The scale and zero point of the quantized tensor.
        """
        min_ = min(self.get_min(), 0)
        max_ = max(self.get_max(), 0)
        
        q_min = q.min_value()           
        q_max = q.max_value()
        if symmetric:
            q_min = -q_max

        scale = (max_ - min_) / (q_max - q_min)
        zero_point = round((min_ * q_max - max_ * q_min) / (min_ - max_))
        if force_zp is not None:
            zero_point = force_zp
            scale = max(abs(min_), abs(max_))
            
            if symmetric:
                scale /= q_max
            else:
                scale /= abs(q_min)
        if force_s is not None:
            scale = force_s

        if force_zp is not None and force_s is not None:
            scale = force_s
            zero_point = force_zp

        if scale == 0.0:
            scale = 1.0
            zero_point = 0

        return scale, zero_point

    def compare(
        self,
        distribution: "Distribution",
        compare_fn: Callable[[np.ndarray, np.ndarray], float],
        norm: bool=True,
    ) -> float:
        """
        It takes a distribution and a function that compares two histograms, and returns the result of
        applying that function to the histograms of the two distributions

        Args:
          distribution ("Distribution"): The distribution to compare to.
          compare_fn (Callable[[np.ndarray, np.ndarray], float]): a function that takes two histograms and
        returns a float.

        Returns:
          the result of the comparison function.
        """
        if norm:
            my_hist = self.normalize()
            other_hist = distribution.normalize()
        else:
            my_hist = np.array(self.get_histogram())
            other_hist = np.array(distribution.get_histogram())

        return compare_fn(my_hist, other_hist)

    def get_bin(self, idx: int) -> Bin:
        """
        Returns a `Bin` object, which is a named tuple with three fields: `min` value of the bin,
        `max` value of the bin, and `count` items inside the bin.

        :param idx: the index of the bin to return, negative indexing is allowed.
        """
        # Allow negative indexes
        if idx < 0:
            idx = len(self) + idx

        lower = self._bin_edges[idx]
        upper = self._bin_edges[idx + 1]

        bin: int = self._histogram[idx]
        if bin < 2:
            return Bin(lower, upper, bin, np.array([]))

        # Get a view of the sorted distribution array
        bin_content_view = self.__slice_distribution(lower, None)[:bin]

        assert (
            len(bin_content_view) == bin
        ), f"Bin Content lenght is {len(bin_content_view)} but should be {bin}"

        return Bin(bin_content_view[0], bin_content_view[-1], bin, bin_content_view)

    def __len__(self) -> int:
        """
        The `__len__` function returns the number of bins in the histogram

        Returns:
          The number of bins in the histogram.
        """

        return len(self._bin_edges) - 1

    def __slice_distribution(
        self, lower: Optional[float], upper: Optional[float]
    ) -> np.ndarray:
        """
        Given a lower and upper bound, return the sorted array of values between the bounds

        Args:
          lower (float): the lower bound of the slice
          upper (float): float

        Returns:
          The slice of the distribution between the lower and upper bounds.
        """
        a = self._sorted_a

        if lower is not None:
            min_idx = a.searchsorted(lower, side="left")
        else:
            min_idx = 0

        if upper is not None:
            max_idx = a.searchsorted(upper, side="right")
        else:
            max_idx = len(a)
        return a[min_idx:max_idx]

    def __clone(self, key: slice) -> "Distribution":
        """
        The function takes a slice of the distribution and returns a new distribution with the same
        parameters as the original distribution, but with the histogram and bin edges sliced to the slice

        Args:
          key (slice): slice

        Returns:
          A new distribution object with the same parameters as the original distribution object, but with
        the histogram and bin edges sliced.
        """

        # Get the min and max bin edges
        min_bin_idx = 0
        max_bin_idx = len(self._histogram)

        for i, edge in enumerate(self._bin_edges):
            if key.start is not None and edge <= key.start:
                min_bin_idx = i

            if key.stop is not None and edge <= key.stop:
                max_bin_idx = i

        if self._include_partial_bins:
            max_bin_idx = min(max_bin_idx + 1, len(self._histogram))
        else:
            min_bin_idx = min(min_bin_idx + 1, max_bin_idx - 1)

        # Update the current distribution parameters as is cheaper than creating a new distribution

        deep_distr = deepcopy(self)

        # Reduced the bin edges / number of bins
        deep_distr._bin_edges = self._bin_edges[
            min_bin_idx : min(max_bin_idx + 1, len(self._bin_edges))
        ]  # The edges have one extra element, so we need to get the last element

        # Slice the Histogram
        deep_distr._histogram = self._histogram[min_bin_idx:max_bin_idx]

        min_bin_val, max_bin_val = (
            self._bin_edges[min_bin_idx],
            self._bin_edges[max_bin_idx],
        )

        view_slice = self.__slice_distribution(min_bin_val, max_bin_val)

        deep_distr._sorted_a = view_slice

        return deep_distr

    def __getitem__(self, key) -> "Distribution":
        """
        It takes a slice of the histogram and returns a new histogram with the same properties as the
        original, but with the slice applied

        :param key: int or slice(float, float, any (unused))
        :return: The histogram of the distribution
        """
        if isinstance(key, slice):

            if key.start is None:
                key = slice(self.get_min(), key.stop, key.step)

            if key.stop is None:

                key = slice(key.start, self.get_max(), None)

            # If the slice is smaller than a bin, create a new histogram
            if abs(key.start - key.stop) < self.get_bin_size():
                return Distribution(
                    self.__slice_distribution(key.start, key.stop), bins=1
                )

            return self.__clone(key)

        bin = self.get_bin(key)
        return self[bin.min : bin.max]

    def get_min(self):
        """
        It returns the minimum value in the bins.
        :return: The minimum value in the stack.
        """
        return self.get_bin(0).min

    def get_max(self):
        """
        It returns the maximum value of the bins.
        :return: The max value of the list
        """
        return self.get_bin(-1).max

    def get_bin_size(self):
        """
        It returns the size of the bins.
        """
        return abs(self._bin_edges[1] - self._bin_edges[0])

    def is_gaussian_shapiro(self, threshold: float = 0.05) -> bool:
        """
        It checks if the distribution is gaussian.
        """
        print("shapiro", shapiro(self._sorted_a).pvalue)
        return shapiro(self._sorted_a).pvalue > threshold

    def is_gaussian(self, threshold: float = 0.05) -> bool:
        fp_arr = self._sorted_a

        if len(fp_arr) < 512:
            return False

        mean = np.mean(fp_arr)
        std = np.std(fp_arr)
        normal_dist = np.random.normal(mean, std, 4096)

        normal_norm = (normal_dist - mean) / std

        fp_norm = (fp_arr - mean) / std
        pvalue = ks_2samp(fp_norm, normal_norm).pvalue
        print(pvalue)

        from matplotlib import pyplot as plt

        if pvalue > threshold and False:
            # clear the plt buffer
            plt.clf()
            _, bins, _ = plt.hist(
                fp_norm, bins=256, density=True, alpha=0.5, label="data"
            )
            _, bins, _ = plt.hist(
                normal_norm, density=True, alpha=0.5, label="normal", bins=bins
            )
            plt.legend()
            plt.show()

        return pvalue > threshold

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__range}, {self.__weights}),\
             min: {self.get_min()}, max: {self.get_max()}, bins: {len(self._bin_edges) -1}"
