import math
from typing import List
from itcl_quantizer.optimizers.IOptimizer import IOptimizer
from itcl_quantizer.quantizer.distributions.distribution import Distribution
import numpy as np

from itcl_quantizer.quantizer.metrics.noise import kl_divergence


class KLDivergenceOptimizer(IOptimizer):
    """Nvidia's implementation of Entropy Calibration based on KL Divergence minimization

    source: https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

    """

    def __init__(self, bin_count=2048, quant_bin_count=128):
        self.__bin_count = bin_count
        self.__target_bin_count = quant_bin_count

    def trim(self, distribution: Distribution) -> Distribution:

        distribution = Distribution(distribution._sorted_a, self.__bin_count)
        divergence: List[float] = []
        p_bin = self.__bin_count

        histogram = distribution.get_histogram()
        for i in range(self.__target_bin_count, self.__bin_count):
            reference_dist = histogram[:i]
            outliers_count = np.sum(histogram[i:])
            reference_dist[i - 1] += outliers_count

            reference_dist /= np.sum(reference_dist)  # normalize

            candidate_dist = self.reduce(histogram[:i])

            candidate_dist /= np.sum(candidate_dist)  # normalize
            # assert len(candidate_dist) == i, f"{len(candidate_dist)} != {i}"
            assert len(candidate_dist) == len(
                reference_dist
            ), f"{len(candidate_dist)} != {len(reference_dist)}"
            divergence.append(kl_divergence(reference_dist, candidate_dist))

        m = np.argmin(divergence)

        threshold = (m + 0.5) * distribution.get_bin_size()

        print(threshold)

        return distribution[:threshold]

    def reduce(self, histogram: List[int]) -> List[int]:
        """
        Reduce the histogram to the target bin count.

        Source: Slide 38 (https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)

        :param histogram: The histogram to reduce.
        :return: The reduced histogram.
        """
        mask = [0 if i == 0 else 1 for i in histogram]  # mask of 1s and 0s

        bins_per_reduced_bin = math.ceil(self.__bin_count / self.__target_bin_count)

        expanded_bins = []

        for i in range(self.__target_bin_count):
            idx = i * bins_per_reduced_bin

            bin = np.array(histogram[idx : idx + bins_per_reduced_bin])

            mask_slice = np.array(mask[idx : idx + bins_per_reduced_bin])

            expanded_bin = mask_slice * (bin.sum() / np.array(mask_slice).sum())

            expanded_bins.extend(expanded_bin)

        assert len(expanded_bins) == len(
            histogram
        ), f"{len(expanded_bins)} != {self.__target_bin_count}"

        return expanded_bins
