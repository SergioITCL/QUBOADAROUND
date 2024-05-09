import numpy as np
from itcl_quantizer.optimizers.IOptimizer import IOptimizer
from itcl_quantizer.quantizer.distributions.distribution import Distribution


class FixedPercentOptimizer(IOptimizer):
    def __init__(self, percent: float):
        self.__percent = percent

    def trim(self, distribution: Distribution) -> Distribution:

        bins = distribution.get_bin_edges()
        fp_array = distribution._sorted_a
        low_percent = np.percentile(fp_array, self.__percent)
        high_percent = np.percentile(fp_array, 1 - self.__percent)

        clipped_array = fp_array.clip(low_percent, high_percent)

        return Distribution(clipped_array, bins=bins)
