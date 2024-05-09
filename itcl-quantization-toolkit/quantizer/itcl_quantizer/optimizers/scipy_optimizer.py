from typing import Callable

import numpy as np
from itcl_quantizer.optimizers.IOptimizer import IOptimizer
from itcl_quantizer.quantizer.distributions.distribution import Distribution
from scipy.optimize import minimize
from itcl_quantization import Quantization


class ScipyOptimizer(IOptimizer):
    def __init__(
        self,
        Q: Quantization,
        noise_metric: Callable[[np.ndarray, np.ndarray], float],
        scipy_optimizer: str,
        verbose: bool = False,
    ):
        self.__noise_metric = noise_metric
        self.__scipy_optimizer = scipy_optimizer
        self.__Q = Q
        self.__verbose = verbose

    def trim(
        self,
        distribution: Distribution,
    ):
        edges = distribution.get_bin_edges()
        Q = self.__Q
        min, max = edges[0], edges[-1]

        def minimize_noise(x):

            if x[1] - x[0] < 1 or x[0] < min or max < x[1]:
                return 999999999

            trimmed = distribution[x[0] : x[1]]

            s, zp = trimmed.quantize()
            quant_arr = Q.quantize(distribution._sorted_a, zp, s)
            dequant_arr = Q.dequantize(quant_arr, zp, s)
            trimmed_quant = Distribution(dequant_arr, bins=edges)

            res = distribution.compare(trimmed_quant, self.__noise_metric)

            return res

        res = minimize(
            minimize_noise,
            (min, max),
            bounds=[(min, max), (min, max)],
            method=self.__scipy_optimizer,
            # tol=0.1,
        )

        lower, upper = res.x
        if self.__verbose:
            print(res)
            print(f"Original range: {min}, {max}")
            print(f"Optimized range: {lower}, {upper}")

        return distribution[lower:upper]
