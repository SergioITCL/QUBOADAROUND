from matplotlib.pyplot import legend
from numpy import percentile
from itcl_quantizer.optimizers.IOptimizer import IOptimizer
from itcl_quantizer.optimizers.fixed_percent import FixedPercentOptimizer
from itcl_quantizer.optimizers.kl_div_optimizer import KLDivergenceOptimizer
from itcl_quantizer.optimizers.minmax import MinMaxOptimizer
from itcl_quantizer.quantizer.distributions.distribution import Distribution
from itcl_quantization import Quantization
import numpy as np


class DebugPlotOptimizer(MinMaxOptimizer):
    def __init__(self, dtype: str, title: str = "", force_zp: int | None = None):
        self.q = Quantization(dtype)
        self.title = title
        self.fzp = force_zp

    def trim(self, distribution: Distribution) -> Distribution:
        # return distribution
        fp_arr = distribution._sorted_a
        scale, zp = distribution.quantize(self.q, force_zp=self.fzp)

        quantized = self.q.quantize(fp_arr, zp, scale)
        quantized = self.q.dequantize(quantized, zp, scale)

        if False:
            """
            Intercuartil
            """
            q25, q75 = percentile(fp_arr, 5), percentile(fp_arr, 95)
            iqr = q75 - q25
            print(q25)
            print(iqr)
            cut_off = iqr * 1.5
            lower, upper = q25 - cut_off, q75 + cut_off
            print(f"Lower: {lower}, upper: {upper}")

        if False and distribution.is_gaussian():
            mean, std = np.mean(fp_arr), np.std(fp_arr)
            cut_off = std * 3
            lower, upper = mean - cut_off, mean + cut_off
            print(f"lower: {lower}, upper: {upper}")
            fp_arr = fp_arr[(fp_arr > lower) & (fp_arr < upper)]

        import seaborn as sns
        from matplotlib import pyplot as plt

        if distribution.is_gaussian() and False:
            _, bins, _ = plt.hist(
                quantized,
                bins=256,
                alpha=0.5,
                label="Quant -> Dequant",
            )
            plt.hist(fp_arr, bins=bins, alpha=0.5, label="FP32")
            plt.show()

        trimmed = FixedPercentOptimizer(0.01).trim(distribution)
        trimmed = KLDivergenceOptimizer().trim(distribution)
        plt.clf()

        _, bins, _ = plt.hist(fp_arr, alpha=0.2, label="FP32", bins=256)
        plt.hist(trimmed._sorted_a, alpha=0.2, label="Trimmed", bins=bins)
        plt.legend()
        plt.show()
        return Distribution(fp_arr)
        trim = []  # input(">")
        if len(trim) > 0:
            print("Trimming")
            mean = np.mean(fp_arr)
            std = np.std(fp_arr)

            cut_off = std * 3
            lower, upper = mean - cut_off, mean + cut_off
            print(lower)
            print(upper)
            outliers = [x for x in fp_arr if x < lower or x > upper]
            outliers_removed = [x for x in fp_arr if x >= lower and x <= upper]
            plt.hist(fp_arr, bins=256, alpha=0.5, label="FP32")
            plt.hist(outliers_removed, bins=256, alpha=0.5, label="FP32")
            plt.show()
            return Distribution(outliers_removed)
        return super().trim(distribution)
