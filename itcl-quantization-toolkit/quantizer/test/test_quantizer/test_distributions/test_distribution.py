from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
from itcl_quantizer.quantizer.distributions.distribution import Distribution
from itcl_quantization import Quantization
from scipy.stats import poisson, expon
from itcl_quantizer.quantizer.metrics.noise import kl_divergence, mse


class TestDistribution:
    def create_normal_distribution(self):
        dist: np.ndarray = np.random.normal(loc=0, scale=1, size=20000)
        return dist

    def add_outilers_to_distribution(self, dist: np.ndarray):
        dist[0] = -10
        dist[-1] = 10
        return dist

    def __test_bins(self, distribution: Distribution):
        for i in range(len(distribution.get_bin_edges()) - 1):
            print("Testing idx ", i)
            bin = distribution.get_bin(i)

    def get_minus_5_to_5(self):
        return np.array(
            [
                -5,
                -4.5,
                -4,
                -3.5,
                -3,
                -2.5,
                -2,
                -1.5,
                -1,
                -0.5,
                0.00,
                0.49,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
            ]
        )

    def test_distribtion_init(self):
        dist = self.create_normal_distribution()

        distribution = Distribution(dist, bins=10)

        np.testing.assert_almost_equal(distribution.get_min(), dist.min())
        np.testing.assert_almost_equal(distribution.get_max(), dist.max())

        self.__test_bins(distribution)

    def test_distribution(self):
        dist = self.create_normal_distribution()
        dist = self.add_outilers_to_distribution(dist)

        distribution = Distribution(dist)

        print(distribution.get_bin_edges())

        print(distribution.get_histogram())
        sum_histogram = np.sum(distribution.get_histogram())

        sliced = distribution[-2:2]

        sum_sliced = np.sum(sliced.get_histogram())

        assert sum_sliced < sum_histogram

        self.__test_bins(sliced)

    def test_slice_smaller_than_bin(self):
        dist = self.create_normal_distribution()
        dist = self.add_outilers_to_distribution(dist)

        START = 0.15
        END = 0.18

        distribution = Distribution(dist, bins=10)[START:END]

        assert len(distribution.get_bin_edges()) == 2
        assert len(distribution.get_histogram()) == 1

        bin = distribution.get_bin(0)

        assert bin.min >= START
        assert bin.max <= END

        self.__test_bins(distribution)

    def test_slice_bigger_than_bin(self):

        dist = self.get_minus_5_to_5()

        START = -0.41
        END = 0.51

        distribution = Distribution(dist, bins=100)

        np.testing.assert_almost_equal(distribution.get_bin_size(), 0.1)

        assert len(distribution.get_bin_edges()) == 101
        assert len(distribution.get_histogram()) == 100

        distribution = distribution[START:END]

        assert len(distribution.get_bin_edges()) == 12
        assert len(distribution.get_histogram()) == 11

        first_bin = distribution.get_bin(0)
        assert first_bin.min <= START <= first_bin.max

        last_bin = distribution.get_bin(-1)
        assert last_bin.min <= END

        distribution = distribution[START:END]

        self.__test_bins(distribution)

    def test_slice_with_nones(self):
        dist = np.array([i for i in range(10)])

        distribution = Distribution(dist, bins=9)
        assert distribution.get_bin_size() == 1
        print(distribution.get_bin_edges())
        sliced = distribution[5:]
        print(sliced.get_bin_edges())
        assert len(sliced.get_bin_edges()) == 5
        assert len(sliced._histogram) == 4

        assert sliced.get_bin(-1).max == 9

        # Do nothing
        sliced = sliced[:9]
        assert len(sliced.get_bin_edges()) == 5
        assert len(sliced._histogram) == 4
        assert sliced.get_bin(-1).max == 9

        sliced = sliced[:8.5]
        print(sliced)

    def test_histogram_limits(self):
        dist = self.get_minus_5_to_5()

        distribution = Distribution(dist, bins=10)

        self.__test_bins(distribution)

        assert distribution.get_bin_size() == 1
        assert distribution.get_bin_edges()[0] == -5
        assert distribution.get_bin_edges()[-1] == 5

        assert distribution.get_min() == -5
        assert distribution.get_max() == 5

    def test_without_incomplete_bins(self):
        dist = self.get_minus_5_to_5()

        distribution = Distribution(dist, bins=10, include_partial_bins=False)

        assert len(distribution.get_bin_edges()) == 11
        assert len(distribution.get_histogram()) == 10

        sliced = distribution[1.8:3.2]
        print(sliced.get_bin_edges())
        assert len(sliced.get_bin_edges()) == 2  # [2, 3]
        print(sliced.get_bin(0))
        print(sliced.get_bin(-1))
        assert sliced.get_bin(0).min == 2
        assert sliced.get_bin(-1).max == 2.5

        self.__test_bins(sliced)


class TestDistributionComparison:
    Q = Quantization(np.int8)

    def add_outilers_to_distribution(self, dist: np.ndarray):
        dist[0] = -10
        dist[-1] = 10
        return dist

    def get_input_distribution(self):
        # return normal distribution
        dist = np.random.normal(loc=0, scale=1, size=200000)

        return dist

    def get_base_quant(self, get: Callable[[], np.ndarray], bins=40):
        dist = get()

        distribution = Distribution(dist, bins)

        scale, zp = distribution.quantize()

        quantized_dist = self.Q.quantize(dist, zp, scale)
        dequantized = self.Q.dequantize(quantized_dist, zp, scale)
        quant_distribution = Distribution(dequantized, bins)

        return distribution, quant_distribution

    def test_kldiv(self):

        distribution, quant_distribution = self.get_base_quant(
            self.get_input_distribution, 32
        )
        div = distribution.compare(quant_distribution, kl_divergence)

        distribution, quant_distribution = self.get_base_quant(
            lambda: self.add_outilers_to_distribution(self.get_input_distribution()), 32
        )

        noisy_div = distribution.compare(quant_distribution, kl_divergence)

        assert div < noisy_div

    def test_mse(self):
        distribution, quant_distribution = self.get_base_quant(
            self.get_input_distribution, 32
        )
        div = distribution.compare(quant_distribution, mse)

        distribution, quant_distribution = self.get_base_quant(
            lambda: self.add_outilers_to_distribution(self.get_input_distribution()), 32
        )

        noisy_div = distribution.compare(quant_distribution, mse)

        assert div < noisy_div

    def demo_exponential_distribution_compare(self):
        import seaborn as sns

        noise_fn = mse

        data_expon = expon.rvs(scale=2, loc=0, size=25000)

        distribution, quant_distribution = self.get_base_quant(lambda: data_expon, 1024)
        data_quant = quant_distribution._sorted_a
        div = distribution.compare(quant_distribution, mse)
        print(div)

        if True:
            fig, axs = plt.subplots(ncols=2)
            # add title
            fig.suptitle(f"KL-Divergence: {div}")
            axs[0].title.set_text("Original Distribution")  # type: ignore
            axs[1].title.set_text("Quantized Distribution")  # type: ignore
            sns.distplot(
                data_expon,
                color="g",
                bins=1024,
                kde=False,
                hist_kws={"linewidth": 15, "alpha": 1},
                ax=axs[0],  # type: ignore
            )
            sns.distplot(
                data_quant,
                bins=256,
                kde=False,
                color="skyblue",
                hist_kws={"linewidth": 15, "alpha": 1},
                ax=axs[1],  # type: ignore
            )
            # ax.set(xlabel="Poisson Distribution", ylabel="Frequency")

            plt.show()

        def trim(limit: float, plot=False):
            trimmed = distribution[0:limit]

            s, zp = trimmed.quantize()

            quantized_trimmed_a = self.Q.quantize(trimmed._sorted_a, zp, s)
            dequantized_trimmed_a = self.Q.dequantize(quantized_trimmed_a, zp, s)
            quant_trimmed = Distribution(
                dequantized_trimmed_a,
                bins=distribution.get_bin_edges(),
            )

            div = distribution.compare(quant_trimmed, mse)

            title = f"KL Divergence: {div}"
            print(title)

            if plot:
                fig, axs = plt.subplots(ncols=2)

                # add title to plot
                axs[0].set_title("Trimmed Distribution")  # type: ignore
                axs[1].set_title("Trimmed Quantized Distribution")  # type: ignore

                sns.distplot(
                    distribution._sorted_a,
                    color="g",
                    bins=1024,
                    kde=True,
                    hist_kws={"linewidth": 15, "alpha": 1},
                    ax=axs[0],  # type: ignore
                    kde_kws={"clip": None},
                )
                sns.distplot(
                    quant_trimmed._sorted_a,
                    bins=256,
                    kde=True,
                    color="skyblue",
                    hist_kws={"linewidth": 15, "alpha": 1},
                    ax=axs[1],  # type: ignore
                    kde_kws={"clip": None},
                )
                # ax.set(xlabel="Poisson Distribution", ylabel="Frequency")
                fig.suptitle(title)
                plt.show()
            return div

        min = 100
        limit = 0
        for i in range(40, 80):
            l = i / 10
            print(l)
            div = trim(l)
            print(f"{l} {div}")
            if div < min:
                limit = l
                min = div
        print(f"Winner {limit}")
        trim(limit, True)

        assert False

    def test_quantize_negative_gt_positve(self):

        data = [i for i in range(10)]

        data.append(-11)

        dist = Distribution(data, bins=3)

        scale, zp = dist.quantize(force_zp=0, symmetric=True)

        assert zp == 0
        assert scale == float(-11 / -127)

    def test_quantize_positive_gt_negative(self):

        data = [i for i in range(10)]

        data.append(-1)

        dist = Distribution(data, bins=3)

        scale, zp = dist.quantize(force_zp=0, symmetric=True)

        assert zp == 0
        assert scale == float(9 / 127)

    def test_equal_positive_negative(self):
        data = [i for i in range(10)]
        data.append(-9)

        dist = Distribution(data, bins=3)
        scale, zp = dist.quantize(force_zp=0, symmetric=True)

        assert zp == 0
        assert scale == float(9 / 127)

    def test_quantize(self):
        data = [i for i in range(10)]
        data.append(-99)

        dist = Distribution(data, bins=3)

        scale, zp = dist.quantize(symmetric=True)

        assert zp == 106
        np.testing.assert_almost_equal(scale, 0.4251968503937008)
