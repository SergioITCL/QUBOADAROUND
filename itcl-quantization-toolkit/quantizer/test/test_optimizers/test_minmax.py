from itcl_quantizer.quantizer.distributions.distribution import Distribution
from itcl_quantizer.optimizers.minmax import MinMaxOptimizer


class TestMinMaxOptimizer:
    def test_min_max(self):

        data = [i for i in range(100)]

        dist = Distribution(data)

        optimizer = MinMaxOptimizer()

        optimizer.trim(dist)
        assert dist.get_min() == 0
        assert dist.get_max() == 99
