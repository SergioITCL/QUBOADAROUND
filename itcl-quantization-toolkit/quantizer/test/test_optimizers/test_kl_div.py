from itcl_quantizer.optimizers.kl_div_optimizer import KLDivergenceOptimizer
from itcl_quantizer.quantizer.distributions.distribution import Distribution


class TestKLDiv:
    def test_reduce(self):

        opt = KLDivergenceOptimizer(8, 2)

        dist = opt.reduce([1, 0, 2, 3, 5, 3, 1, 7])

        assert dist == [2, 0, 2, 2, 4, 4, 4, 4]
