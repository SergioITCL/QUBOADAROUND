from itcl_quantizer.quantizer.distributions.distribution import Distribution


class IOptimizer:
    def __init__(self):
        pass

    def trim(self, distribution: Distribution) -> Distribution:
        raise NotImplementedError
