from itcl_quantizer.optimizers.IOptimizer import IOptimizer
from itcl_quantizer.quantizer.distributions.distribution import Distribution


class MinMaxOptimizer(IOptimizer):
    def __init__(self):
        pass

    def trim(self, distribution: Distribution) -> Distribution:
        """
        This optimizer does not trim the distribution.

        Args:
          distribution (Distribution): The distribution to be trimmed.

        Returns:
          The distribution is being returned.
        """
        return distribution
