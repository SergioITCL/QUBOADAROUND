from itcl_quantizer.equalizers.param_equalizer.abstract_param_optimizer import (
    AbstractParamOptimizer,
)
from simanneal import Annealer

from itcl_quantizer.tensor_extractor.tensor import NodeTensorBase


class ParamEqAnnealer(AbstractParamOptimizer, Annealer):
    """Simmulated Annearling implementation of the Abstrac Param Optimizer

    Args:
        AbstractParamOptimizer (_type_): _description_
        Annealer (_type_): _description_
    """

    def __init__(
        self,
        t_min=0.1,
        t_max=25000,
        updates=10000,
        steps=5000,
    ):
        """

        Args:
            t_min (float, optional): Minimum Temperature. Defaults to 0.1.
            t_max (int, optional): Maximum Temperature. Defaults to 25000.
            updates (int, optional): Number of updates to show the user. Defaults to 100.
            steps (int, optional): Number of iterations. Defaults to 5000.
        """
        self.updates = updates
        self.Tmax = t_max
        self.Tmin = t_min
        self.steps = steps

    def anneal(self):
        """Main Function of the Super() Annealer class

        Returns:
            _type_: _description_
        """
        best, cost = super().anneal()

        return best, cost

    def energy(self) -> float:
        """Function that calculates the energy or loss

        Returns:
            float: loss
        """

        energy: float = self._cost_fn(self.state)
        return energy

    def optimize(self) -> tuple[list[NodeTensorBase], float]:
        """Base IOptimizer Optimization class

        Returns:
            Tuple[List[np.ndarray], float]: A tuple with the improved neighborhood and the final cost
        """
        return self.anneal()  # type: ignore
