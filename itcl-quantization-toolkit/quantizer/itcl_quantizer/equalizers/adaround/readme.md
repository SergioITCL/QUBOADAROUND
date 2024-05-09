# AdaRound

Adaround is a layer equalizer that changes how the quantized weights are rounded from a float to an integer. 

The rounding policy is a binary array that indicates if the weight should be floored (0) or ceiled (1). 

This adaptive rounding reduces the quantization noise compared to a round-to-nearest rounding policy.

## How can a layer be adarounded?

Adaroundable layers should override the ```adaround()``` method, this method updates the rounding policy of specific nodes that implement the IRoundingPolicy interface by taking into account an specific cost function. 


## Details

Each rounding optimizer implements the IRoundOptimizer interface, and is in charge of finding the best numpy binary array that minimizes the cost function.

The RoundingAnnealer is a clear example of a Rounding Optimizer that implements the Simulated Annealing metaheuristic algorithm. 


The IRoundingPolicy interface allows a class to be adarounded, as it implements a getter and a setter to update the current rounding policy by the optimizer.

The Adaround class implements the adaround orchestrator itself. This orchestrator can apply adaround to a quantized network, taking the network loss/accuracy into account, or just to a single layer. 