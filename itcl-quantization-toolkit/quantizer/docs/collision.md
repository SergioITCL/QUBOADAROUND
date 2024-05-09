## API

Each `Layer` has a `calc_collisions` method. This method takes as input the
quantization result of the layer and a collision function.

`calc_collisions` returns a dictionary with the number of collisions for each named tensor node.

### Collision function

By default, a collision function takes as input the following arguments:

1. `x`: the floating point tensor
2. `x'`: the quantized tensor

and returns an integer with the number of collisions between `x` and `x'`.

Note: `x` and `x'` are assumed to be in the same shape.

**Default collision function**

The default collision function takes 3 arguments, `x`, `x'` and `epsilon`.

Epsilon is an integer that represents the number of decimals that a floating point number is rounded to before a comparison is made.

```python
from itcl_quantizer.util.collisions import calc_collisions

collision_policy = partial(calc_collisions, epsilon=7)
```

### Example

```python
from itcl_quantizer.util.collisions import calc_collisions

net = build(...)

collision_policy = partial(calc_collisions, epsilon=7)

for q_res in net.as_quant_results():
    print("Layer:", q_res.layer)
    for name, collisions in q_res.layer.calc_collisions(
        q_res, collision_policy
    ).items():
        print(f"  Quantized node {name} has {collisions} collisions")
```

## NOTE - Batch Tensors

Some tensors are batched, such as an input tensor of a Dense layer. In this case, the number of collisions is the average number of collisions in each axis of the batched tensor.
