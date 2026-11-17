# xtorch implementation order

The tests in `tests.py` are numbered in dependency order. Implement one stage,
run that stage, and move on only when it passes. The suite uses only Python's
standard library and NumPy; pytest is not required.

## Commands

Run everything and stop at the first failure:

```sh
python -m unittest -v -f tests
```

Run a single stage (replace the class name as you progress):

```sh
python -m unittest -v tests.TestStage01TensorBasics
```

Run the final suite after every stage is implemented:

```sh
python -m unittest -v tests
```

## Implementation sequence

1. **`xtorch/tensor.py`: basic tensor state** — construction, `shape`, `ndim`,
   `zero_grad`, and `detach`. Run `TestStage01TensorBasics`.
2. **`xtorch/autograd.py`** — implement a deduplicating parent-first
   topological sort and `unbroadcast`. `unbroadcast` must remove leading
   dimensions introduced by broadcasting and sum dimensions whose original
   size was one. Run `TestStage02AutogradUtilities`.
3. **`xtorch/tensor.py`: arithmetic and reductions (forward pass)** — coerce
   numbers/arrays to constant tensors, implement `+`, `*`, unary `-`, `-`,
   powers, `sum`, and `mean`, and propagate `requires_grad`. Run
   `TestStage03TensorArithmetic`.
4. **`xtorch/tensor.py`: scalar reverse-mode autodiff** — give every operation
   a local `_backward`, seed scalar outputs with a gradient of one, traverse
   the graph in reverse topological order, and accumulate (do not overwrite)
   gradients. Non-scalar outputs require an explicit, shape-compatible
   gradient. Run `TestStage04ScalarAutograd`.
5. **`xtorch/tensor.py`: broadcasting and reduction gradients** — use
   `unbroadcast` for elementwise operands; expand reduction gradients back
   over reduced axes. Support integer and tuple axes. Run
   `TestStage05BroadcastAndReductionAutograd`.
6. **`xtorch/tensor.py`: matrix multiplication** — implement forward and
   reverse passes for matrix-matrix and vector-matrix products. Run
   `TestStage06MatmulAutograd`.
7. **`xtorch/tensor.py`: shape operations** — implement `reshape` (both a
   tuple and separate dimensions), `transpose`, and `T`, including inverse
   transformations in backward. Run `TestStage07ShapeOperations`.
8. **`xtorch/tensor.py`: nonlinear operations** — implement ReLU, exponential,
   logarithm, and tanh with their derivatives. Define ReLU's derivative at
   zero as zero. Run `TestStage08Activations`.
9. **`xtorch/nn/module.py`, then the other existing `xtorch/nn` building
   blocks** — recursively discover `Parameter` instances through modules and
   Python containers, without returning shared parameters twice. Verify
   `Linear`, activations, `Sequential`, and gradient clearing. Run
   `TestStage09Modules`.
10. **`xtorch/nn/loss.py`** — MSE is already expressed using tensor ops;
    implement mean cross entropy for a batch of class-index targets. Use a
    max-shift/log-sum-exp approach so large logits stay finite, and propagate
    the correct logits gradient. Run `TestStage10Losses`.
11. **`xtorch/optim/sgd.py`** — implement plain SGD, optional momentum, and
    safe handling of parameters whose gradient is `None`. Run
    `TestStage11SGD`.
12. **`xtorch/optim/adam.py`** — implement first/second moments, timestep,
    bias correction, epsilon stabilization, and missing-gradient handling.
    Run `TestStage12Adam`.
13. **Whole-library check** — run `TestStage13EndToEndTraining`. It trains a
    nonlinear `Sequential` model using autograd, MSE, recursive parameter
    discovery, gradient clearing, and Adam. Then run the complete suite.

Passing all stages means every public behavior represented by the supplied
library skeleton works together: tensor computation, reverse-mode autodiff,
neural-network composition, losses, and both optimizers.
