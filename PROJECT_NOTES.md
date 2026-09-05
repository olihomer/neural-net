# Neural net v1: project notes

## Purpose and priorities

This is a from-scratch C++ neural-network project and, above all, a learning exercise. The aim is to expose the calculations and data movement clearly enough that each stage can be understood and inspected. Correctness and conceptual clarity take precedence over framework-like generality or maximum performance.

The project should remain independent of external maths and machine-learning libraries. Do not introduce Eigen, BLAS, or similar dependencies. If matrix or tensor types become useful, build only the operations the project needs and keep their relationship to the underlying neural-network equations visible.

## Current repository structure

- `neural net v1/neural.hpp` and `neural.cpp` contain the network representation, forward propagation, backpropagation, accumulated gradients, gradient descent, setters/getters, and diagnostic printing.
- `neural net v1/data_set.hpp` and `data_set.cpp` define a generic supervised sample (`inputs` and `outputs`) and load the project's simple text format.
- `neural net v1/mnist_data.hpp` and `mnist_data.cpp` load labelled 28 x 28 MNIST CSV rows, normalise pixel values to `[0, 1]`, and create ten-element one-hot targets.
- `neural net v1/main.cpp` currently constructs a `784 -> 10 -> 10` network, trains it repeatedly on 100 MNIST examples, and samples predictions for inspection.
- `test.cpp`/`test.hpp` are currently only a small placeholder rather than neural-network tests.
- The Xcode project is `neural net v1.xcodeproj`. The repository also contains local data and large/archive files; these are not source-code targets.

## Current network model

The current `layer` struct stores:

- `size`
- `activation` and `pre_activation`
- backpropagated `error`
- `bias` and accumulated `bias_gradient`
- incoming `weight` and accumulated `weight_gradient`

Each non-input layer owns a weight row per neuron, with one entry per activation in the preceding layer. In other words, `weight[current_neuron][previous_neuron]` represents the connection into the current neuron. The input layer uses its `activation` vector as the supplied feature values.

`neural::propogate()` performs an explicit dense forward pass. For every neuron after the input layer it computes the weighted sum plus bias, stores that value as `pre_activation`, and stores `sigmoid(pre_activation)` as `activation`. The established distinction between `pre_activation` and `activation` is useful and should be retained.

Training currently works in batches represented by an entire `data_set`:

1. Clear accumulated bias and weight gradients.
2. Copy each sample's inputs into the first layer and run forward propagation.
3. Form the output error as `activation - target`.
4. Walk backwards through the layers, accumulating weight gradients as error times the previous activation.
5. Propagate hidden-layer error through the next layer's weights and multiply by the sigmoid derivative evaluated at `pre_activation`.
6. Accumulate squared error for reporting.
7. Apply the accumulated gradients later through `gradient_descent()` using the fixed learning rate.

The source comments describe the output error as the logistic/cross-entropy form, while the reported `cost_function()` is squared error. This is worth keeping visible as a learning/design question if the loss calculation is revisited; do not paper over the distinction with an opaque abstraction.

## Weight initialisation history

Weight initialisation was an important prior fix. Random, non-identical initial weights are necessary to break symmetry so neurons can learn different features. Their scale also matters: overly large magnitudes can push sigmoid units into saturated regions with very small derivatives, while unsuitable fixed or identical values can prevent useful differentiation between neurons.

The current constructor initialises weights explicitly with small random values (approximately `-0.098` to `0.1`) and biases with positive random values. Treat this as intentional current behaviour, not disposable boilerplate. Any future change should explain:

- the distribution and scale being chosen;
- how it relates to fan-in/fan-out;
- the implications for sigmoid saturation and gradient flow;
- whether deterministic seeding is wanted for tests and reproducible experiments.

Do not silently replace this with zero initialisation or arbitrary constants.

## Possible `NeuronLayer` direction

A `NeuronLayer` abstraction has been considered but is not yet the current implementation. It could eventually own the vectors now held by `layer` and provide invariants for their dimensions. The benefit would be grouping layer state and reducing indexing mistakes—not concealing the mathematics.

If introduced, prefer a staged change:

1. Give the existing layer data a clear type and preserve its visible fields or simple accessors.
2. Establish dimensional invariants for activations, biases, incoming weights, and gradients.
3. Move only cohesive layer-level behaviour when doing so clarifies the forward/backward equations.
4. Keep network-level traversal and learning steps readable in order.

Avoid a broad hierarchy of neuron, layer, optimiser, tensor, and loss abstractions all at once. Each abstraction should answer a concrete problem encountered in the current code.

## C++ conventions for future changes

- Prefer `std::size_t` for vector sizes and indices and avoid narrowing conversions.
- Pass large read-only values, including vectors and datasets, by `const` reference where appropriate.
- Use trailing underscores for new or renamed private members. Existing `m_` members can be migrated deliberately rather than through an unrelated mass rename.
- Add `[[nodiscard]]` to query functions where accidentally discarding the result is likely a mistake.
- Add `noexcept` only when the implementation and members support that guarantee; it should communicate a real contract rather than decorate every function.
- Prefer clear range-based loops where they do not hide indices needed by the maths, and avoid copying full samples or layers unintentionally.
- Keep headers self-sufficient and use the standard header that declares each facility.
- Explain public API renames and preserve compatibility when it matters. Current code includes the public name `propogate`; correcting such names should be a conscious change, not incidental churn.

## Testing and change strategy

Numerical changes should be developed in small, inspectable steps. Useful checks include:

- a tiny network with manually chosen weights and a hand-calculated forward pass;
- a finite-difference gradient check on a very small network;
- confirmation that gradients are reset between batches;
- dimension/invariant checks for topology, inputs, and targets;
- deterministic initialisation in tests where exact results matter;
- a small MNIST smoke test kept separate from mathematical unit checks.

When proposing structural work, first explain which current problem it addresses, how the data layout or equations will change, and how the result will be verified. Preserve the user's in-progress edits and avoid unrelated source cleanup.
