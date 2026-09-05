# Project guidance

This repository is a learning exercise: favour code that makes the neural-network maths and data flow easy to follow.

- Read `PROJECT_NOTES.md` before architectural or neural-network changes.
- Keep the implementation explicit and self-contained. Do not add Eigen, BLAS, or other maths/ML libraries. If matrices or tensors become useful, implement the required operations in this project.
- Preserve the established terms `activation` and `pre_activation`. Explain any terminology or structural change before applying it.
- A `NeuronLayer` abstraction is under consideration. Do not introduce it as a broad rewrite; propose a small, understandable migration tied to a concrete benefit.
- Prefer conceptual clarity and correctness over maximum optimisation. Avoid abstractions that hide forward propagation, backpropagation, or gradient descent.
- Use modern C++ consistently when touching code: `std::size_t` for sizes/indices, `const` references for large read-only objects, trailing underscores for private data members, and `[[nodiscard]]` or `noexcept` where their contracts are genuinely appropriate.
- Keep interface changes deliberate. Note existing naming and compatibility before renaming public methods.
- Treat weight initialisation as algorithmically important. Do not replace it with arbitrary constants or change its distribution without explaining the effect on symmetry, saturation, and training.
- Add or update focused checks when changing numerical behaviour. For learning-oriented work, small deterministic examples are preferable to opaque end-to-end tests alone.
- Do not modify datasets, generated Xcode files, or unrelated user changes unless the task specifically requires it.

