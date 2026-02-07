# Z2-Graded Fermions

TensorToolkit supports fermionic tensor networks by treating fermion parity as a
Z2 grading. This allows the library to automatically track sign changes that
arise from exchanging fermionic modes.

## What is Z2 grading?

A Z2 grading labels each sector as even (0) or odd (1). When you reorder fermionic
indices, the sign is determined by how many odd legs cross.

## How TensorToolkit represents fermions

Choose a fermionic quantum-number type from `qlten::special_qn`, for example:

- `fZ2QN` (pure fermion parity)
- `fU1QN` (U1 charge with parity derived from charge mod 2)
- `fZ2U1QN` (explicit parity plus U1 charge)

If the QN type is fermionic, tensor operations automatically handle signs.

## Practical rules

- Use `IN` and `OUT` directions consistently. Pair `OUT` with `IN` in contractions.
- Use `Dag()` to take the Hermitian conjugate; it handles fermionic signs.
- Use `GetQuasi2Norm()` if you need a positive-definite norm for diagnostics.

## Related docs

- [Fermion design (dev)](../../../dev/design/fermion-design.md)
- [Tensor contraction view](tensor-contraction-view.md)
