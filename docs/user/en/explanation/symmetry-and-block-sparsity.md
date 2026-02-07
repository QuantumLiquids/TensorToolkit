# Symmetry and Block Sparsity

TensorToolkit enforces symmetry constraints by storing tensors in block-sparse form.
Only the blocks consistent with conservation rules are present, which reduces
memory and improves performance.

## Why block sparsity matters

For large tensor networks, dense tensors quickly become infeasible.
Symmetry constraints remove many combinations of indices, allowing you to keep
only the physically allowed blocks.

Benefits:

- Reduced memory usage
- Faster contractions and decompositions
- Exact enforcement of conserved quantities

## How it works

Each index is a set of sectors. A tensor is compatible with a target divergence
when the signed sum of the sector QNs is zero (or any specified target).
TensorToolkit only stores blocks that satisfy this rule.

## Practical implications

- You must initialize tensors with a target divergence (`Random` or `Fill`).
- Contractions automatically respect symmetry if index directions match.
- Fermionic tensors use the same block-sparse approach, with additional sign rules.

## Related docs

- [Indices and quantum numbers](indices-and-quantum-numbers.md)
- [Tensor contraction view](tensor-contraction-view.md)
