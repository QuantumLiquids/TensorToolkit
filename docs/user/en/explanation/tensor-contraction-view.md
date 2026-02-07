# Tensor Contraction View

Contraction is the core operation in tensor networks. Conceptually, it is the
sum over shared indices between tensors.

## Index matching

A contraction is valid when:

- the index dimensions match
- the indices are inverses of each other (IN vs OUT)

TensorToolkit uses direction to track sign conventions and symmetry divergence.

## Result ordering

The general contraction API keeps all uncontracted indices of the left tensor
first, then those of the right tensor. This predictable order is useful when you
construct larger networks.

## Practical tips

- Keep a consistent index ordering when building networks.
- For performance, minimize transpositions by grouping contracted axes.
- Use `Contract` for general cases and rely on Doxygen for specialized APIs.

## Related docs

- [Contractions and decompositions (tutorial)](../tutorials/contractions-and-decompositions.md)
- [Symmetry and block sparsity](symmetry-and-block-sparsity.md)
