# Matrix Decomposition Concepts

TensorToolkit provides SVD, QR, and EVD operations for block-sparse tensors.
These are used for truncation, canonicalization, and algorithmic building blocks.

## SVD (Singular Value Decomposition)

SVD factors a tensor (reshaped as a matrix) into:

```text
T = U * S * Vt
```

Here `Vt` is the right SVD factor returned by the API (`Vt` variable). In
matrix notation this corresponds to a transposed/conjugate-transposed right
factor (`V^T` or `V^\dagger`, depending on scalar type/convention).

In TensorToolkit, SVD respects symmetry sectors. You provide:

- `ldims`: how many leading indices go to the left side
- `trunc_err`: target truncation error (squared-norm convention)
- `Dmin` and `Dmax`: min/max kept dimension

Clarifications:

- `ldims` always splits by prefix. If you want a non-prefix bipartition,
  transpose the tensor so the desired left group is the leading indices.
- The new bond index directions are fixed: `U` appends an `OUT` index, `Vt`
  prepends an `IN` index, and `S` is `IN x OUT`.
- `trunc_err` uses the squared-norm convention:
  `1 - sum(kept s_i^2) / sum(all s_i^2)`. There is no outer square root.

This is the standard tool for bond-dimension truncation.

## QR decomposition

QR is used for orthogonalization or canonical forms. It is faster than SVD when
you do not need singular values.

## EVD (Eigenvalue Decomposition)

EVD applies to symmetric or Hermitian matrices. It is commonly used for
density matrices and environment calculations.

## How to choose

- Use **SVD** for truncation and entanglement-aware splitting.
- Use **QR** for orthogonalization or preconditioning.
- Use **EVD** when the object is Hermitian and eigenvalues are meaningful.

## Related docs

- [Contractions and decompositions (tutorial)](../tutorials/contractions-and-decompositions.md)
- [API reference](../reference/api-reference.md)
