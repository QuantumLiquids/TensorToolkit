# Tensor Manipulation

Location: `include/qlten/tensor_manipulation/`

## What lives here

- Contractions and permutations
- Decompositions (SVD, QR, EVD)
- Index fusion and reshaping helpers

## Key headers

- `qlten/tensor_manipulation/ten_ctrct.h`
- `qlten/tensor_manipulation/ten_decomp/ten_svd.h`
- `qlten/tensor_manipulation/ten_decomp/ten_qr.h`
- `qlten/tensor_manipulation/ten_decomp/ten_evd.h`

## Responsibilities

- Implement core numerical tensor operations on block-sparse data
- Bridge to BLAS/LAPACK through the backend selector
- Keep APIs consistent with MPI variants

## Related tests

- `tests/test_tensor_manipulation/`
