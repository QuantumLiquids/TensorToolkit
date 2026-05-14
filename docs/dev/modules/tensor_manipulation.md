# Tensor Manipulation

Location: `include/qlten/tensor_manipulation/`

## What lives here

- Contractions and permutations
- Decompositions (SVD, QR, EVD)
- Index fusion and reshaping helpers

## Key headers

- `qlten/tensor_manipulation/index_lineage.h`
- `qlten/tensor_manipulation/ten_ctrct.h`
- `qlten/tensor_manipulation/contract_contiguous_axes.h`
- `qlten/tensor_manipulation/dmrg/contract_1sector.h`
- `qlten/tensor_manipulation/dmrg/block_expand.h`
- `qlten/tensor_manipulation/ten_decomp/ten_svd.h`
- `qlten/tensor_manipulation/ten_decomp/ten_qr.h`
- `qlten/tensor_manipulation/ten_decomp/mat_evd.h`

`qlten/tensor_manipulation_all.h` aggregates these non-MPI tensor
manipulation headers, including the DMRG helpers under `qlten::dmrg`.

## Design notes

- [Index lineage design](../design/index-lineage.md)
- [Tensor contraction API design](../design/tensor-contraction-api.md)

## Responsibilities

- Implement core numerical tensor operations on block-sparse data
- Bridge to BLAS/LAPACK through the backend selector
- Keep APIs consistent with MPI variants

## Related tests

- `tests/test_tensor_manipulation/`
