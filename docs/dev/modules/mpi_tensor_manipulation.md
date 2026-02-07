# MPI Tensor Manipulation

Location: `include/qlten/mpi_tensor_manipulation/`

## What lives here

- MPI-enabled decompositions (SVD and related helpers)
- Distributed data movement for block-sparse tensors

## Key headers

- `qlten/mpi_tensor_manipulation/ten_decomp/mpi_svd.h`

## Responsibilities

- Provide master/slave MPI workflows for blockwise SVD
- Mirror the CPU API while distributing heavy work across ranks

## Related tests

- `tests/test_tensor_mpi/`
