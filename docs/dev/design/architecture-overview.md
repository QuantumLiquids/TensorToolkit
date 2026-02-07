# Architecture Overview

TensorToolkit is a header-only C++ library that provides tensor network
building blocks with symmetry-aware, block-sparse data structures. The library
is organized around a small set of core abstractions and layered operation
modules.

## Core layers

1. **Core data structures** (`include/qlten/qltensor/`)
   - `Index` and `QNSector` define symmetry-aware index spaces.
   - `QLTensor` stores block-sparse tensor data with divergence constraints.

2. **Tensor manipulation** (`include/qlten/tensor_manipulation/`)
   - Contractions, permutations, decompositions (SVD/QR/EVD).
   - Uses BLAS/LAPACK and HPTT for performance.

3. **MPI tensor manipulation** (`include/qlten/mpi_tensor_manipulation/`)
   - MPI-enabled variants of decompositions and data movement.

4. **Framework + utilities** (`include/qlten/framework/`, `include/qlten/utility/`)
   - Backend selection for high-performance numerics.
   - Randomness, timers, and common helpers.

## Key design points

- **Symmetry first**: indices and tensors carry QN metadata; only valid blocks exist.
- **Header-only**: distribution is simple and avoids ABI issues.
- **Backend flexibility**: BLAS/LAPACK selection is centralized in the backend selector.
- **MPI as a first-class feature**: MPI-enabled operations live in
  `include/qlten/mpi_tensor_manipulation/` and mirror the CPU APIs in
  `include/qlten/tensor_manipulation/`. Example mapping:
  `SVD` (CPU) ↔ `MPISVDMaster`/`MPISVDSlave` (MPI).

## Runtime linking note

Although the library is header-only, many tensor operations link to external
BLAS/LAPACK libraries. Tests and MPI-enabled operations also link against an MPI
implementation.

Notes:

- `QLTensor` headers currently include `mpi.h`:
  `qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h` includes
  `qlten/framework/hp_numeric/mpi_fun.h`, which includes `mpi.h`. So including
  `qlten/qltensor/qltensor.h` (or the umbrella headers `qlten/qlten.h` /
  `qlten/qltensor_all.h`) requires MPI headers at compile time, even if you do
  not call MPI APIs.
- There is currently no supported build option to remove this compile-time MPI
  header dependency.
- If you only use the core types (`Index`, `QNSector`, `QLTensor`) and avoid
  decompositions/BLAS wrappers and MPI calls, you can keep runtime dependencies
  minimal (no BLAS/MPI linking beyond what your application uses).

## Entry points

- `qlten/qlten.h`: umbrella header for common usage.
- `qlten/qltensor_all.h` and `qlten/tensor_manipulation_all.h`: convenience bundles.

## Related docs

- [Backend architecture](backend-architecture.md)
- [Fermion design](fermion-design.md)
