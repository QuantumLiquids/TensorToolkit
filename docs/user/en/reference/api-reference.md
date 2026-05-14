# API Reference

TensorToolkit is header-only and documented via Doxygen.

## Doxygen entry point

From the repository root, build the docs and open:

```bash
cd docs
doxygen Doxyfile
```

Then open:

```
docs/build/html/index.html
```

## Common headers

- `qlten/qlten.h` (full umbrella header)
- `qlten/core.h` (core tensor/index/QN entry point)
- `qlten/tensor_manipulation_all.h` (non-MPI tensor manipulation bundle)
- `qlten/mpi_tensor_manipulation_all.h` (MPI tensor manipulation bundle)
- `qlten/qltensor/` (core tensor and index types)
- `qlten/tensor_manipulation/` (contractions and decompositions)
- `qlten/mpi_tensor_manipulation/` (MPI-enabled operations)

## Contraction APIs

- `qlten::Contract(&a, &b, axes_set, &c)` is the general arbitrary-axis
  contraction API.
- `qlten::ContractContiguousAxes(a, b, a_start, b_start, size, c)` is the
  performance-oriented API for contiguous ascending contracted axes. New code
  should prefer this name over the legacy bool-tagged contiguous-axes
  `Contract<T, QNT, bool, bool>(...)` overload.
- DMRG-specialized helpers live under `qlten::dmrg` and are included by the
  full `qlten/qlten.h` umbrella and `qlten/tensor_manipulation_all.h`.

## Suggested reading path

1. `Index`, `QNSector`, `QLTensor`
2. `Contract` and `SVD`
3. MPI extensions if needed

If you are new, start with the tutorials and use Doxygen as a lookup.
