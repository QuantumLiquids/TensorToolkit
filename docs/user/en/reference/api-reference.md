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

- `qlten/qlten.h` (umbrella header)
- `qlten/qltensor/` (core tensor and index types)
- `qlten/tensor_manipulation/` (contractions and decompositions)
- `qlten/mpi_tensor_manipulation/` (MPI-enabled operations)

## Suggested reading path

1. `Index`, `QNSector`, `QLTensor`
2. `Contract` and `SVD`
3. MPI extensions if needed

If you are new, start with the tutorials and use Doxygen as a lookup.
