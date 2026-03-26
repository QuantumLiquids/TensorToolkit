# Build and Run Tests

TensorToolkit uses GoogleTest and CMake for unit tests.

## Configure and build

From the `build` directory:

```bash
cmake .. \
  -DQLTEN_BUILD_UNITTEST=ON \
  -DGTest_DIR=/path/to/googletest \
  -DHP_NUMERIC_USE_OPENBLAS=ON
make -j4
```

Notes:

- Select exactly one BLAS backend: `-DHP_NUMERIC_USE_OPENBLAS=ON` (or `..._MKL` /
  `..._AOCL`).
- CMake test configuration always requires MPI discovery (`FindMPI`).
- CPU test binaries link `MPI::MPI_CXX`.
- GPU test binaries also link `MPI::MPI_CXX`, because some GPU tests still
  include MPI wrappers.

## Run all tests

```bash
ctest --output-on-failure
```

## Run a single test

```bash
ctest -R test_ten_svd --output-on-failure
```

## MPI tests on a single node

Some MPI tests may require oversubscribe on a workstation:

```bash
OMPI_MCA_rmaps_base_oversubscribe=1 ctest -R test_mpi --output-on-failure
```

## Tips

- If CMake reports missing `OpenMP_CXX`, see the note in
  [Build and install](build-and-install.md).
- For GPU-related tests, enable GPU support in CMake and ensure CUDA libraries are on your path.
- To verify the installed package surface rather than only the in-tree tests,
  run `cmake --build . --target verify-package-cpu` or
  `cmake --build . --target verify-package-gpu` from the current build tree.
