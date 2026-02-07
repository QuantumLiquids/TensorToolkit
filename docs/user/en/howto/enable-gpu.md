# Enable GPU Support

TensorToolkit can use CUDA for GPU-accelerated tensor operations. GPU support is
controlled at build time and is primarily used by tests and integration into
your own projects. The bundled examples are CPU-only for now.

## Requirements

- CUDA toolkit (11.0+)
- cuBLAS and cuSOLVER (part of CUDA)
- cuTENSOR (separate package, required for GPU tensor operations)

If you follow this page's `ctest` flow, MPI discovery is also required by the
test CMake (`find_package(MPI REQUIRED)` in `tests/CMakeLists.txt`).

## Configure CMake

```bash
cmake .. \
  -DHP_NUMERIC_USE_OPENBLAS=ON \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_ROOT=/path/to/cutensor
```

If your CUDA toolkit is not in a standard location, also set:

- `CUDAToolkit_ROOT=/path/to/cuda`

## Build and test

```bash
make -j4
ctest --output-on-failure
```

If CMake cannot find MPI while configuring tests, set `MPI_CXX_COMPILER` (or
adjust `CMAKE_PREFIX_PATH`) to your MPI installation.

## Notes

- When `QLTEN_USE_GPU=ON`, the bundled HPTT build is disabled automatically.
- The headers are gated by the compile-time macro `USE_GPU`. In this repo,
  `QLTEN_USE_GPU=ON` sets `-DUSE_GPU=1` for tests. If you integrate
  TensorToolkit elsewhere, define `USE_GPU` yourself and link CUDA + cuTENSOR.
- The examples CMake hard-errors if `QLTEN_USE_GPU=ON` (examples are CPU-only).
- GPU test binaries currently do not link `MPI::MPI_CXX`, but test
  configuration still requires MPI package/header discovery.
- Ensure your runtime environment can locate CUDA libraries (for example via
  `LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS).
