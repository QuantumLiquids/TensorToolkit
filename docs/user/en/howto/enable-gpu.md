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
  -DQLTEN_USE_GPU=ON
```

If CUDA or cuTENSOR are not discoverable automatically, set one of:

- `CUDAToolkit_ROOT=/path/to/cuda`
- `CUTENSOR_ROOT=/path/to/cutensor/prefix`
- `CUTENSOR_INCLUDE_DIR=/path/to/cutensor/include`
- `CUTENSOR_LIBRARY=/path/to/libcutensor.so`
- `QLTEN_CUDA_ARCHITECTURES=90` (or another semicolon-separated list) if you want explicit GPU code generation targets

Typical `CUTENSOR_ROOT` values:

- NVHPC cluster install:
  `/opt/nvidia/hpc_sdk/.../math_libs/<cuda-version>/targets/x86_64-linux`
- user-local install:
  `${HOME}/.local/usr`

See [Install cuTENSOR without root](install-cutensor.md) for cluster-specific examples.

If `QLTEN_CUDA_ARCHITECTURES` is left empty, TensorToolkit defers to
CMake/NVCC defaults. The `CUDAARCHS` environment variable is also honored by
CMake.

## Build and test

```bash
make -j4
ctest --output-on-failure
```

To verify the installed package surface itself from an existing build tree, run:

```bash
cmake --build . --target verify-package-gpu
```

If CMake cannot find MPI while configuring tests, set `MPI_CXX_COMPILER` (or
adjust `CMAKE_PREFIX_PATH`) to your MPI installation.

## Notes

- When `QLTEN_USE_GPU=ON`, the bundled HPTT build is disabled automatically.
- The headers are gated by the compile-time macro `USE_GPU`. In this repo,
  `QLTEN_USE_GPU=ON` sets `-DUSE_GPU=1` for tests. Downstream package consumers
  should not define `USE_GPU` or wire CUDA/cuTENSOR manually; install the GPU
  package and link `TensorToolkit::TensorToolkit` instead.
- The examples CMake hard-errors if `QLTEN_USE_GPU=ON` (examples are CPU-only).
- GPU test binaries link `MPI::MPI_CXX` because some GPU tests still include MPI
  wrappers.
- GitHub CI currently automates the CPU package verifier only. Run
  `verify-package-gpu` on a real GPU node before treating GPU packaging changes
  as validated.
- Ensure your runtime environment can locate CUDA libraries (for example via
  `LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS).
