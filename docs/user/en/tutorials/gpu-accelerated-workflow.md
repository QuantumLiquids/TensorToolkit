# GPU-Accelerated Workflow

This tutorial shows the end-to-end steps to enable GPU support and validate your setup.
GPU enablement is used by tests and by downstream integration; the bundled examples
are CPU-only for now.

## 1. Install CUDA + cuTENSOR

Ensure the following are available:

- CUDA toolkit (11.0+)
- cuBLAS and cuSOLVER (bundled with CUDA)
- cuTENSOR (separate package)

## 2. Configure CMake

```bash
mkdir -p build
cd build
cmake .. \
  -DHP_NUMERIC_USE_OPENBLAS=ON \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_ROOT=/path/to/cutensor
```

The headers are gated by the compile-time macro `USE_GPU`. In this repo,
`QLTEN_USE_GPU=ON` sets `-DUSE_GPU=1` for tests. If you integrate TensorToolkit
elsewhere, define `USE_GPU` yourself and link CUDA + cuTENSOR.

If your CUDA install is non-standard:

```bash
cmake .. \
  -DHP_NUMERIC_USE_OPENBLAS=ON \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_ROOT=/path/to/cutensor \
  -DCUDAToolkit_ROOT=/path/to/cuda
```

## 3. Build and run tests

```bash
make -j4
ctest --output-on-failure
```

## 4. Validate performance

For GPU-heavy workloads, compare GPU and CPU outputs on a small tensor first,
then scale up. This helps catch configuration issues early.

## Notes

- When GPU support is enabled, the bundled HPTT build is disabled automatically.
- The examples CMake hard-errors if `QLTEN_USE_GPU=ON` (examples are CPU-only).
- Make sure your runtime environment can locate CUDA libraries (for example via
  `LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS).
- If you are on a shared cluster, confirm which CUDA and cuTENSOR modules are supported.
