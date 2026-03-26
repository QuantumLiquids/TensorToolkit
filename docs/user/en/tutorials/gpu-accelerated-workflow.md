# GPU-Accelerated Workflow

This tutorial shows the end-to-end steps to enable GPU support and validate your setup.
GPU enablement is used by tests and by downstream integration; the bundled examples
are CPU-only for now.

## 1. Install CUDA + cuTENSOR

Ensure the following are available:

- CUDA toolkit (11.0+)
- cuBLAS and cuSOLVER (bundled with CUDA)
- cuTENSOR (separate package)

There are two common deployment models:

- NVHPC cluster: cuTENSOR already lives inside the NVIDIA HPC SDK tree
- user-local install: cuTENSOR is unpacked under a home-directory prefix such as
  `${HOME}/.local/usr`

See [Install cuTENSOR without root](../howto/install-cutensor.md) for both flows.

## 2. Configure CMake

```bash
mkdir -p build
cd build
cmake .. \
  -DHP_NUMERIC_USE_OPENBLAS=ON \
  -DQLTEN_USE_GPU=ON
```

If you want to pin the generated GPU architectures explicitly, add for example:

```bash
-DQLTEN_CUDA_ARCHITECTURES=90
```

For H200/Hopper, `90` is the relevant target. On mixed or newer clusters, use a
semicolon-separated list such as `90;100` when needed.

The headers are gated by the compile-time macro `USE_GPU`. In this repo,
`QLTEN_USE_GPU=ON` sets `-DUSE_GPU=1` for tests. For downstream projects, prefer
the installed-package flow:

```cmake
find_package(TensorToolkit CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE TensorToolkit::TensorToolkit)
```

The GPU package target carries the required CUDA/cuBLAS/cuSOLVER/cuTENSOR usage
requirements for the installed variant.

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

To verify the installed GPU package from the current build tree:

```bash
cmake --build . --target verify-package-gpu
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
- GitHub CI currently automates only the CPU package verifier. GPU package
  verification is expected to run on a real GPU node.
