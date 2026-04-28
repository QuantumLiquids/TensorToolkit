# CMake Options

This page lists the top-level CMake options defined in
[`CMakeLists.txt`](../../../../CMakeLists.txt) and the related cache variables
that affect the installed package surface.

## Top-level options

| Option | Default | Description |
| --- | --- | --- |
| `QLTEN_USE_GPU` | `OFF` | Select the GPU package variant. This changes the exported public dependency set from CPU BLAS/LAPACK/HPTT to CUDA/cuBLAS/cuSOLVER/cuTENSOR. |
| `QLTEN_COMPILE_HPTT_LIB` | `ON` | Build and install the bundled HPTT library for CPU transpose operations. Forced `OFF` when `QLTEN_USE_GPU=ON`. |
| `QLTEN_BUILD_UNITTEST` | `OFF` | Build unit tests. The package verification targets are top-level targets and do not depend on this option. |
| `QLTEN_BUILD_EXAMPLES` | `OFF` | Build the CPU-only example programs in `examples/`. |
| `QLTEN_TIMING_MODE` | `OFF` | Enable `QLTEN_TIMING_MODE` for targets built inside this source tree. This is not exported through the installed package. |
| `QLTEN_MPI_TIMING_MODE` | `OFF` | Enable `QLTEN_MPI_TIMING_MODE` for targets built inside this source tree. This is not exported through the installed package. |

`QLTEN_TIMING_MODE` and `QLTEN_MPI_TIMING_MODE` only enable diagnostic timing
prints in inline header code. They are intentionally not part of the installed
package interface. Downstream projects that want this instrumentation should opt
in on their own targets, for example:

```cmake
target_compile_definitions(my_target PRIVATE QLTEN_TIMING_MODE QLTEN_MPI_TIMING_MODE)
```

## CPU backend options

These options live in `cmake/Modules/MathBackend.cmake` and affect CPU builds
and CPU package exports.

| Option | Default | Description |
| --- | --- | --- |
| `HP_NUMERIC_USE_MKL` | `OFF` | Select Intel MKL and export `HP_NUMERIC_BACKEND_MKL`. |
| `HP_NUMERIC_USE_AOCL` | `OFF` | Select AMD AOCL and export `HP_NUMERIC_BACKEND_AOCL`. |
| `HP_NUMERIC_USE_OPENBLAS` | `OFF` | Select OpenBLAS and export `HP_NUMERIC_BACKEND_OPENBLAS`. |

Exactly one CPU backend should be active for CPU installs. If none is set, the
build auto-selects a backend based on the current machine.

## Package export behavior

- CPU installs export MPI, the selected BLAS/LAPACK backend, OpenMP when
  required, and HPTT when `QLTEN_COMPILE_HPTT_LIB=ON`.
- If `QLTEN_COMPILE_HPTT_LIB=OFF`, downstream consumers must also make the
  external HPTT dependency discoverable with `CMAKE_PREFIX_PATH`,
  `hptt_INCLUDE_DIR`, or `hptt_LIBRARY`.
- GPU installs export MPI, `CUDAToolkit`, `CUTENSOR`, and the CUDA runtime,
  cuBLAS, and cuSOLVER link interfaces.
- CPU and GPU installs should use different prefixes because they export
  different public dependency sets.
- Supported downstream entry points are `CMAKE_PREFIX_PATH` and
  `TensorToolkit_DIR`.
- On the cluster nodes used for this repository, source `~/myenv.sh` before
  configuring so `CC`, `CXX`, `MKLROOT`, `AOCL_ROOT`, CUDA toolchain paths, and
  other backend hints are populated consistently.

Downstream usage should be:

```cmake
find_package(TensorToolkit CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE TensorToolkit::TensorToolkit)
```

## Common cache variables for GPU discovery

These are not top-level `option()` entries, but they are commonly used when
configuring GPU builds:

- `CUDAToolkit_ROOT` to point CMake at a non-standard CUDA toolkit location
- `CUTENSOR_ROOT` to point CMake at a cuTENSOR prefix or target directory
- `CUTENSOR_INCLUDE_DIR` and `CUTENSOR_LIBRARY` as exact-path overrides
- `QLTEN_CUDA_ARCHITECTURES` for explicit test-build CUDA targets

## Verification targets

TensorToolkit defines these top-level verification targets in any configured
build tree:

- `verify-package-cpu`
- `verify-package-gpu`

These targets run the install-consume verification scripts under
`tests/package_consumer/`. The CPU target configures, builds, and runs its
consumer smoke. The GPU target configures, builds, and runs the downstream GPU
consumer smoke. On nodes without a working CUDA driver or visible GPU it is
reported as skipped; on real GPU nodes it should run to completion. Each
invocation uses a fresh temporary scratch directory under `/tmp` so concurrent
runs do not clobber each other.

GitHub CI currently automates `verify-package-cpu`. `verify-package-gpu` is
intended for manual or self-hosted GPU-node verification.
