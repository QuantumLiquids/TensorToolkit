# Developer Setup

This guide describes a typical developer build for TensorToolkit.

## Prerequisites

- C++17 compiler
- CMake 3.27+
- BLAS/LAPACK
- MPI headers (required; examples and CPU tests link `MPI::MPI_CXX`)
- GoogleTest (for unit tests)

## Recommended build

```bash
mkdir -p build
cd build

cmake .. \
  -DQLTEN_BUILD_UNITTEST=ON \
  -DQLTEN_BUILD_EXAMPLES=OFF \
  -DHP_NUMERIC_USE_OPENBLAS=ON \
  -DGTest_DIR=/path/to/googletest

make -j4
ctest --output-on-failure
```

Notes:

- `tests/` and `examples/` both call `find_package(MPI REQUIRED)`.
- Examples, CPU tests, and GPU tests all currently link `MPI::MPI_CXX`.
- MPI headers are also required to compile `QLTensor` today, because
  `include/qlten/qltensor/blk_spar_data_ten/blk_spar_data_ten.h` includes
  `include/qlten/framework/hp_numeric/mpi_fun.h`, which includes `mpi.h`.
  Including any of the following will therefore pull in `mpi.h`:
  - `qlten/qltensor/qltensor.h`
  - `qlten/qltensor_all.h`
  - `qlten/qlten.h`
- There is currently no supported "no-MPI headers" mode for `QLTensor`. If you
  need an MPI-free build, restrict usage to the non-MPI headers (for example
  `qlten/qltensor/index.h` and `qlten/qltensor/qnsct.h`) and avoid including
  `qltensor.h`/`qlten.h`.
- CMake uses `FindMPI`. Choose an MPI implementation by setting
  `MPI_CXX_COMPILER` or adjusting `PATH`/`CMAKE_PREFIX_PATH` to your MPI install.

## macOS (Homebrew LLVM)

If you need OpenMP support, use Homebrew LLVM:

```bash
cmake .. \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DQLTEN_BUILD_UNITTEST=ON \
  -DHP_NUMERIC_USE_OPENBLAS=ON \
  -DGTest_DIR=/opt/homebrew/lib/cmake/GTest
```

If CMake reports missing `OpenMP_CXX`, rerun the exact same CMake command once.

## Useful options

- `QLTEN_USE_GPU=ON` to enable GPU code paths (tests/examples)
- `QLTEN_COMPILE_HPTT_LIB=OFF` to use external HPTT (see below)
- `QLTEN_BUILD_EXAMPLES=ON` to build examples

## BLAS backend selection (tests/examples)

Tests/examples include `cmake/Modules/MathBackend.cmake`, which requires a BLAS
backend selection. Set exactly one of:

- `-DHP_NUMERIC_USE_MKL=ON`
- `-DHP_NUMERIC_USE_AOCL=ON`
- `-DHP_NUMERIC_USE_OPENBLAS=ON`

If you omit these, `MathBackend.cmake` will auto-select based on CPU vendor, and
may fail if `MKLROOT`/`AOCL_ROOT` are not set. Explicit selection is recommended.

Auto-selection rules (from `cmake/Modules/MathBackend.cmake`):

- If `CMAKE_SYSTEM_PROCESSOR` matches `x86_64`:
  - On Linux, if `/proc/cpuinfo` reports `vendor_id=AuthenticAMD`, select AOCL.
  - Otherwise select MKL (including non-Linux x86_64 where `/proc/cpuinfo` is missing).
- If `CMAKE_SYSTEM_PROCESSOR` matches `arm64`/`aarch64`, select OpenBLAS.
  - On macOS arm64, the module also adds Homebrew defaults:
    `/opt/homebrew/opt/openblas` and `/opt/homebrew/opt/lapack`.
- Otherwise select OpenBLAS.

Failure modes to expect when relying on auto-selection:

- **MKL**: configuration fails if `MKLROOT` is not set.
- **AOCL**: configuration fails if `AOCL_ROOT` is not set.
- **OpenBLAS**: configuration can fail if headers/libs are not discoverable;
  set `CMAKE_PREFIX_PATH` (or `OpenBLAS_ROOT`) to your OpenBLAS install.

If CMake cannot find BLAS/LAPACK for your chosen backend:

- **MKL**: set `MKLROOT` and ensure `${MKLROOT}/include` and `${MKLROOT}/lib` are visible.
- **AOCL**: set `AOCL_ROOT` and ensure `${AOCL_ROOT}/include` and `${AOCL_ROOT}/lib` are visible.
- **OpenBLAS**: set `CMAKE_PREFIX_PATH` (or `OpenBLAS_ROOT`) to your OpenBLAS install.

You can also set `BLA_VENDOR` or adjust `CMAKE_PREFIX_PATH` to steer
`FindBLAS`/`FindLAPACK`.

## External HPTT

To use a system-installed HPTT:

```bash
cmake .. \
  -DQLTEN_COMPILE_HPTT_LIB=OFF \
  -Dhptt_INCLUDE_DIR=/path/to/hptt/include \
  -Dhptt_LIBRARY=/path/to/hptt/lib/libhptt.a
```

Alternatively, set `CMAKE_PREFIX_PATH` to a prefix that contains `include/` and `lib/`.

## GPU toolchain (tests)

If you enable `QLTEN_USE_GPU=ON`, CMake expects:

- CUDA Toolkit (11.0+)
- cuBLAS and cuSOLVER (bundled with CUDA)
- cuTENSOR (separate package)

TensorToolkit's cuTENSOR finder now supports both NVHPC-provided cuTENSOR and
user-local installs. If auto-discovery fails, set one of:

- `CUTENSOR_ROOT`
- `CUTENSOR_INCLUDE_DIR` + `CUTENSOR_LIBRARY`
- `CUDAToolkit_ROOT` for non-standard CUDA installations

Examples:

- NVHPC cluster: `CUTENSOR_ROOT=/opt/nvidia/hpc_sdk/.../math_libs/<cuda-version>/targets/x86_64-linux`
- local install: `CUTENSOR_ROOT=$HOME/.local/usr`
- GPU arch override: `QLTEN_CUDA_ARCHITECTURES=90` for H200/Hopper, or a semicolon-separated list for multi-target builds

Notes:

- The headers are gated by the compile-time macro `USE_GPU`. In this repo,
  `QLTEN_USE_GPU=ON` sets `-DUSE_GPU=1` for the test targets. Downstream
  projects should prefer the installed-package flow with
  `find_package(TensorToolkit CONFIG REQUIRED)` and
  `target_link_libraries(... PRIVATE TensorToolkit::TensorToolkit)` rather than
  defining `USE_GPU` and wiring CUDA/cuTENSOR manually. See
  `docs/user/en/howto/enable-gpu.md`.

## Where to look in the code

- Core tensor types: `include/qlten/qltensor/`
- Tensor operations: `include/qlten/tensor_manipulation/`
- MPI operations: `include/qlten/mpi_tensor_manipulation/`
