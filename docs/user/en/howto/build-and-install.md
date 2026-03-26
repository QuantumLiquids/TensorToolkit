# Build and Install

TensorToolkit installs as a CMake package. Downstream projects should consume
the installed package with `find_package(TensorToolkit CONFIG REQUIRED)` and
link the imported target instead of manually propagating include directories or
backend definitions.

## Prerequisites

- C++17 compiler
- CMake 3.27+
- MPI development files
- One CPU BLAS/LAPACK backend for CPU installs, or the CUDA toolchain plus
  cuTENSOR for GPU installs

On the cluster nodes used for this repository, source the production
environment before configuring:

```bash
source ~/myenv.sh
```

That environment should provide compiler selection and dependency hints such as
`CC`, `CXX`, `MKLROOT`, `AOCL_ROOT`, `CUDAToolkit_ROOT`, and cuTENSOR paths when
needed.

## Install a CPU package

Use a dedicated install prefix for each variant. A CPU install should not share
the same prefix as a GPU install.

```bash
git clone https://github.com/QuantumLiquids/TensorToolkit.git
cd TensorToolkit

cmake -S . -B build/package-cpu \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=/path/to/tensortoolkit-cpu \
  -DQLTEN_USE_GPU=OFF \
  -DQLTEN_COMPILE_HPTT_LIB=ON \
  -DQLTEN_BUILD_UNITTEST=OFF \
  -DQLTEN_BUILD_EXAMPLES=OFF \
  -DHP_NUMERIC_USE_MKL=ON

cmake --build build/package-cpu -j4
cmake --install build/package-cpu
```

If MKL is not available, replace `-DHP_NUMERIC_USE_MKL=ON` with
`-DHP_NUMERIC_USE_AOCL=ON` or `-DHP_NUMERIC_USE_OPENBLAS=ON`.

When `QLTEN_COMPILE_HPTT_LIB=ON`, the CPU install also installs the bundled
HPTT dependency into the same prefix.

## Install a GPU package

GPU installs export a different public dependency set, so they should use a
different prefix.

```bash
source ~/myenv.sh

cmake -S . -B build/package-gpu \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=/path/to/tensortoolkit-gpu \
  -DQLTEN_USE_GPU=ON \
  -DQLTEN_BUILD_UNITTEST=OFF \
  -DQLTEN_BUILD_EXAMPLES=OFF

cmake --build build/package-gpu -j4
cmake --install build/package-gpu
```

GPU installs require `CUDAToolkit` and `CUTENSOR`. In this mode
`QLTEN_COMPILE_HPTT_LIB` is forced `OFF`.

To verify the installed-package surface from the configured build tree, run:

- CPU: `cmake --build build/package-cpu --target verify-package-cpu`
- GPU: `cmake --build build/package-gpu --target verify-package-gpu`

The GPU verifier configures, builds, and runs a downstream smoke consumer, so
it should be executed on a real GPU node. GitHub CI currently automates only
the CPU verifier.

## Installed package layout

The standard package metadata is installed under:

```text
<prefix>/${CMAKE_INSTALL_LIBDIR}/cmake/TensorToolkit
```

That directory contains `TensorToolkitConfig.cmake`,
`TensorToolkitConfigVersion.cmake`, `TensorToolkitTargets.cmake`, the package
dependency helper, and the copied `FindCUTENSOR.cmake` / `Findhptt.cmake`
modules used by the installed package.

Headers are installed under `<prefix>/include/qlten/`.

## Consume the installed package

Point CMake at the install prefix with `CMAKE_PREFIX_PATH`, or set
`TensorToolkit_DIR` directly to the package-config directory.

```cmake
find_package(TensorToolkit CONFIG REQUIRED)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE TensorToolkit::TensorToolkit)
```

Example configure command:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/tensortoolkit-cpu
```

The imported target carries the variant-specific public requirements:

- MPI for all installs
- CPU BLAS/LAPACK backend definitions and libraries for CPU installs
- OpenMP when required by the selected CPU backend or bundled HPTT
- external HPTT discovery when the package was installed with
  `QLTEN_COMPILE_HPTT_LIB=OFF`
- CUDA, cuBLAS, cuSOLVER, and cuTENSOR for GPU installs
- `QLTEN_TIMING_MODE` and `QLTEN_MPI_TIMING_MODE` when those options were
  enabled in the installed variant

Because those timing macros affect public headers, they are part of the package
variant properties and should not be toggled separately in downstream code.

If a CPU package was installed with `QLTEN_COMPILE_HPTT_LIB=OFF`, downstream
consumer configuration must also make HPTT discoverable, either by adding the
HPTT prefix to `CMAKE_PREFIX_PATH` or by setting `hptt_INCLUDE_DIR` and
`hptt_LIBRARY`.

## Compatibility variables during migration

The installed package also exposes temporary compatibility variables for
downstream repos that have not finished migrating:

- `QLTEN_HEADER_PATH`
- `QLTEN_USE_GPU`
- `TensorToolkit_USE_GPU`

These shims are transitional only. New downstream code should use
`TensorToolkit::TensorToolkit`.

## Next steps

- [Build and run tests](build-and-run-tests.md)
- [CMake options](../reference/cmake-options.md)
- [Quick start tutorial](../tutorials/quick-start-trg.md)
