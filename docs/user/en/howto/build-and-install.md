# Build and Install

TensorToolkit is header-only, but it can build and install the bundled HPTT
library for CPU tensor transpose operations.

## Prerequisites

- C++17 compiler
- CMake 3.27+
- MPI headers (required to compile TensorToolkit headers today)
- BLAS/LAPACK (required for tensor manipulation, decompositions, and tests)

## Quick build and install

```bash
git clone https://github.com/QuantumLiquids/TensorToolkit.git
cd TensorToolkit
mkdir -p build
cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/install
make -j4
make install
```

This installs:

- TensorToolkit headers under `<prefix>/include/qlten/`
- HPTT (if `QLTEN_COMPILE_HPTT_LIB=ON`, which is the default)

## Common CMake options

- `-DQLTEN_COMPILE_HPTT_LIB=ON|OFF` to control building the bundled HPTT
- `-DQLTEN_BUILD_UNITTEST=ON|OFF` to enable tests
- `-DQLTEN_BUILD_EXAMPLES=ON|OFF` to build example programs
- `-DQLTEN_USE_GPU=ON|OFF` to enable GPU support (tests/examples)

See [CMake options](../reference/cmake-options.md) for the full list.

## BLAS/LAPACK backend selection (hp_numeric)

TensorToolkit routes BLAS/LAPACK calls through `backend_selector.h`, which
requires exactly one of these compile definitions:

- `HP_NUMERIC_BACKEND_MKL`
- `HP_NUMERIC_BACKEND_AOCL`
- `HP_NUMERIC_BACKEND_OPENBLAS`
- `HP_NUMERIC_BACKEND_KML` (Huawei Kunpeng Math Library; supported in headers but not wired via a CMake option here)

If you use this repo's CMake for tests or examples, include
`cmake/Modules/MathBackend.cmake` and set exactly one of:

- `-DHP_NUMERIC_USE_MKL=ON`
- `-DHP_NUMERIC_USE_AOCL=ON`
- `-DHP_NUMERIC_USE_OPENBLAS=ON`

`MathBackend.cmake` sets the matching `HP_NUMERIC_BACKEND_*` define and BLAS
include paths.

If you only build/install headers and HPTT (no tests/examples), you do not need
to select a backend. The backend macros are required when you compile code that
uses TensorToolkit's BLAS/LAPACK wrappers (for example, tests, examples, or your
own application code that includes tensor manipulation headers).

If you enable tests/examples and do not set `HP_NUMERIC_USE_*`, the module will
auto-select a backend based on CPU vendor. This can fail if MKL/AOCL environment
variables are not set, so explicit selection (often OpenBLAS) is recommended.

If you integrate TensorToolkit into your own project, define one
`HP_NUMERIC_BACKEND_*` macro and ensure your BLAS/LAPACK headers and libraries
are on the include/link paths.

## macOS: OpenMP not found

If CMake reports missing `OpenMP_CXX`, you are likely using Apple Clang.
Install LLVM via Homebrew and configure CMake to use it:

```bash
brew install llvm

cmake .. \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
```

If CMake still fails, rerun the exact same `cmake ..` command once.

## Next steps

- [Build and run tests](build-and-run-tests.md)
- [Quick start tutorial](../tutorials/quick-start-trg.md)
