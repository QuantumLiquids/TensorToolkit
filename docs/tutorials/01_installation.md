# Installation Guide

This tutorial will guide you through installing TensorToolkit on your system.

## Prerequisites

### Required Dependencies
- **C++17** or later compiler (GCC 7+, Clang 5+, or Intel ICC 17+)
- **CMake** 3.12 or newer
- **Git** for cloning the repository
- **BLAS and LAPACK** - Intel MKL (recommended for x86) or OpenBLAS (recommended for Arm64)
- **MPI** - OpenMPI, MPICH or IntelMPI for parallel computing (always required)

### Optional Dependencies
- **CUDA** - CUDA 11.0+ for GPU acceleration
- **GoogleTest** - Required if building tests

For x86 systems, we recommend using Intel oneAPI to provide the compiler, BLAS/LAPACK (MKL), and MPI.

## Installation 

### 1: Quick Install (Recommended)

TensorToolkit is header-only and requires HPTT for CPU tensor transpose operations. The default build uses the bundled `external/hptt` and installs it along with headers.

```bash
# Clone the repository
git clone https://github.com/QuantumLiquids/TensorToolkit.git
cd TensorToolkit

# Create build directory
mkdir build && cd build

# Configure with default options
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install/location

# Build and install
make -j$(nproc)
make install
```

> **Note:** Depending on your system configuration, you may need to run `make install` with `sudo` privileges to install TensorToolkit system-wide.

This installs TensorToolkit headers and HPTT into the specified prefix. If you wish to use an external HPTT installation instead of the bundled one, configure CMake with `-DQLTEN_COMPILE_HPTT_LIB=OFF` and the installation will skip the compiling.

### 2: Building Tests

If you want to build and run the test suite, GoogleTest is required:

```bash
# From the build directory
cmake .. -DQLTEN_BUILD_UNITTEST=ON \
         -DGTest_DIR=/path/to/googletest \
         -DQLTEN_USE_GPU=ON/OFF \
         -DCUTENSOR_ROOT=/path/to/cutensor/if/using/cuda

# Build tests
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

**Note:** When building tests with `-DQLTEN_BUILD_UNITTEST=ON`, you must also specify `-DGTest_DIR=/path/to/googletest` pointing to your GoogleTest installation.

## CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `QLTEN_USE_GPU` | `OFF` | Enable CUDA GPU support for tests  |
| `QLTEN_BUILD_UNITTEST` | `OFF` | Build test suite (requires GoogleTest) |
| `QLTEN_BUILD_EXAMPLES` | `OFF` | Build example programs |
| `QLTEN_COMPILE_HPTT_LIB` | `ON` | Compile HPTT library (disabled if using GPU) |
| `QLTEN_TIMING_MODE` | `ON` | Enable timing mode |
| `QLTEN_MPI_TIMING_MODE` | `ON` | Enable MPI timing mode |

**Important:** TensorToolkit always includes MPI support - there is no option to disable it since parallel computing is a core feature.

## Next Steps

After successful installation:

1. **[Quick Start Tutorial](02_quick_start.html)** - Write your first TensorToolkit program
2. **[Core Concepts](03_core_concepts.html)** - Learn about indices and quantum numbers
3. **[API Reference](../api/core.html)** - Explore the complete API documentation

---

*Installation complete? Move on to the [Quick Start Tutorial](02_quick_start.html)!*


## FAQ

### macOS: Could NOT find OpenMP_CXX
```bash
> cmake ..
CMake Error at /opt/homebrew/share/cmake/Modules/FindPackageHandleStandardArgs.cmake:233 (message):
  Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
Call Stack (most recent call first):
  /opt/homebrew/share/cmake/Modules/FindPackageHandleStandardArgs.cmake:603 (_FPHSA_FAILURE_MESSAGE)
  /opt/homebrew/share/cmake/Modules/FindOpenMP.cmake:616 (find_package_handle_standard_args)
  tests/CMakeLists.txt:61 (find_package)
```
If you meet the problem, you can check your clang version:
```bash
> clang --version
Apple clang version 17.0.0 (clang-1700.0.13.5)
Target: arm64-apple-darwin24.5.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```
If you see output like the above, you're using Apple's Clang, which lacks OpenMP support.
Install the LLVM toolchain via Homebrew as follows:
```bash
brew install llvm
```

Then set environment variables for LLVM:
```bash
echo 'export CC=/opt/homebrew/opt/llvm/bin/clang' >> ~/.zshrc
echo 'export CXX=/opt/homebrew/opt/llvm/bin/clang++' >> ~/.zshrc
echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"' >> ~/.zshrc
echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"' >> ~/.zshrc
source ~/.zshrc
```
Manually set this clang as compiler in CMake configuration.

## Appendix: CUDA Setup on HPC Cluster
The following CUDA libraries are required:
- **cuBLAS** - Basic linear algebra operations
- **cuSOLVER** - Linear system solvers
- **cuTENSOR** - Tensor operations (CUDA 11.0+)

These dependencies are included in the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk).
Ask your system administrator which module to load (for example, `nvhpc`).

Some clusters only provide the standard CUDA toolkit, which includes cuBLAS and cuSOLVER but not cuTENSOR. In that case, install cuTENSOR in your home directory from the official source: [cuTENSOR](https://developer.nvidia.com/cutensor).


##### Further help
Read the top-level `CMakeLists.txt`.
