# QuantumLiquids/TensorToolkit

TensorToolkit is a high-performance tensor operations library written in C++.
It serves as the foundation for tensor network algorithms (like Density Matrix Renormalization Group,
Project Entanglement Pair State)
and provides essential functionalities such as tensor contraction, transpose, and more.
The package is header-only.
___

## Functionality

TensorToolkit offers the following key features:

- [x] Abelian Quantum Numbers
- [x] MPI Parallelization
- [x] Grassmann Tensor network
- [x] CUDA Single Card

---

## Documentation

- User docs: `docs/user/en/README.md`
- Developer docs: `docs/dev/index.md`
- Doxygen landing page: `docs/mainpage.md`

## To-Do List
- [ ] CUDA Multi-Card Support

---

## Dependencies
TensorToolkit is primarily header-only (no binary installation step is required to use the headers).
This repo's CMake project can optionally build/install the bundled
[HPTT](https://github.com/springer13/hptt.git) library for fast CPU tensor transpose.

Many common workflows (tests, decompositions, contractions, and MPI/GPU variants) require external
dependencies:

- **C++ Compiler:** C++17 or later
- **Build System:** CMake 3.27 or newer
- **Parallelization:** MPI (headers required; tests link `MPI::MPI_CXX`)
- **Math Libraries:** BLAS and LAPACK (required for tensor manipulation and tests)
- **GPU Acceleration (optional):** CUDA compiler + CUDA toolkit (cuBLAS, cuSOLVER) + cuTENSOR
- **Testing Framework (optional):** GoogleTest

---
## Installation

### 1) Get the source

```bash
git clone https://github.com/QuantumLiquids/TensorToolkit.git
cd TensorToolkit
```

### 2) Build and install (CPU)

TensorToolkit is header-only. For CPU transpose, we depend on HPTT. The default build uses the bundled `external/hptt` and installs it along with headers.

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ \
         -DCMAKE_INSTALL_PREFIX=/path/to/install/location
make -j4 && make install
```

This installs TensorToolkit headers and HPTT into the given prefix. If you prefer an external HPTT, point CMake to it (see `external/hptt` docs) and set the relevant CMake options accordingly.

### 3) Build tests

To build and run the unit tests:

1. CPU test build (from `build` directory):
    ```bash
    cmake .. \
             -DQLTEN_BUILD_UNITTEST=ON \
             -DGTest_DIR=/path/to/googletest \
             -DHP_NUMERIC_USE_OPENBLAS=ON \
             -DQLTEN_USE_GPU=OFF
    make -j16
    ```

2. GPU test build (from `build` directory):
    ```bash
    cmake .. \
             -DQLTEN_BUILD_UNITTEST=ON \
             -DGTest_DIR=/path/to/googletest \
             -DHP_NUMERIC_USE_OPENBLAS=ON \
             -DQLTEN_USE_GPU=ON \
             -DCUTENSOR_ROOT=/path/to/cutensor
    make -j16
    ```
    Notes:
    - Select exactly one BLAS backend: `-DHP_NUMERIC_USE_OPENBLAS=ON` (or `..._MKL` / `..._AOCL`).
    - CMake test configuration requires MPI to be discoverable (`FindMPI`) for both CPU/GPU configurations.
    Tip: Use `${HOME}` instead of `~` when defining CMAKE path variables.

3. Run the tests:
    ```bash
    ctest
    ```

### 4) Build documentation

Install Doxygen (required) and Graphviz (optional, for diagrams) (macOS):

```bash
brew install doxygen graphviz
```

Generate HTML docs:

```bash
cd docs
doxygen Doxyfile
open build/html/index.html
```

User-facing docs live under `docs/user/en/` (tutorials, how-to, explanation, reference).
Developer docs live under `docs/dev/`, and the Doxygen landing page is `docs/mainpage.md`.

The documentation includes:
- **User docs**: tutorials, how-to guides, explanations, and reference pages
- **Developer docs**: architecture, design, setup, testing, and practices
- **API reference**: complete class and function documentation in Doxygen

All documentation is generated in `docs/build/html/` and can be viewed by opening `index.html`.

---
## Author

Hao-Xin Wang

For any inquiries or questions regarding the project,
you can reach out to Hao-Xin via email at wanghaoxin1996@gmail.com.

---
## Acknowledgments

TensorToolkit is built upon the foundation laid by the [GraceQ/tensor](https://tensor.gracequantum.org) project.
While initially inspired by GraceQ/tensor,
TensorToolkit expands upon its capabilities by adding additional basic tensor operations, improving performance, and most
importantly, introducing support for MPI parallelization.
We would like to express our gratitude to the following individuals for their contributions and guidance:

- Rong-Yang Sun, the author of [GraceQ/tensor](https://tensor.gracequantum.org), for creating the initial framework that
  served as the basis for TensorToolkit.
- Yi-Fan Jiang, providing me with extensive help and guidance in writing parallel DMRG
- Hong Yao, my PhD advisor. His encouragement and continuous support
  of computational resources played crucial roles in the implementation of parallel DMRG in TensorToolkit.
- Zhen-Cheng Gu, my postdoc advisor, one of the pioneers in the field of tensor network.

Their expertise and support have been invaluable in the development of TensorToolkit.

---
## License

TensorToolkit is released under the LGPL3 License. Please see the LICENSE file for more details.
