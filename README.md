# QuantumLiquids/TensorToolkit

TensorToolit is a high-performance tensor basic operation library written in C++.
It serves as the foundation for tensor network algorithms (like Density Matrix Renormalization Group,
Project Entanglement Pair State)
and provides essential functionalities such as tensor contraction, transpose, and more.
The package is header-only.
___

## Functionality

TensorToolit offers the following key features:

- [x] Abelian Quantum Numbers
- [x] MPI Parallelization
- [x] Grassmann Tensor network
- [x] CUDA Single Card

## To-Do List
- [ ] CUDA Multi-Card Support

---

## Dependencies
TensorToolkit itself is header-only and requires no installation dependencies. To install the CPU code,
It needs to compile the [HPTT](https://github.com/springer13/hptt.git) for CPU-based tensor transpose operation, 
while HPTT only requires a basic C++ compiler and CMake. 

To build test cases or develop programs like *TRG*, *DMRG* or *TDVP* based on TensorToolkit, the following are required:

- **C++ Compiler:** C++17 or later
- **Build System:** CMake 3.12 or newer
- **Math Libraries:** BLAS and LAPACK
- **Parallelization:** MPI
- **GPU Acceleration (optional):** CUDA compiler, CUDA toolkit (cuBLAS, cuSolver, cuTensor2)
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

1. From the `build` directory:
    ```bash
    cmake .. -DQLTEN_BUILD_UNITTEST=ON \
             -DGTest_DIR=/path/to/googletest \
             -DQLTEN_USE_GPU=ON/OFF \
             -DCUTENSOR_ROOT=/path/to/cutensor/if/using/cuda
    make -j16
    ```
    Tip: Use `${HOME}` instead of `~` when defining CMAKE path variables.

2. Run the tests:
    ```bash
    ctest
    ```

### 4) Build documentation

Install Doxygen and Graphviz (macOS):

```bash
brew install doxygen graphviz
```

Generate HTML docs:

```bash
cd docs
doxygen Doxyfile
open build/html/index.html
```

The high-level user guides are under `docs/tutorials/` and appear in the generated HTML as "Tutorials".

The documentation includes:
- **User Tutorials**: Progressive learning path from installation to advanced features
- **Developer Guide**: Architecture, design, and contribution guidelines  
- **API Reference**: Complete class and function documentation

All documentation is generated in `docs/build/html/` and can be viewed by opening `index.html`.

---
## Author

Hao-Xin Wang

For any inquiries or questions regarding the project,
you can reach out to Hao-Xin via email at wanghaoxin1996@gmail.com.

---
## Acknowledgments

TensorToolit is built upon the foundation laid by the [GraceQ/tensor](https://tensor.gracequantum.org) project.
While initially inspired by GraceQ/tensor,
TensorToolit expands upon its capabilities by adding additional basic tensor operations, improving performance, and most
importantly, introducing support for MPI parallelization.
We would like to express our gratitude to the following individuals for their contributions and guidance:

- Rong-Yang Sun, the author of [GraceQ/tensor](https://tensor.gracequantum.org), for creating the initial framework that
  served as the basis for TensorToolit.
- Yi-Fan Jiang, providing me with extensive help and guidance in writing parallel DMRG
- Hong Yao, my PhD advisor. His encouragement and continuous support
  of computational resources played crucial roles in the implementation of parallel DMRG in TensorToolit.
- Zhen-Cheng Gu, my postdoc advisor, one of the pioneers in the field of tensor network.

Their expertise and support have been invaluable in the development of TensorToolit.

---
## License

TensorToolit is released under the LGPL3 License. Please see the LICENSE file for more details.