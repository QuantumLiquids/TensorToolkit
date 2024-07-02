# QuantumLiquids/TensorToolkit

TensorToolit is a high-performance tensor basic operation library written in C++.
It serves as the foundation for tensor network algorithms (like Density Matrix Renormalization Group,
Project Entanglement Pair State)
and provides essential functionalities such as tensor contraction, transpose, and more.

## Functionality

TensorToolit offers the following key features:

- [x] Abelian Quantum Numbers
- [x] MPI Parallelization Support
- [x] Grassmann Tensor network

## To-Do List

- [ ] GPU

## Dependence

Please note that the project requires the following dependencies
to be installed in order to build and run successfully:

- C++17 Compiler
- CMake (version 3.12 or higher)
- Intel MKL or OpenBlas
- MPI
- Boost::serialization, Boost::mpi (version 1.74 or higher)
- GoogleTest (if testing is required)

## Install

Clone the repository into a desired directory and change into that location:

```
git clone https://github.com/QuantumLiquids/TensorToolit.git
cd TensorToolit
```

Using CMake:

```
mkdir build && cd build
cmake .. 
make -j4 && make install
```

You may want to specify `CMAKE_CXX_COMPILER` as your favorite C++ compiler,
and `CMAKE_INSTALL_PREFIX` as your install directory when you're calling `cmake`

## Author

Hao-Xin Wang

For any inquiries or questions regarding the project,
you can reach out to Hao-Xin via email at wanghaoxin1996@gmail.com.

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

## License

TensorToolit is released under the LGPL3 License. Please see the LICENSE file for more details.