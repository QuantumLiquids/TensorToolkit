# TensorToolkit Documentation

**TensorToolkit** is a high-performance C++ library for tensor network algorithms, providing essential functionalities such as tensor contraction, SVD decomposition, and MPI parallelization. It's designed for quantum many-body calculations and tensor network methods like DMRG, PEPS, and TRG.

## 📖 Prerequisites knowledge

- **Basic tensor-network**, including concept of symmetry, fermionic tensor-network, and their manipulations.
- **Basic Modern C++**
- **Basic MPI** for parallel computing (optional)
- **Basic CUDA** for GPU acceleration (optional)

## 🚀 Quick Start

**New to TensorToolkit? Start here:**

1. **[Installation Guide](tutorials/01_installation.html)** - Build and install TensorToolkit
2. **[Quick Start Tutorial](tutorials/02_quick_start.html)** - Your first tensor program (TRG algorithm example)
3. **[Indices and Quantum Numbers](tutorials/03_indices_and_quantum_numbers.html)** - Understanding tensor structure and symmetry

## 📚 User Tutorials

### Core Tensor Operations
- **[Tensor Creation](tutorials/04_tensor_creation.html)** - Creating tensors with quantum numbers and indices
- **[Tensor Unary Operations](tutorials/05_tensor_unary_operations.html)** - Transpose, reshape, and element-wise operations
- **[Ising Vertex Tensor](tutorials/06_ising_vertex_tensor.html)** - Building the Ising model vertex tensor

### Advanced Operations
- **[Tensor Contractions](tutorials/07_tensor_contractions.html)** - Contraction, linear combination, and index fusion
- **[Matrix Decomposition](tutorials/08_matrix_decomposition.html)** - SVD, EVD, QR decomposition methods
- **[Complete Ising TRG](tutorials/09_ising_trg_example.html)** - Full Tensor Renormalization Group implementation

### Parallel Computing
- **[MPI Basics](tutorials/10_mpi_basics.html)** - Basic MPI operations for tensors
- **[MPI Advanced](tutorials/11_mpi_advanced.html)** - Distributed SVD and advanced parallel operations
- **[GPU Acceleration](tutorials/12_gpu_setup.html)** - CUDA backend setup and usage

## 🔧 Developer Resources

### API Reference
- **[Core API](api/core.html)** - QLTensor, Index, and quantum number classes
- **[Operations API](api/operations.html)** - Tensor manipulation functions
- **[MPI API](api/mpi.html)** - Parallel computing functions
- **[GPU API](api/gpu.html)** - CUDA acceleration functions

### Development
- **[Developer Guide](dev/index.html)** - Architecture, design principles, and contribution guidelines
- **[Testing Guide](dev/testing.html)** - How to run tests and contribute code
- **[Performance Guide](dev/performance.html)** - Optimization and benchmarking

## 🎯 Key Features

- **Abelian Quantum Numbers** - U(1), Z₂, and custom symmetries
- **Fermion/Boson Support** - Automatic sign handling for fermionic systems
- **MPI Parallelization** - Distributed tensor operations
- **GPU Acceleration** - CUDA backend with cuTENSOR
- **High Performance** - Optimized BLAS/LAPACK integration
- **Header-Only** - Easy integration into existing projects

## 🔗 External Resources

- **[GitHub Repository](https://github.com/QuantumLiquids/TensorToolkit)**
- **[Issue Tracker](https://github.com/QuantumLiquids/TensorToolkit/issues)**
- **[Research Papers](https://arxiv.org/abs/1704.04374)** - HPTT performance analysis

---

*Need help? Check the [tutorials](tutorials/index.html) or open an issue on GitHub.*


