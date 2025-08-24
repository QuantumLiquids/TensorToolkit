# TensorToolkit Tutorials

Welcome to the TensorToolkit learning path! These tutorials are designed to take you from installation to implementing advanced tensor network algorithms.

## 🎯 Learning Path

### 🚀 Getting Started (Essential)
1. **[Installation Guide](01_installation.html)** - Set up TensorToolkit on your system
2. **[Quick Start Tutorial](02_quick_start.html)** - Your first tensor program (TRG algorithm)
3. **[Indices and Quantum Numbers](03_indices_and_quantum_numbers.html)** - Understanding tensor structure and symmetry

### 📚 Core Tensor Operations (Foundation)
4. **[Tensor Creation](04_tensor_creation.html)** - Creating tensors with quantum numbers and indices
5. **[Tensor Unary Operations](05_tensor_unary_operations.html)** - Transpose, reshape, and element-wise operations
6. **[Ising Vertex Tensor](06_ising_vertex_tensor.html)** - Building the Ising model vertex tensor

### ⚡ Advanced Operations (Performance)
7. **[Tensor Contractions](07_tensor_contractions.html)** - Contraction, linear combination, and index fusion
8. **[Matrix Decomposition](08_matrix_decomposition.html)** - SVD, EVD, QR decomposition methods
9. **[Ising TRG Algorithm](09_ising_trg_algorithm.html)** - Levin-Nave Tensor Renormalization Group structure
10. **[Complete Ising TRG](09_ising_trg_example.html)** - Full Tensor Renormalization Group implementation

### 🔬 Parallel Computing (Scaling)
10. **[MPI Basics](10_mpi_basics.html)** - Basic MPI operations for tensors
11. **[MPI Advanced](11_mpi_advanced.html)** - Distributed SVD and advanced parallel operations
12. **[GPU Acceleration](12_gpu_setup.html)** - CUDA backend setup and usage

### 📐 Mathematical Foundations (Optional)
13. **[Z2-Graded Tensor Network](13_z2_graded_tensor_network.md)** - Mathematical foundations for fermionic TNs

## 📋 Prerequisites

- **C++17** or later compiler
- **Basic linear algebra** knowledge
- **Understanding of quantum mechanics** (helpful but not required)
- **MPI** for parallel computing tutorials (optional)
- **CUDA** for GPU tutorials (optional)

## 🎓 What You'll Learn

After completing these tutorials, you'll be able to:

- **Create and manipulate** symmetry-blocked sparse tensors
- **Handle quantum numbers** for bosonic and fermionic systems
- **Perform tensor operations** like contraction, SVD, and EVD
- **Scale computations** using MPI parallelization
- **Accelerate with GPUs** using CUDA backends
- **Implement algorithms** like TRG and DMRG
- **Optimize performance** for your specific use case

## 🚀 Quick Start Example

Here's a minimal example to get you excited:

```cpp
#include "qlten/qlten.h"
using namespace qlten;

// Create a simple U(1) symmetric tensor
auto qn0 = U1QN(0);
Index<U1QN> idx({QNSector<U1QN>(qn0, 2)}, TenIndexDirType::IN);
QLTensor<double, U1QN> tensor({idx, idx});
tensor.Random(U1QN(0)); // Random data with divergence 0
```

## 🔗 Next Steps

- **Complete the tutorials** in order for best results
- **Check the API reference** for detailed function documentation
- **Explore examples** in the test suite
- **Join the community** on GitHub

---

*Need help? Check the [main documentation](../index.html) or open an issue on GitHub.*


