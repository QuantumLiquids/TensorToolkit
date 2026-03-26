# Developer Documentation

This section targets contributors and maintainers. It is organized by document
**type** and by **module**.

## Assumed background

These docs assume familiarity with:

- Modern C++ (C++17, templates, and header-only libraries)
- Basic tensor-network concepts (indices, quantum numbers, and block sparsity):
  - [docs/user/en/explanation/indices-and-quantum-numbers.md](../user/en/explanation/indices-and-quantum-numbers.md)
  - [docs/user/en/explanation/symmetry-and-block-sparsity.md](../user/en/explanation/symmetry-and-block-sparsity.md)
- CMake-based C++ builds and linking against HPC libraries (BLAS/LAPACK, MPI):
  - [docs/dev/guides/dev-setup.md](guides/dev-setup.md)
  - [docs/user/en/reference/runtime-environment.md](../user/en/reference/runtime-environment.md)
- Fermionic/Z2-graded conventions if you touch fermion code paths:
  - [docs/user/en/explanation/z2-graded-fermions.md](../user/en/explanation/z2-graded-fermions.md)
  - [docs/dev/design/fermion-design.md](design/fermion-design.md)

## By document type
- Architecture and design: `docs/dev/design/`
- Guides: `docs/dev/guides/`
- Testing: `docs/dev/testing/`
- Practices and standards: `docs/dev/practices/`
- Module index: `docs/dev/modules/`
- Reference: `docs/dev/reference/`

## By module

- [QLTensor core](modules/qltensor.md)
- [Tensor manipulation](modules/tensor_manipulation.md)
- [MPI tensor manipulation](modules/mpi_tensor_manipulation.md)

## Start here

- [Architecture overview](design/architecture-overview.md)
- [CMake package architecture](design/2026-03-25-cmake-package-architecture.md)
- [Developer setup](guides/dev-setup.md)
- [Downstream CMake package migration](guides/2026-03-25-downstream-cmake-package-migration.md)
- [Coding standards](practices/coding-standards.md)
- [Testing strategy](testing/testing-strategy.md)
