# Testing Strategy

TensorToolkit uses GoogleTest and CMake. Tests are organized by domain:

- `tests/test_qltensor/` for core tensor types
- `tests/test_tensor_manipulation/` for contractions and decompositions
- `tests/test_tensor_mpi/` for MPI-enabled operations
- `tests/test_utility/` for utility helpers

## Running tests

```bash
ctest --output-on-failure
```

Run a single test:

```bash
ctest -R test_ten_svd --output-on-failure
```

## MPI tests

MPI tests should be run under `mpirun` or with CTest. The test CMake specifies:

- `test_ten_mpi_basic`: 2 ranks
- `test_mpi_svd`: 3 ranks

Manual runs (from the build directory):

```bash
mpirun -np 2 ./tests/test_ten_mpi_basic
mpirun -np 3 ./tests/test_mpi_svd
```

On a workstation, OpenMPI may require oversubscribe for quick experiments:

```bash
OMPI_MCA_rmaps_base_oversubscribe=1 ctest -R test_mpi --output-on-failure
```

## What we prioritize

- Correctness of symmetry-aware block structures
- Fermionic sign handling
- Consistency between CPU and MPI implementations

## Coverage snapshot

See [Coverage report](coverage-report.md) for the latest recorded snapshot.
