# MPI Parallel Basics

TensorToolkit is built with MPI support and provides MPI-enabled tensor
operations for distributed workloads.

## What this tutorial covers

- How MPI fits into TensorToolkit
- How to run MPI-enabled binaries
- When to use MPI-specific APIs

## MPI in TensorToolkit

MPI is required to build and run MPI-enabled binaries and to run TensorToolkit's
unit tests. Core tensor headers also require MPI headers at compile time today
(because `qlten/framework/hp_numeric/mpi_fun.h` includes `mpi.h`).

MPI-specific functionality lives under:

- `include/qlten/mpi_tensor_manipulation/`

## Running an MPI program

If you compile a TensorToolkit program with MPI and want to run across multiple
processes, use `mpirun` or `mpiexec`:

```bash
mpirun -np 4 ./your_program
```

On a workstation, you may need to enable oversubscribe for quick experiments:

```bash
OMPI_MCA_rmaps_base_oversubscribe=1 mpirun -np 4 ./your_program
```

## When to use MPI APIs

Use MPI-specific APIs when the operation is explicitly distributed, such as:

- distributed SVD (`MPISVDMaster` / `MPISVDSlave`)
- multi-process tensor decompositions

If you are only using local tensor operations, standard (non-MPI) APIs are enough.

## Next steps

- [Distributed SVD (MPI)](../howto/distributed-svd.md)
- [Contractions and decompositions](contractions-and-decompositions.md)
