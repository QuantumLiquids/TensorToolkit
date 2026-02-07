# Runtime Environment

Many TensorToolkit workflows link to external BLAS/LAPACK and MPI libraries at
runtime (especially tests, decompositions, and distributed operations). This
page summarizes what must be available on your system to run such binaries.

Note: core tensor headers currently include `mpi.h`, so MPI headers are also
required at compile time.

## Required runtime components (when linked)

- **MPI runtime** (OpenMPI, MPICH, or vendor MPI) when your binary links MPI
- **BLAS/LAPACK** implementation (MKL, AOCL, OpenBLAS, or equivalent) when your
  binary links BLAS/LAPACK

## Optional runtime components

- **CUDA libraries** when GPU support is enabled
  - cuBLAS, cuSOLVER, cuTENSOR

## Common environment variables

- `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) to locate shared libraries
- `OMP_NUM_THREADS` to control OpenMP threading (if your BLAS uses OpenMP)

## AOCL note

If you build against AOCL, ensure the BLIS and FLAME shared libraries are
visible at runtime, for example:

```
export LD_LIBRARY_PATH=$AOCL_ROOT/lib:$LD_LIBRARY_PATH
```

## Troubleshooting

- If tests fail with missing symbols, verify your BLAS/LAPACK runtime matches
  the headers used at compile time.
- If MPI binaries fail to launch, check that `mpirun` uses the same MPI
  installation you compiled with.
