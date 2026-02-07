# Distributed SVD (MPI)

TensorToolkit provides an MPI-enabled SVD for block-sparse tensors.
The master rank performs the orchestration, while the other ranks participate as workers.

## Headers

```cpp
#include "qlten/qlten.h"
#include "qlten/mpi_tensor_manipulation/ten_decomp/mpi_svd.h"
```

## Minimal pattern

```cpp
using namespace qlten;
namespace qn = qlten::special_qn;

// Assume MPI is initialized and you have a communicator
MPI_Comm comm = MPI_COMM_WORLD;
int rank = 0;
MPI_Comm_rank(comm, &rank);

QLTensor<double, qn::U1QN> T({/* indices */});
T.Random(qn::U1QN::Zero());

QLTensor<double, qn::U1QN> U, Vt;
QLTensor<double, qn::U1QN> S;

double err = 0.0;
size_t kept = 0;

if (rank == 0) {
  MPISVDMaster(&T, /*ldims=*/2, qn::U1QN::Zero(),
               /*trunc_err=*/1e-8, /*Dmin=*/1, /*Dmax=*/512,
               &U, &S, &Vt, &err, &kept, comm);
} else {
  MPISVDSlave<double>(comm);
}
```

## Notes

- Call `MPISVDSlave` on all non-master ranks at the same time the master calls `MPISVDMaster`.
- The interface mirrors the CPU `SVD` API; see the API reference for details.
- `U`, `S`, and `Vt` must be default-constructed (empty) on entry.

## Related docs

- [Contractions and decompositions](../tutorials/contractions-and-decompositions.md)
- [MPI parallel basics](../tutorials/mpi-parallel-basics.md)
