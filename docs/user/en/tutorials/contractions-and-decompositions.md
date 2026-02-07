# Contractions and Decompositions

This tutorial shows how to:

- contract two tensors with the general `Contract` API
- perform a symmetry-aware SVD for truncation

It is written as a working pattern you can copy into your own code.

## Contraction: tensor network building block

Use the pointer-based `Contract` API for general contractions.
It contracts specified axes and produces a new tensor with the remaining indices.

```cpp
#include "qlten/qlten.h"
#include "qlten/tensor_manipulation/ten_ctrct.h"

using namespace qlten;
namespace qn = qlten::special_qn;

// Build two compatible tensors A(l, m) and B(m^dagger, r)
QNSector<qn::Z2QN> s0(qn::Z2QN(0), 1), s1(qn::Z2QN(1), 1);
Index<qn::Z2QN> l({s0, s1}, TenIndexDirType::IN);
Index<qn::Z2QN> m({s0, s1}, TenIndexDirType::OUT);
Index<qn::Z2QN> r({s0, s1}, TenIndexDirType::OUT);

QLTensor<double, qn::Z2QN> A({l, m});
QLTensor<double, qn::Z2QN> B({InverseIndex(m), r});
A.Random(qn::Z2QN(0));
B.Random(qn::Z2QN(0));

QLTensor<double, qn::Z2QN> C; // must be default (empty)
Contract(&A, &B, {{1}, {0}}, &C); // C has indices [l, r]
```

Notes:

- Contracted indices must be inverse of each other.
- `C` must be default-constructed before calling `Contract`.
- Result ordering is: remaining indices of `A`, then remaining indices of `B`.

Contraction requirements and validation:

- Axes are **0-based**.
- The order of pairs in `axes_set` defines which axes are paired; the order
  of pairs does not change the logical result, but it must be consistent.
- `axes_set` must contain two lists of the same length (one for `A`, one for `B`).
- Each axis index must be in range for its tensor.
- For each pair, the full index must match `InverseIndex` (direction + QN sectors),
  not just the raw dimension.
- In debug builds, a mismatch triggers an `assert` with detailed diagnostics.
  In release builds, mismatches are not validated and can lead to undefined behavior.

## Decomposition: symmetry-aware SVD

TensorToolkit provides a block-sparse SVD that respects symmetry sectors.
The API accepts a bipartition (left dims vs right dims) and truncation options.

```cpp
#include "qlten/tensor_manipulation/ten_decomp/ten_svd.h"

QLTensor<double, qn::U1QN> T({/* indices */});
T.Random(qn::U1QN::Zero());

QLTensor<double, qn::U1QN> U, Vt;
QLTensor<double, qn::U1QN> S; // note: S is real-valued

double err = 0.0;
size_t kept = 0;

// Split after the first 2 indices
SVD(&T, /*ldims=*/2, qn::U1QN::Zero(),
    /*trunc_err=*/1e-8, /*Dmin=*/1, /*Dmax=*/512,
    &U, &S, &Vt, &err, &kept);
```

Typical workflow:

1. Contract the network into a single tensor (or pair of tensors).
2. Reshape/split into left and right index groups using `ldims`.
3. Perform SVD with truncation to control bond dimension.
4. Reassemble tensors using `U`, `S`, and `Vt`.

## Where to go next

- [MPI parallel basics](mpi-parallel-basics.md)
- [Matrix decomposition concepts (explanation)](../explanation/matrix-decomposition-concepts.md)
- [API reference](../reference/api-reference.md)
