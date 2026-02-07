# Indices and Quantum Numbers

TensorToolkit uses quantum-number (QN) labels to represent symmetry constraints.
Instead of storing a full dense tensor, it organizes data into symmetry-allowed
blocks, which can be much smaller and faster.

## Core ideas

- **Quantum number (QN)**: a label that encodes a symmetry charge (for example U(1) charge or Z2 parity).
- **Sector**: a subspace on an index with a specific QN and a degeneracy (dimension).
- **Index**: a list of sectors plus a direction (IN or OUT).
- **Divergence**: the signed sum of QNs over all indices in a tensor (OUT adds, IN subtracts).

Only blocks consistent with the requested divergence are stored.

## A concrete example

```cpp
#include "qlten/qlten.h"

using namespace qlten;
namespace qn = qlten::special_qn;

QNSector<qn::U1QN> s0(qn::U1QN(0), 1);
QNSector<qn::U1QN> s1(qn::U1QN(1), 2);
Index<qn::U1QN> idx({s0, s1}, TenIndexDirType::OUT);

QLTensor<double, qn::U1QN> T({idx, InverseIndex(idx)});
T.Random(qn::U1QN::Zero());
```

This creates a rank-2 tensor whose blocks respect total charge 0.

## Direction conventions

TensorToolkit uses a simple convention:

- `IN` corresponds to ket-like legs
- `OUT` corresponds to bra-like legs

When you flip an index direction, its QN effectively changes sign in the divergence sum.

## Built-in QNs

The most common QN types live in `qlten::special_qn`, including:

- `U1QN` (bosonic U(1))
- `Z2QN` (parity)
- `fZ2QN`, `fU1QN`, `fZ2U1QN` (fermionic variants)

Choose a fermionic QN type when you want fermionic sign handling in contractions.

## Next steps

- [Symmetry and block sparsity](symmetry-and-block-sparsity.md)
- [Z2-graded fermions](z2-graded-fermions.md)
- [Tensor basics walkthrough](../tutorials/tensor-basics-walkthrough.md)
