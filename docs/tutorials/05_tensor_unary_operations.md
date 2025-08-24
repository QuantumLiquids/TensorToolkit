### Tensor Unary Operations

This tutorial summarizes common in-place and return-a-new-tensor unary operations on `QLTensor`. Construction and initialization are covered in the previous section.

### What you'll learn

- Basic queries and norms (`Get2Norm`, `GetQuasi2Norm`, `HasNan`, `GetMaxAbs`)
- Scaling and negation (`operator-`, `operator*`, `operator*=`)
- Conjugation and fermionic utilities (`Dag`, `ActFermionPOps`)
- Element-wise transforms (inverse/sqrt/square/sign/bound/randomize-magnitude)
- Structural transforms (`Transpose`, `FuseIndex`, `RemoveTrivialIndexes`)
- Diagonal-matrix helpers (`DiagMatInv`)

## 1. Setup

```cpp
#include "qlten/qlten.h"
#include "qlten/qltensor/special_qn/special_qn_all.h"

using qlten::QLTensor;
namespace special_qn = qlten::special_qn;
```

Assume `QLTensor<double, special_qn::U1QN> A = /* ... */;` is a valid tensor.

## 2. Queries and norms

For Z2-graded (fermionic) tensors, elements are partitioned into even blocks \(E\) and odd blocks \(O\):

- **Bosonic 2-norm**: \( \|A\|_2 = \sqrt{\sum_i |a_i|^2} \).
- **Fermionic graded 2-norm**: 
  \[ \|A\|_{2,\mathrm{graded}} = \sqrt{\sum_{i\in E} |a_i|^2\; - \; \sum_{i\in O} |a_i|^2}. \]
  This can be ill-defined if the odd contribution exceeds the even one.
- **Quasi 2-norm** (always non-negative, both boson/fermion):
  \[ \|A\|_{2,\mathrm{quasi}} = \sqrt{\sum_i |a_i|^2}. \]

Accordingly: `Normalize()` uses the graded 2-norm (`Get2Norm()`), while `QuasiNormalize()` uses the quasi 2-norm (`GetQuasi2Norm()`). For fermionic workflows, prefer quasi normalization when you need a conventional magnitude.

```cpp
double n2 = A.Get2Norm();        // possible ill-defined for fermion tensor
double qn2 = A.GetQuasi2Norm();  // always well-defined
bool any_nan = A.HasNan();
double max_abs = A.GetMaxAbs();

double prev = A.Normalize();     // Normalize by  Get2Norm()
A.QuasiNormalize();              // Normalize by GetQuasi2Norm()
```

## 3. Scaling and negation

```cpp
auto B = -A;          // unary minus, returns a copy
auto C = 2.0 * A;     // scalar-tensor product (also A * 2.0)
A *= 0.5;             // in-place scaling
```

## 4. Conjugation and fermionic ops

```cpp
A.Dag();              // Hermitian conjugate: conjugate elements and flip index directions
A.ActFermionPOps();   // Apply parity operators on fermionic legs 
```

## 5. Element-wise transforms

```cpp
A.ElementWiseInv();           // reciprocal in-place
A.ElementWiseInv(1e-12);      // with tolerance for near-zeros
A.ElementWiseSqrt();
A.ElementWiseSquare();
A.ElementWiseSign();          // real: {-1,0,1}; complex: preserve phase, unit magnitude
A.ElementWiseClipTo(1.0);     // clip magnitude to limit (phase preserved for complex)

// Randomize magnitudes in-place but keep original signs/phases
std::mt19937 rng(123);
std::uniform_real_distribution<double> dist(0.0, 1.0);
A.ElementWiseRandomizeMagnitudePreservePhase(dist, rng);
```

## 6. Structural transforms

```cpp
A.Transpose({1,0,2});     // reorder axes by permutation
A.FuseIndex(0,1);         // fuse adjacent indices (left_axis, right_axis)
A.RemoveTrivialIndexes(); // drop all dim-1 indices
```

## 7. Diagonal matrix utilities

For a tensor representing a diagonal matrix (two indices that are the same up to direction):

```cpp
A.DiagMatInv();        // element-wise inverse on the diagonal
A.DiagMatInv(1e-12);   // tolerant version
```

**References**
  - Kraus, C. V., Schuch, N., Verstraete, F., & Cirac, J. I. (2010). Fermionic projected entangled pair states. *Phys. Rev. A*, 81(5), 052338. https://doi.org/10.1103/PhysRevA.81.052338
  - I. Pižorn and F. Verstraete, "Fermionic implementation of projected entangled pair states algorithm," Phys. Rev. B 81, 245110 (2010). https://link.aps.org/doi/10.1103/PhysRevB.81.245110

—

Next: Tensor Contractions.
