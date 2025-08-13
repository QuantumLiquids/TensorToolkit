### Tensor Creation

This tutorial shows how to construct and initialize `QLTensor` consistently with the actual API (constructors, initialization, element access, and basic properties).

### What you'll learn

- Create `Index<QNT>` and `QLTensor<ElemT, QNT>` with supported quantum numbers
- Initialize data with symmetry-aware Fill/Random
- Set elements via dense coordinates or quantum-number sectors
- Query basic tensor properties and divergence
- Build common objects like identity/fermion-parity operators

## 1. Setup and types

```cpp
#include "qlten/qlten.h" // It include every thing.

using qlten::Index;
using qlten::QNSector;
using qlten::QLTensor;
using qlten::TenIndexDirType;
namespace special_qn = qlten::special_qn;
```
Or simply
```cpp
using namespace qlten;
using namespace qlten::special_qn;
```
Prefer the non-virtual, efficient quantum numbers under `qlten::special_qn`, for example `special_qn::U1QN`, `special_qn::Z2QN`, `special_qn::fU1QN`, `special_qn::fZ2U1QN`.

## 2. Building indices

```cpp
// U(1) sectors: q=0 (degeneracy 2), q=1 (degeneracy 3), q=2 (degeneracy 1)
QNSector<special_qn::U1QN> s0(special_qn::U1QN(0), 2);
QNSector<special_qn::U1QN> s1(special_qn::U1QN(1), 3);
QNSector<special_qn::U1QN> s2(special_qn::U1QN(2), 1);

Index<special_qn::U1QN> idx({s0, s1, s2}, TenIndexDirType::IN);
```

## 3. Constructing tensors

```cpp
// Scalar (rank-0): use an explicit empty IndexVec
QLTensor<double, special_qn::U1QN> scalar(qlten::IndexVec<special_qn::U1QN>{});

// Vector, matrix, and rank-3 examples
QLTensor<double, special_qn::U1QN> vec({idx});
QLTensor<double, special_qn::U1QN> mat({idx, qlten::InverseIndex(idx)});
QLTensor<double, special_qn::U1QN> ten3({idx, idx, qlten::InverseIndex(idx)});
```

Notes:
- Constructors accept `IndexVec<QNT>` by value or rvalue; brace-init with `{...}` works for rank ≥ 1.
- A default-constructed tensor is “default/empty” (no storage). Use the index constructors above to make a usable tensor.

## 4. Initialization (divergence-aware)

### Globally Initialization: Random/Fill
`QLTensor` stores only blocks consistent with a target divergence (charge/parity conservation). Use the API below:

```cpp
// Random values in [0,1), respecting divergence
mat.Random(special_qn::U1QN::Zero());

// Fill with a constant on all allowed blocks for a given divergence
mat.Fill(special_qn::U1QN::Zero(), 1.5);

// Scalar case: divergence must be zero
scalar.Fill(special_qn::U1QN::Zero(), 2.0);
```

### Element-wise initialization: operator()

You can also setting elements by coordinates directly. Blocks are created on demand.

```cpp
// Scalar element
scalar() = 2.0;

// Variadic index access for rank-2
mat(0, 1) = 3.14;
double v = mat(0, 1);

// Vector-form access
mat(std::vector<size_t>{0, 1}) = 2.0;

// Build an identity by hand
QLTensor<double, special_qn::U1QN> Id({idx, qlten::InverseIndex(idx)});
for (size_t i = 0; i < idx.dim(); ++i) {
  Id(i, i) = 1.0;
}
```

Note: manual element-wise initialization does not enforce a single divergence across blocks. 
Use `Fill(div, value)`/`Random(div)` when you require a consistent divergence.

### Identity and parity operators

```cpp
// Eye generates identity (bosonic) or fermion-parity operator (fermionic)
auto eye = qlten::Eye<double, special_qn::U1QN>(idx); // 2-index operator [idx, idx^†]
```

## 5. Element access

Two ways to set/get elements:

- Dense coordinates (across the full product space). Blocks are auto-created as needed.

```cpp
// Matrix-like access
mat({0,0}) = 3.14;
double v = mat({0,0});

// Generic coordinate vector
mat.SetElem(std::vector<size_t>{0,1}, 2.0);
auto val = mat.GetElem(std::vector<size_t>{0,1});
```

- By quantum-number sector plus in-block coordinates. This targets a specific symmetry block.

```cpp
// qn sector across indices (order matches tensor indices)
std::vector<special_qn::U1QN> qn_sector = {special_qn::U1QN(0), special_qn::U1QN(0)};
// coordinates inside that block (degeneracy coordinates)
std::vector<size_t> blk_coors = {0, 0};
mat.SetElemByQNSector(qn_sector, blk_coors, 1.0);
```

## 6. Properties and utilities

```cpp
size_t r = mat.Rank();                 // number of indices
size_t n = mat.size();                 // product of index dimensions
auto shape = mat.GetShape();           // vector of dimensions
bool is_scalar = mat.IsScalar();
bool is_default = mat.IsDefault();
auto div = mat.Div();                  // aggregated divergence (if blocks exist)
double norm2 = mat.Get2Norm();         // 2-norm 
double qnorm2 = mat.GetQuasi2Norm();   // bosonic-like 2-norm
size_t num_blocks = mat.GetQNBlkNum(); // number of symmetry blocks present
double mem_gb = mat.GetRawDataMemUsage(); // estimated raw data memory in GB

mat.Dag();                             // conjugate + flip index directions. Fermion sign will be account if fermion tensor
```

Advanced block introspection (optional):

```cpp
const auto &blkmap = mat.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
for (const auto &kv : blkmap) {
  const auto &data_blk = kv.second;           // contains block coors, shape, QN info
  (void)data_blk;
}
```

## 7. Copy/move semantics

```cpp
QLTensor<double, special_qn::U1QN> A({idx, qlten::InverseIndex(idx)});
A.Random(special_qn::U1QN::Zero());

QLTensor<double, special_qn::U1QN> B = A;  // copy-construct
QLTensor<double, special_qn::U1QN> C({idx, qlten::InverseIndex(idx)});
C = A;                                      // copy-assign

QLTensor<double, special_qn::U1QN> D = std::move(A); // move-construct
```

Type-changing “clone” is not provided. Create a new tensor with the target `ElemT` and copy data as needed.

## 8. Practical examples

### 8.1 Transfer matrix (Z2)

```cpp
QNSector<special_qn::Z2QN> e(special_qn::Z2QN(0), 1), o(special_qn::Z2QN(1), 1);
Index<special_qn::Z2QN> spin({e, o}, TenIndexDirType::IN);
QLTensor<double, special_qn::Z2QN> T({spin, spin});

double beta = 1.0;
T.SetElem({0,0}, std::exp(beta));
T.SetElem({1,1}, std::exp(beta));
T.SetElem({0,1}, std::exp(-beta));
T.SetElem({1,0}, std::exp(-beta));
```

### 8.2 Projection operator (U1)

```cpp
QNSector<special_qn::U1QN> up(special_qn::U1QN(1), 1), dn(special_qn::U1QN(-1), 1);
Index<special_qn::U1QN> sidx({up, dn}, TenIndexDirType::IN);
QLTensor<double, special_qn::U1QN> P({sidx, sidx});
P.Fill(special_qn::U1QN::Zero(), 0.0);
P.SetElem({0,0}, 1.0); // |↑><↑|
```

### 8.3 Random tensor respecting symmetry

```cpp
QLTensor<double, special_qn::U1QN> R({idx, qlten::InverseIndex(idx)});
R.Random(special_qn::U1QN::Zero());
```

## 9. Best practices

- Plan indices and symmetry layout up front
- Use `Fill(div, value)` and `Random(div)` to preserve symmetry
- Prefer `special_qn::*` quantum numbers for performance and clarity
- Use `InverseIndex(idx)` when building bra/ket pairs
- Avoid relying on internal data structures unless necessary

—

Ready to manipulate tensors? Continue with Tensor Operations.
