# Tensor Basics Walkthrough

This tutorial shows the core ideas behind TensorToolkit tensors: indices with quantum-number
sectors, symmetry-aware storage, and simple construction patterns.

## Goal

By the end you should be able to:

- Define indices with quantum-number sectors
- Create a tensor with those indices
- Initialize data consistently with symmetry
- Inspect basic properties

## Step 1: Include the headers

The umbrella header brings in the most common types:

```cpp
#include "qlten/qlten.h"

using namespace qlten;
namespace qn = qlten::special_qn;
```

## Step 2: Build indices with quantum-number sectors

Each index is a list of quantum-number sectors plus a direction (IN or OUT).
A sector has a quantum number and a degeneracy (dimension).

```cpp
QNSector<qn::U1QN> s0(qn::U1QN(0), 1); // charge 0, degeneracy 1
QNSector<qn::U1QN> s1(qn::U1QN(1), 2); // charge 1, degeneracy 2

Index<qn::U1QN> idx_out({s0, s1}, TenIndexDirType::OUT);
Index<qn::U1QN> idx_in = InverseIndex(idx_out); // OUT -> IN
```

## Step 3: Create a tensor

A tensor is built from an index list. This example is a rank-2 tensor:

```cpp
QLTensor<double, qn::U1QN> A({idx_out, idx_in});
```

## Step 4: Initialize data with a target divergence

Symmetry-aware tensors only store blocks consistent with a target divergence.
Use `Random` or `Fill` to create consistent block structure.

```cpp
A.Random(qn::U1QN::Zero()); // random values, divergence = 0
// or
A.Fill(qn::U1QN::Zero(), 1.0);
```

## Step 5: Inspect properties

```cpp
size_t rank = A.Rank();           // number of indices
auto shape = A.GetShape();        // dimensions (per index)
auto div = A.Div();               // divergence if blocks exist
size_t blocks = A.GetQNBlkNum();  // number of symmetry blocks
```

## Step 6: Build a simple operator

```cpp
auto eye = qlten::Eye<double, qn::U1QN>(idx_out); // identity operator
```

## Notes for beginners

- Direction matters: IN vs OUT controls sign when checking symmetry constraints.
- For fermionic tensors, choose fermionic quantum-number types such as `qn::fZ2QN`.
- If you only need dense tensors, use a single-sector index with one QN and set the
  degeneracy to the total dimension (for example, `U1QN(0)` or `Z2QN(0)`), and use
  a matching divergence such as `U1QN::Zero()`.

## Next steps

- [Contractions and decompositions](contractions-and-decompositions.md)
- [Indices and quantum numbers (explanation)](../explanation/indices-and-quantum-numbers.md)
