### Tensor Contractions

This tutorial shows how to contract two `QLTensor`s using the standard pointer-based API, and an advanced reference-based variant optimized for specific contiguous-axis patterns.

### What you'll learn

- Basic pointer-based `Contract` for arbitrary axis pairs
- Result index ordering and common patterns (matrix multiply, partial trace)
- Alternative pointer-based signature with separate axis lists
- Advanced reference-based `Contract` with contiguous, ordered axes (performance-oriented)

## 1) Setup

```cpp
#include "qlten/qlten.h"                            // This actually include everything
#include "qlten/tensor_manipulation/ten_ctrct.h"    // But you can find pointer-based API here 
#include "qlten/tensor_manipulation/ten_ctrct_based_mat_trans.h" // or find Advanced API here

using namespace qlten;
using Z2QN = special_qn::Z2QN;
using IndexT = Index<Z2QN>;
using QNSctT = QNSector<Z2QN>;
using DTen   = QLTensor<QLTEN_Double, Z2QN>;
```

## 2) Pointer-based contraction (recommended start)

The simplest and most general form is:

```cpp
template<typename TenElemT, typename QNT>
void Contract(
    const QLTensor<TenElemT, QNT>* A,
    const QLTensor<TenElemT, QNT>* B,
    const std::vector<std::vector<size_t>>& axes,  // {{axes_of_A}, {axes_of_B}}
    QLTensor<TenElemT, QNT>* C                     // must be default (empty)
);
```

Notes
- The indices to be contracted must match pairwise: `A[a_i] == InverseIndex(B[b_i])`.
- The result `C` is created inside and will have all non-contracted indices of `A` first, then those of `B` (in saved order).

### Example 2.1: Matrix multiplication-like contraction

```cpp
// Z2 sectors per leg (degeneracy 1)
Z2QN q0(0), q1(1);
QNSctT s0(q0, 1), s1(q1, 1);

// Define indices: left (IN), mid (OUT), right (OUT)
IndexT idx_l({s0, s1}, TenIndexDirType::IN);
IndexT idx_m({s0, s1}, TenIndexDirType::OUT);
IndexT idx_r({s0, s1}, TenIndexDirType::OUT);

// A(l, m), B(m^†, r) -> C(l, r)
DTen A({idx_l, idx_m});
DTen B({InverseIndex(idx_m), idx_r});
DTen C;  // must be default (empty)

// Initialize (respecting symmetry as needed)
A.Random(Z2QN(0));
B.Random(Z2QN(0));

// Contract A's axis 1 with B's axis 0
Contract(&A, &B, {{1}, {0}}, &C);
// C now has indices [l, r]
```

### Example 2.2: Contract multiple axes at once

```cpp
// A(l, x, y), B(y^†, x^†, r) -> C(l, r) by contracting (x,y)
IndexT idx_x({s0, s1}, TenIndexDirType::OUT);
IndexT idx_y({s0, s1}, TenIndexDirType::OUT);
IndexT idx_r2({s0, s1}, TenIndexDirType::OUT);

DTen A2({idx_l, idx_x, idx_y});
DTen B2({InverseIndex(idx_y), InverseIndex(idx_x), idx_r2});
DTen C2;

A2.Random(Z2QN(0));
B2.Random(Z2QN(0));

// Contract A2 axes {1,2} with B2 axes {1,0}
Contract(&A2, &B2, {{1, 2}, {1, 0}}, &C2);
// C2 has indices [l, r]
```

### Alternative pointer-based signature

You can also pass axis lists separately; this is equivalent to the above.

```cpp
// Signature
template<typename TA, typename TB, typename TC, typename QNT>
void Contract(
    const QLTensor<TA, QNT>* A, const std::vector<size_t>& axesA,
    const QLTensor<TB, QNT>* B, const std::vector<size_t>& axesB,
    QLTensor<TC, QNT>* C);

// Usage (same as Example 2.2):
Contract(&A2, std::vector<size_t>{1, 2}, &B2, std::vector<size_t>{1, 0}, &C2);
```

## 3) Advanced: reference-based contraction (contiguous, ordered axes)

This variant avoids general tensor transpositions by reducing them to fast matrix transpositions under constraints. It can be faster when your contraction axes are contiguous and ordered. Include the header:

```cpp
#include "qlten/tensor_manipulation/ten_ctrct_based_mat_trans.h"
```

Key constraints
- The contracted axes in each tensor must be contiguous and in ascending order (step 1).
- Periodic wrapping is allowed (e.g., for rank-5, `{3,4,0}` is contiguous).

API
```cpp
template<typename TenElemT, typename QNT, bool a_ctrct_tail = true, bool b_ctrct_head = true>
void Contract(
    const QLTensor<TenElemT, QNT>& A,
    const QLTensor<TenElemT, QNT>& B,
    size_t a_ctrct_axes_start,   // start of contiguous block in A
    size_t b_ctrct_axes_start,   // start of contiguous block in B
    size_t ctrct_axes_size,      // number of axes to contract
    QLTensor<TenElemT, QNT>& C   // must be default (empty)
);
```

### Example 3.1: Tail–head contiguous contraction

```cpp
// A(l, x, y), B(x^†, y^†, r) with contiguous blocks (x,y) in A and (x^†,y^†) in B
DTen A3({idx_l, idx_x, idx_y});
DTen B3({InverseIndex(idx_x), InverseIndex(idx_y), idx_r});
DTen C3;

A3.Random(Z2QN(0));
B3.Random(Z2QN(0));

// Contract A3 axes starting at 1 (x,y) with B3 axes starting at 0 (x^†,y^†), size=2
// Default template params a_ctrct_tail=true, b_ctrct_head=true match this pattern
Contract(A3, B3, /*a_start=*/1, /*b_start=*/0, /*size=*/2, C3);
// C3 has indices [l, r]
```

When to use advanced API
- Prefer the pointer-based API for generality and clarity.
- Use the reference-based API when your contraction axes are contiguous and ordered; it may reduce overhead by using matrix-transpose-based preparation.

Remarks
- In all APIs, `C` must be default-constructed (empty) before the call.
- Fermionic signs are handled internally according to index directions; follow IN/OUT conventions from earlier tutorials.

—

Next: Matrix Decomposition.
