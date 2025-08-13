### Indices and Quantum Numbers

TensorToolkit supports symmetry-aware tensor networks through quantum-number blocked indices. This tutorial introduces symmetric tensors, the `special_qn` quantum numbers, the legacy `QNVal` route, fermionic behavior, and how to extend a custom `special_qn::QNT`.

### Table of contents

1. What is a symmetric tensor in tensor networks?
2. Core concepts in TensorToolkit
   - Index, direction, and quantum-number sectors
   - Built-in quantum numbers in `special_qn`
   - Legacy `QNVal`-based quantum numbers
3. Fermionic quantum numbers and fermionic tensor networks
4. Using indices and tensors: concise examples
5. Extending with your own quantum number type `special_qn::QNT`

### 1. What is a symmetric tensor in tensor networks?

In many-body physics, local and global symmetries restrict allowed states and transitions. A symmetric tensor respects these selection rules by organizing its data into block-sparse form labeled by quantum numbers (QNs). Each index is decomposed into quantum-number sectors, and only blocks that satisfy a conservation law (divergence) are stored. This yields large memory/time savings and enforces exact symmetry conservation.

Physically, examples include:
- U(1) symmetries: particle number, Sz magnetization
- Z2 symmetries: parity, spin-flip
- Combined symmetries: direct products like Z2×U1

### 2. Core concepts in TensorToolkit

#### 2.1 Index, direction, and quantum-number sectors

APIs live under `qlten::`. An index is a linear space assembled from quantum-number sectors with an orientation.

- Index direction `IN` vs `OUT` controls sign when computing block divergences and determines contraction compatibility.
- Each sector is a degenerate subspace labeled by a QN and an integer degeneracy.

Key headers:
- `qlten/qltensor/index.h`: `template<typename QNT> class Index`
- `qlten/qltensor/qnsct.h`: `template<typename QNT> class QNSector`, `QNSectorVec<QNT>`

Minimal types:
- `TenIndexDirType`: `IN`, `OUT`, `NDIR`

```cpp
#include "qlten/qltensor/index.h"
#include "qlten/qltensor/qnsct.h"
#include "qlten/qltensor/special_qn/special_qn_all.h"
using namespace qlten;

// Bosonic U(1) index with two sectors: N=0 (degeneracy 1), N=1 (degeneracy 2)
Index<special_qn::U1QN> u1_out({
    QNSector<special_qn::U1QN>(special_qn::U1QN(0), 1),
    QNSector<special_qn::U1QN>(special_qn::U1QN(1), 2)
}, TenIndexDirType::OUT);

// Flip direction when needed
auto u1_in = InverseIndex(u1_out); // OUT -> IN

// Introspection
size_t dim = u1_out.dim();              // total dimension (sum over sector dims)
auto dir = u1_out.GetDir();             // IN / OUT / NDIR
auto &sectors = u1_out.GetQNScts();     // vector<QNSector<QNT>>
```

Notes:
- For built-in QNs, `QNT::dim()` is 1. A sector’s dimension equals `QNT::dim() * degeneracy`.
- The block divergence for a multi-index object is the signed sum of sector QNs: OUT adds, IN subtracts (see `CalcDiv`).

#### 2.2 Built-in quantum numbers in `special_qn`

Header hub:
- `qlten/qltensor/special_qn/special_qn_all.h`

Common choices:
- Bosonic: `U1QN`, `U1U1QN`, `ZNN` variants
- Fermionic: `fZ2QN` (parity), `fU1QN` (fermionic U1), `fZ2U1QN` (parity × bosonic U1), `fU1U1QN` (fermionic U1 × bosonic U1)

Examples:
```cpp
// Z2 fermion parity index: sectors even(0) and odd(1)
Index<special_qn::fZ2QN> parity_idx({
    QNSector<special_qn::fZ2QN>(special_qn::fZ2QN(0), 1),
    QNSector<special_qn::fZ2QN>(special_qn::fZ2QN(1), 1)
}, TenIndexDirType::OUT);

// U(1) particle number index: N = 0,1,2
Index<special_qn::U1QN> number_idx({
    QNSector<special_qn::U1QN>(special_qn::U1QN(0), 1),
    QNSector<special_qn::U1QN>(special_qn::U1QN(1), 2),
    QNSector<special_qn::U1QN>(special_qn::U1QN(2), 1)
}, TenIndexDirType::OUT);

// Combined symmetry Z2 × U1 (fermionic): (parity, number)
Index<special_qn::fZ2U1QN> combo_idx({
    QNSector<special_qn::fZ2U1QN>(special_qn::fZ2U1QN(0, 0), 1),
    QNSector<special_qn::fZ2U1QN>(special_qn::fZ2U1QN(1, 1), 2)
}, TenIndexDirType::OUT);
```

#### 2.3 Legacy `QNVal`-based quantum numbers

There is an older, more generic abstraction for quantum numbers based on `qlten/qltensor/qn/qnval.h` (e.g., `U1QNVal` in `qnval_u1.h`). It exists for backward compatibility and will be removed in the future. Prefer `special_qn::QNT` classes for new code. If you are migrating, note that some `special_qn` classes include limited compatibility shims to read old data, but new APIs are non-virtual and more efficient.

### 3. Fermionic quantum numbers and fermionic tensor networks

Fermionic behavior is determined entirely by the QN type `QNT` at compile time. If `QNT` provides `IsFermionParityOdd()`/`IsFermionParityEven()`, the class is treated as fermionic via `Fermionicable<QNT>`; otherwise it is bosonic.

Implications:
- Indices `Index<QNT>` and tensors `QLTensor<ElemT, QNT>` become fermionic if and only if `QNT` is fermionic.
- Contractions and permutations automatically account for fermion parity through the block structure and dedicated methods.
- Norms differ: `QLTensor::Get2Norm()` uses even-minus-odd for fermions; `GetQuasi2Norm()` is always positive.
- Operators relevant to fermions: `QLTensor::ActFermionPOps()` can apply parity operators when you permute/cross fermion lines in network manipulations.

Example QNs with fermion parity:
- `special_qn::fZ2QN`: parity 0/1
- `special_qn::fU1QN`: integer-valued with odd/even parity derived from charge mod 2
- `special_qn::fZ2U1QN`: explicit (parity, U1) pair

### 4. Using indices and tensors: concise examples

```cpp
#include "qlten/qltensor/index.h"
#include "qlten/qltensor/qnsct.h"
#include "qlten/qltensor/qltensor.h"
#include "qlten/qltensor/special_qn/special_qn_all.h"
using namespace qlten;

// Build two indices (bosonic U(1))
Index<special_qn::U1QN> i_out({
    QNSector<special_qn::U1QN>(special_qn::U1QN(0), 1),
    QNSector<special_qn::U1QN>(special_qn::U1QN(1), 1)
}, TenIndexDirType::OUT);
Index<special_qn::U1QN> j_in = InverseIndex(i_out);

// Construct a rank-2 tensor with these indices
QLTensor<double, special_qn::U1QN> A({i_out, j_in});

// Fill with random numbers consistent with a target divergence (here 0 charge)
A.Random(special_qn::U1QN::Zero());

// Query divergence
auto div = A.Div(); // U1QN(0) if blocks exist consistently

// Fermionic example: parity × number
Index<special_qn::fZ2U1QN> f_idx({
    QNSector<special_qn::fZ2U1QN>(special_qn::fZ2U1QN(1, 1), 2)
}, TenIndexDirType::OUT);
QLTensor<double, special_qn::fZ2U1QN> F({f_idx, f_idx});
F.Random(special_qn::fZ2U1QN::Zero());
auto n2 = F.Get2Norm();        // even-minus-odd
auto qn = F.GetQuasi2Norm();   // sum all element square
```

### 5. Extending with your own `special_qn::QNT`

To introduce a new symmetry, add a lightweight value type under `qlten::special_qn` that:
- Provides value semantics and hashing
- Implements arithmetic: unary minus, `operator+=`, `operator+`, `operator-`
- Exposes `size_t dim() const` (usually 1 for 1D irreps)
- Supports streaming (`StreamRead/StreamWrite`) and display (`Show`)
- Equality and `Hash()`
- Optional fermion markers `IsFermionParityOdd()`/`IsFermionParityEven()` to turn on fermionic behavior

Skeleton:
```cpp
#include "qlten/framework/vec_hash.h"
#include "qlten/framework/bases/showable.h"

namespace qlten { namespace special_qn {

class MySymQN : public Showable {
 public:
  MySymQN(void) = default;
  explicit MySymQN(int val) : val_(val), hash_(CalcHash_()) {}
  MySymQN(const MySymQN &rhs) : val_(rhs.val_), hash_(rhs.hash_) {}

  MySymQN &operator=(const MySymQN &rhs) { val_ = rhs.val_; hash_ = rhs.hash_; return *this; }

  MySymQN operator-() const { return MySymQN(-val_); }
  MySymQN &operator+=(const MySymQN &rhs) { val_ += rhs.val_; hash_ = CalcHash_(); return *this; }
  MySymQN operator+(const MySymQN &rhs) const { return MySymQN(val_ + rhs.val_); }
  MySymQN operator-(const MySymQN &rhs) const { return *this + (-rhs); }

  bool operator==(const MySymQN &rhs) const { return hash_ == rhs.hash_; }
  bool operator!=(const MySymQN &rhs) const { return !(*this == rhs); }

  size_t dim() const { return 1; }
  size_t Hash() const { return hash_; }

  // Optional: enable fermionic behavior
  // bool IsFermionParityOdd() const { return val_ & 1; }
  // bool IsFermionParityEven() const { return !(IsFermionParityOdd()); }

  void StreamRead(std::istream &is) { is >> val_ >> hash_; }
  void StreamWrite(std::ostream &os) const { os << val_ << "\n" << hash_ << "\n"; }
  void Show(const size_t indent = 0) const override {
    std::cout << IndentPrinter(indent) << "MySymQN: " << val_ << "\n";
  }

  static MySymQN Zero() { return MySymQN(0); }

 private:
  size_t CalcHash_() const {
    size_t h = static_cast<size_t>(val_);
    return _HASH_XXROTATE(h);
  }
  int val_ = 0; size_t hash_ = 0;
};

} } // namespaces
```

After defining the type, use it directly as the `QNT` template argument: `Index<MySymQN>`, `QLTensor<double, MySymQN>`.
