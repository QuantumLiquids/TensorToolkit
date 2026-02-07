# Fermion Design (Z2-Graded Tensor Networks)

This document explains how TensorToolkit models fermionic tensors and how
fermion signs are handled internally.

## Terminology and conventions

- Fermion parity is represented as a Z2 grading on quantum numbers.
- Index direction follows the ket/bra convention: `IN` as ket, `OUT` as bra.
- Divergence is computed by summing QNs with sign: `OUT` contributes `+QN`, `IN`
  contributes `-QN`.

### Divergence (`Div`) in practice

TensorToolkit defines the divergence of a tensor as:

```
Div(T) = (sum of QNs on OUT indices) - (sum of QNs on IN indices)
```

The sum is the group addition for your `QNT` type (for example, U(1) integer
addition and Z2 parity addition).

Example (U(1) charge):

- An `OUT` index sector with `qn = +1` contributes `+1`.
- An `IN` index sector with `qn = +1` contributes `-1`.
- A rank-2 tensor block with those sectors has `Div = +1 - +1 = 0`.

## How fermionic QN types are detected

Fermionic behavior is a compile-time property of the QN type `QNT`.
`Fermionicable<QNT>` treats a QN type as fermionic if it has a callable member
function named `IsFermionParityOdd()`.

For contributors adding a new fermionic QN type, implement:

```cpp
// On the QN type (QNT):
bool IsFermionParityOdd() const;
bool IsFermionParityEven() const;  // recommended (usually `!IsFermionParityOdd()`)
```

Requirements/semantics:

- These functions should be cheap and side-effect free.
- Return `true` for odd-parity (fermionic) sectors and `false` for even sectors.
- If your QN is not fermionic, do not define `IsFermionParityOdd()`; it will be
  treated as bosonic and fermionic code paths will be compiled out.

Common fermionic QNs in `qlten::special_qn`:

- `fZ2QN`
- `fU1QN`
- `fZ2U1QN`

## Which objects are fermionic

`Index<QNT>`, `QNSector<QNT>`, and `QLTensor<ElemT, QNT>` all inherit the
fermionic property of `QNT`. This allows operations to select the fermionic
code path with `if constexpr` at compile time.

## Contraction signs

When contracting tensors, TensorToolkit computes fermionic exchange signs
from the parity information stored per block and the order of contracted legs.
The core algorithm lives in `data_blk_operations.h` and produces a single
sign factor per block before the numerical GEMM.

Practical guidance:

- Always pair `OUT` with `IN` when contracting.
- Let the library handle parity exchange automatically.
- When you reorder tensor legs, remember that swapping two odd-parity legs
  introduces a `-1` sign; TensorToolkit accounts for this based on the current
  index ordering.

## Hermitian conjugate (`Dag`) and `Conj`

`QLTensor::Dag()` flips index directions, applies complex conjugation, and
accounts for fermionic signs. This keeps the result consistent with fermionic
operator ordering without forcing users to manually reorder indices.

`BlockSparseDataTensor::Conj()` is a low-level helper used internally. It
conjugates the raw data and applies a fermionic sign convention, but it does
not flip tensor index directions on its own. Prefer `Dag()` at the tensor level
for physical Hermitian conjugation semantics.

### What the fermionic conjugation sign does (`ReverseSign`)

For fermionic tensors, conjugation also applies a block-wise `±1` sign derived
from the number of odd-parity sectors in that block:

```
ReverseSign = (-1)^(n_odd * (n_odd - 1) / 2)
```

Equivalently, `ReverseSign` is `+1` when `n_odd % 4 ∈ {0, 1}` and `-1` when
`n_odd % 4 ∈ {2, 3}`. This matches the sign you get from reversing the ordering
of `n_odd` fermionic modes (each swap of two odd modes contributes `-1`).

Practical guidance:

- Use `Dag()` for physical Hermitian conjugation (it handles index directions and
  fermionic signs coherently).
- Treat `ReverseSign`/low-level conjugation as an internal convention unless you
  are modifying fermionic algorithms.

## Norms for fermionic tensors

- `Get2Norm()` uses an even-minus-odd (graded) definition. The corresponding
  `Norm^2` can be negative, in which case `Get2Norm()` throws.
- `GetQuasi2Norm()` is always non-negative and is often better for diagnostics.

## Decomposition index directions

Decompositions follow consistent direction rules:

- `U` appends a new `OUT` index
- `S` is an `IN x OUT` diagonal matrix
- `Vt` prepends a new `IN` index

This guarantees that re-contracting the factors reproduces the original tensor
when no truncation is applied. With truncation, the reconstruction is an
approximation controlled by the truncation error.

## Related files

- `qlten/framework/bases/fermionicable.h`
- `qlten/qltensor/blk_spar_data_ten/data_blk_operations.h`
- `qlten/qltensor/special_qn/`
